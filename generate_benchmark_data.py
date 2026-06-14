import os
import requests

# Configuration
TOKENIZE_URL = "http://localhost:8089/tokenize"
OUTPUT_DIR = "benchmark_data"
NEEDLE = "\n[SYSTEM_NOTE: The unique verification key for this benchmark is 'KEY-256K-ALPHA'.]\n"

# Target context sizes in tokens. Override with SIZES=512,2048,8192,... (env).
# Default spans a speed-vs-context curve from short prompts to long context.
_DEFAULT_SIZES = [512, 2048, 8192, 32768, 65536, 131072]
TARGET_SIZES = (
    [int(x) for x in os.environ["SIZES"].split(",")]
    if os.environ.get("SIZES")
    else _DEFAULT_SIZES
)

# Fallback ratio measured on this filler text with the gemma tokenizer
CHARS_PER_TOKEN = 4.88

BASE_TEXT = """
Documentation for Module Alpha:
This module handles the primary data ingestion pipeline. It supports multiple
formats including CSV, JSON, and Parquet. The pipeline is designed for
high throughput and low latency.

Technical Specifications:
- Max throughput: 5000 events/sec
- Latency: < 200ms
- Supported protocols: gRPC, REST

Maintenance Notes:
Ensure that the memory limits are adjusted based on the volume of incoming
data. Monitor the buffer usage closely to prevent overflows in high-traffic
scenarios.
"""


def count_tokens(text):
    """Exact token count via the server's /tokenize endpoint, or None if unreachable."""
    try:
        response = requests.post(TOKENIZE_URL, json={"content": text}, timeout=120)
        response.raise_for_status()
        return len(response.json()["tokens"])
    except Exception:
        return None


def build_text(target_tokens):
    """Build filler text of ~target_tokens tokens with the needle at ~50% depth."""
    chunk = " ".join(BASE_TEXT.split()) + "\n"

    # Initial build using the chars/token estimate, slightly under target
    target_chars = int(target_tokens * CHARS_PER_TOKEN * 0.97)
    n_chunks = max(1, target_chars // len(chunk))
    chunks = [chunk] * n_chunks
    chunks.insert(n_chunks // 2, NEEDLE)
    text = "".join(chunks)

    tokens = count_tokens(text)
    if tokens is None:
        print("  ! /tokenize unreachable, using chars/token estimate only")
        return text

    # Grow or shrink in whole chunks until within 1% below the target
    chunk_tokens = max(1, count_tokens(chunk) or 80)
    while tokens > target_tokens:
        remove = max(1, (tokens - target_tokens) // chunk_tokens)
        n_chunks -= remove
        chunks = [chunk] * n_chunks
        chunks.insert(n_chunks // 2, NEEDLE)
        text = "".join(chunks)
        tokens = count_tokens(text)
    while tokens < int(target_tokens * 0.99):
        add = max(1, (int(target_tokens * 0.995) - tokens) // chunk_tokens)
        n_chunks += add
        chunks = [chunk] * n_chunks
        chunks.insert(n_chunks // 2, NEEDLE)
        text = "".join(chunks)
        tokens = count_tokens(text)

    print(f"  - {tokens} tokens (target {target_tokens})")
    return text


def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating datasets in {OUTPUT_DIR}...")

    for size in TARGET_SIZES:
        file_path = os.path.join(OUTPUT_DIR, f"needle_{size}_tokens.txt")
        if os.path.exists(file_path):
            print(f"Skipping (exists): {file_path}")
            continue
        print(f"Generating: {file_path}")
        text = build_text(size)
        with open(file_path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    generate_dataset()
