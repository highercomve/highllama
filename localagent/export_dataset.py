#!/usr/bin/env python3
import argparse
import json
import sqlite3
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Export Claude Local Proxy database calls to formats ready for Unsloth / Hugging Face training."
    )
    parser.add_argument(
        "--db-path",
        default=os.path.expanduser("~/.local/state/localagent/dataset.db"),
        help="Path to the SQLite database (default: ~/.local/state/localagent/dataset.db)"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["sharegpt", "openai", "jsonl"],
        default="sharegpt",
        help="Export format: 'sharegpt' (for Unsloth / LLaMA-Factory), 'openai' (standard messages), or 'jsonl' (raw dump)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: dataset_<format>.[json/jsonl])"
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Filter by model name (optional)"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of exported entries (oldest first)"
    )
    parser.add_argument(
        "--latest",
        type=int,
        help="Export only the N latest entries (newest first)"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=0,
        help="Filter: Only export conversations with at least N turns (default: 0)"
    )
    parser.add_argument(
        "--has-tools",
        action="store_true",
        help="Filter: Only export conversations that contain at least one tool call"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found at {args.db_path}", file=sys.stderr)
        print("Make sure the local proxy has processed and logged at least one remote model call.", file=sys.stderr)
        sys.exit(1)
        
    db_path = args.db_path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT timestamp, model, system, messages, messages_flat, conversations, tools FROM dataset_calls"
    params = []
    
    if args.model:
        query += " WHERE model = ?"
        params.append(args.model)
        
    if args.latest:
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(args.latest)
    else:
        query += " ORDER BY timestamp ASC"
        if args.limit:
            query += " LIMIT ?"
            params.append(args.limit)
        
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if args.latest:
        rows = rows[::-1]
    
    if not rows:
        print("No entries found in database matching criteria.", file=sys.stderr)
        sys.exit(0)
        
    output_format = args.format
    output_path = args.output
        
    exported_data = []
    
    for row in rows:
        timestamp, model, system, messages_json, messages_flat_json, conversations_json, tools_json = row
        
        messages = json.loads(messages_json)
        messages_flat = json.loads(messages_flat_json)
        conversations = json.loads(conversations_json)
        tools = json.loads(tools_json) if tools_json else None
        
        # Apply filters
        if args.min_turns:
            non_system_turns = len([m for m in conversations if m.get("from") in ["human", "gpt"]])
            if non_system_turns < args.min_turns:
                continue
                
        if args.has_tools:
            has_tool_call = False
            for m in conversations:
                val = m.get("value", "")
                if "<tool_call>" in val or "tool_use" in val:
                    has_tool_call = True
                    break
            if not has_tool_call:
                continue
        
        if output_format == "sharegpt":
            item = {"conversations": conversations}
            if system:
                item["system"] = system
            if tools:
                item["tools"] = tools
            exported_data.append(item)
        elif output_format == "openai":
            item = {"messages": messages_flat}
            if tools:
                item["tools"] = tools
            exported_data.append(item)
        elif output_format == "jsonl":
            item = {
                "timestamp": timestamp,
                "model": model,
                "system": system,
                "messages": messages,
                "messages_flat": messages_flat,
                "conversations": conversations
            }
            if tools:
                item["tools"] = tools
            exported_data.append(item)
            
    # Write output
    if output_path:
        # Write to file
        if output_format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in exported_data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(exported_data, f, indent=2, ensure_ascii=False)
                
        info_dest = sys.stdout
        print(f"Successfully exported {len(exported_data)} entries to: {output_path}", file=info_dest)
    else:
        # Write to stdout
        if output_format == "jsonl":
            for item in exported_data:
                sys.stdout.write(json.dumps(item) + "\n")
        else:
            json.dump(exported_data, sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
        sys.stdout.flush()
        
        info_dest = sys.stderr
        print(f"Successfully exported {len(exported_data)} entries to stdout", file=info_dest)
            
    print(f"Format: {output_format.upper()}", file=info_dest)
    if output_format == "sharegpt":
        print(f"\nTo load this dataset in your Unsloth training script:", file=info_dest)
        print(f"""```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# 1. Load the exported dataset
dataset = load_dataset("json", data_files="{output_path or 'dataset_sharegpt.json'}")

# 2. Format it using get_chat_template (e.g. for LLaMA-3)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
    mapping = {{"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}},
)

def format_prompts(examples):
    convs = examples["conversations"]
    texts = [tokenizer.apply_chat_template(c, tokenize=False) for c in convs]
    return {{"text": texts}}

dataset = dataset.map(format_prompts, batched=True)
```""", file=info_dest)
    elif output_format == "openai":
        print(f"\nTo load this dataset in your Unsloth training script:", file=info_dest)
        print(f"""```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# 1. Load the exported dataset
dataset = load_dataset("json", data_files="{output_path or 'dataset_openai.json'}")

# 2. Format it using standard messages template (e.g. for LLaMA-3)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
    mapping = {{"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}},
)

def format_prompts(examples):
    msgs = examples["messages"]
    texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in msgs]
    return {{"text": texts}}

dataset = dataset.map(format_prompts, batched=True)
```""", file=info_dest)
    elif output_format == "jsonl":
        print(f"\nTo load this dataset in your Unsloth training script:", file=info_dest)
        print(f"""```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# 1. Load the exported JSONL dataset
dataset = load_dataset("json", data_files="{output_path or 'dataset_jsonl.jsonl'}")

# 2. Since JSONL contains all formats, you can map either ShareGPT
#    (conversations) or OpenAI (messages_flat). Here is the ShareGPT mapping:
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
    mapping = {{"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}},
)

def format_prompts(examples):
    convs = examples["conversations"]
    texts = [tokenizer.apply_chat_template(c, tokenize=False) for c in convs]
    return {{"text": texts}}

dataset = dataset.map(format_prompts, batched=True)
```""", file=info_dest)

if __name__ == "__main__":
    main()
