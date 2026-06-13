#!/usr/bin/env python3
import argparse
import json
import sqlite3
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Export local proxy database calls to standard formats."
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
        help="Export format: 'sharegpt' (standard ShareGPT conversations), 'openai' (standard messages), or 'jsonl' (raw dump)"
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
    
    query = "SELECT timestamp, model, system, messages, messages_flat, conversations, tools, has_tool_calls FROM dataset_calls"
    where_clauses = []
    params = []

    if args.model:
        where_clauses.append("model = ?")
        params.append(args.model)
    if args.has_tools:
        where_clauses.append("has_tool_calls = 1")

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

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
        timestamp, model, system, messages_json, messages_flat_json, conversations_json, tools_json, has_tool_calls = row

        messages = json.loads(messages_json)
        messages_flat = json.loads(messages_flat_json)
        conversations = json.loads(conversations_json)
        tools = json.loads(tools_json) if tools_json else None

        if args.min_turns:
            non_system_turns = len([m for m in conversations if m.get("from") in ["human", "gpt"]])
            if non_system_turns < args.min_turns:
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
        print(f"\nTo load this dataset in Python:", file=info_dest)
        print(f"""```python
from datasets import load_dataset

# 1. Load the exported dataset
dataset = load_dataset("json", data_files="{output_path or 'dataset_sharegpt.json'}")

# 2. Access conversations
for entry in dataset["train"]:
    conversations = entry["conversations"]
    for msg in conversations:
        print(f"[{{msg['from']}}] {{msg['value'][:100]}}...")
```""", file=info_dest)
    elif output_format == "openai":
        print(f"\nTo load this dataset in Python:", file=info_dest)
        print(f"""```python
from datasets import load_dataset

# 1. Load the exported dataset
dataset = load_dataset("json", data_files="{output_path or 'dataset_openai.json'}")

# 2. Access messages
for entry in dataset["train"]:
    messages = entry["messages"]
    for msg in messages:
        print(f"[{{msg['role']}}] {{msg['content'][:100]}}...")
```""", file=info_dest)
    elif output_format == "jsonl":
        print(f"\nTo load this dataset in Python:", file=info_dest)
        print(f"""```python
from datasets import load_dataset

# 1. Load the exported JSONL dataset
dataset = load_dataset("json", data_files="{output_path or 'dataset_jsonl.jsonl'}")

# 2. Access parsed formats
for entry in dataset["train"]:
    # Can access entry['conversations'] (ShareGPT) or entry['messages_flat'] (OpenAI)
    print(entry["conversations"])
```""", file=info_dest)

if __name__ == "__main__":
    main()
