#!/usr/bin/env python3
"""Simple chat client for Flash-MoE Rust inference server."""
import argparse
import sys
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Chat with Flash-MoE server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="models--mlx-community--Qwen3.5-35B-A3B-4bit")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="local")

    print(f"model: {args.model}  port: {args.port}\n")

    messages = []
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/clear":
            messages.clear()
            print("[cleared]\n")
            continue

        messages.append({"role": "user", "content": line})

        stream = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=4096,
            stream=True,
        )

        response_parts = []
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                response_parts.append(delta.content)

        sys.stdout.write("\n\n")
        if response_parts:
            messages.append({"role": "assistant", "content": "".join(response_parts)})


if __name__ == "__main__":
    main()
