#!/usr/bin/env python3

import os
import sys
from typing import Iterator
import numpy as np
from transformers import AutoTokenizer
from moe_infer import Context, Cache # type: ignore

class Conversation:
    def __init__(self, tokenizer_path: str, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
        self.context = Context()
        self.context.load_model(model_path, pipeline_mode="FusedWoods")
        
        self.messages = []
        self.cache = self.context.new_cache()
    
    def __del__(self):
        self.context.unload_model()
        
    
    def chat(self, message: str) -> Iterator[str]:
        self.messages.append({
            "role": "user",
            "content": message,
        })

        input_ids = np.array(self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, enable_thinking=False).input_ids)
        input_ids = input_ids[self.cache.pos:] # get new ids

        completion = ""
        completion_ids = []
        for token, logits in self.context.stream_generate(input_ids, self.cache):
            completion_ids.append(token)
            
            new_completion = self.tokenizer.decode(completion_ids)
            yield new_completion[len(completion):]
            completion = new_completion

hf_path = "hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
path = "data/models--mlx-community--Qwen3.6-35B-A3B-4bit"

c = Conversation(hf_path, path)

while True:
    message = input("> ")
    for e in c.chat("hello"):
        print(e, end="", flush=True)
    
    print()
    print(c.context.telemetry())


