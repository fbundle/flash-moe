#!/usr/bin/env python3

from typing import Iterator
import numpy as np
from transformers import AutoTokenizer
from moe_infer import Model, Engine, Cache  # type: ignore

def get_qwen3_response(completion: str) -> str:
    completion = completion.removesuffix("<|im_end|>")
    completion = completion.split("</think>")[-1]
    return completion
    

class Conversation:
    def __init__(self, tokenizer_path: str, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = Model(model_path)
        
        self.engine = Engine(self.model, pipeline_mode="FusedExp")
        self.cache = Cache(self.model)
        
        self.messages = []
    
    def chat(self, message: str) -> str:
        self.messages.append({
            "role": "user",
            "content": message,
        })

        input_ids = np.array(self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, enable_thinking=False).input_ids, dtype=np.int64)

        completion = ""
        completion_ids = []

        for token, logits in self.engine.stream_generate(input_ids, self.cache):
            completion_ids.append(token)

            new_completion = self.tokenizer.decode(completion_ids)

            addon_completion = new_completion[len(completion):]
            print(addon_completion, end="", flush=True)
            completion = new_completion

        response = get_qwen3_response(completion)

        self.messages.append({"role": "assistant", "content": response})

        return response

hf_path = "hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
path = "data/models--mlx-community--Qwen3.6-35B-A3B-4bit"

c = Conversation(hf_path, path)

while True:
    message = input("> ")
    _ = c.chat(message)
    
    print()
    print(c.engine.telemetry())


