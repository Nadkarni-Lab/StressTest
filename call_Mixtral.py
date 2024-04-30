# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:26:31 2024

@author: klange01
"""

import tiktoken
from transformers import AutoTokenizer, pipeline
import transformers
import torch


class GPTAgent:


    def __init__(self):

        self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.float16, 
                          "load_in_4bit": True},
        )


    def num_tokens(self, string: str) -> int:
        # Use Mixtral tokenizer to calculate token count
        return len(self.tokenizer.encode(string))


    def ask_gpt(self, prompt, temperature=0.01, max_new_tokens=8192,
                print_response=False, sleep=None):

        # Prepare the prompt according to Mixtral's requirements

        messages = [{"role": "user", "content": prompt}]

        prompt = self.tokenizer.apply_chat_template (
            messages, tokenize=False, add_generation_prompt=True)

        # Perform inference
        outputs = self.pipeline(prompt, max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature,
                                pad_token_id=self.tokenizer.eos_token_id)

        response_content = outputs[0]["generated_text"]

        response_content = response_content [
            response_content.find ("[/INST]") + len ("[/INST]"):]

        if print_response:
            print(response_content)

        return response_content