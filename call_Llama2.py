# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:26:31 2024

@author: klange01
"""

import tiktoken
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class GPTAgent:


    def __init__(self, model_id="meta-llama/Llama-2-70b-chat-hf"):
        # model_id="meta-llama/Llama-2-7b-chat-hf"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    
        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            add_bos_token=True, 
            trust_remote_code=True
        )


    def num_tokens(self, string: str) -> int:
        # Use Mixtral tokenizer to calculate token count
        return len(self.tokenizer.encode(string))


    def ask_gpt(self, prompt, temperature=0.01, max_new_tokens=8192, print_response=False, sleep=None):
        eval_prompt = f"[INST] {prompt} [/INST]"
        
        model_input = self.eval_tokenizer(
            eval_prompt, 
            return_tensors="pt"
        ).to("cuda")
        
        self.model.eval()
        
        with torch.no_grad():
            response = self.eval_tokenizer.decode(
                self.model.generate(
                    **model_input, max_new_tokens=max_new_tokens
                )[0], 
                skip_special_tokens=True
            )
            
            if print_response:
                print(response)
            
            response_content = response[
                response.find("[/INST]") + len("[/INST]"):]
        
        return response_content