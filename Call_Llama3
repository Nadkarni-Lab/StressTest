import json
import time

import warnings

from transformers import AutoTokenizer

# Set the warning to appear only once
warnings.filterwarnings("once", category=UserWarning)


import os



class GPTAgent:
    
    def __init__(self, model_id = "meta-llama/Meta-Llama-3-70B-Instruct", cuda_gpus = None,
                 device_map = "auto"):
        
        if cuda_gpus != None:
            # Specify which GPUs to make visible to PyTorch
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_gpus

        import transformers
        import torch
    
        self.model_id = model_id

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map = device_map)
        
        # Load the tokenizer associated with Llama-3
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    
    def num_tokens (self, prompt):
        
        if type (prompt) == str:
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},]        
        elif type (prompt) == list:
            messages = prompt
        else:
            print ("Prompt is not a list or str, breaking")
            return None
                    
        return len (self.tokenizer.tokenize(f"{messages}"))

        
    def ask_gpt (self, prompt, 
                 temp=None, 
                 max_tokens=None, 
                 print_response = True, 
                 sleep = None, 
                 save_file = None,
                 messages = None):
        
        if sleep:
            time.sleep (sleep)
            
        if type (prompt) == str:
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},]        
        elif type (prompt) == list:
            messages = prompt
        else:
            print ("Prompt is not a list or str, breaking")
            return None
            
        tokenized_prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            tokenized_prompt,
            max_length=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temp,
            # top_p=0.9,
        )
        
        response_content = outputs[0]["generated_text"][len(tokenized_prompt):]
        
        if print_response: print (
                "\n", response_content, "\n")
                
        return response_content
    
    
    def flex_json (self, x):
        """
        Tries to parse a string as JSON. If it fails, makes two more attempts:
        1. Extract and parse a JSON-like substring.
        2. Replace single quotes with double and try parsing again.
        """
        
        try:
            return json.loads(x)
        
        except json.JSONDecodeError:
            pass  # Proceed to next attempt
        
        if x.find ("[") >= 0 and x.find ("[") < x.find ("{"):
            
            start, end = x.find("["), x.rfind("]") + 1
        
        else:
            
            start, end = x.find("{"), x.rfind("}") + 1
        
        try:    
            if start >= 0 and end > 0:  # Ensure indices are valid
                return json.loads(x[start:end])
        
        except json.JSONDecodeError:
            pass  # Proceed to final attempt
        
        try:
            if start >= 0 and end > 0:  # Ensure indices are valid
                x = x.replace("'", '"')
                return json.loads(x[start:end])  # Reuse start/end from previous attempt
        
        except json.JSONDecodeError as e:
            print ("JSON failed:", e)  # All attempts failed
    
            return "Fail"  # Optionally, handle failure to parse the string
