import json
import time
import warnings
import os

# Set the warning to appear only once
warnings.filterwarnings("once", category=UserWarning)

class GPTAgent:
    
    def __init__(self, model_id="google/gemma-7b-it", cuda_gpus=None, device_map="auto"):

        if cuda_gpus is not None:
            # Specify which GPUs to make visible to PyTorch
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_gpus

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    
        # Load the Gemma model
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16
        )

    def num_tokens(self, prompt):
        if isinstance(prompt, str):
            messages = [
                {"role": "user", "content": prompt}
            ]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            print("Prompt is not a list or str, breaking")
            return None

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return len(self.tokenizer.tokenize(prompt_text))

    def ask_gpt(self, prompt, temp=None, max_tokens=None, print_response=True, sleep=None):
        
        if sleep:
            time.sleep(sleep)
        
        if isinstance(prompt, str):
            messages = [
                {"role": "user", "content": prompt}
            ]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            print("Prompt is not a list or str, breaking")
            return None

        tokenized_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer.encode(tokenized_prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs.to(self.model.device),
            max_new_tokens=max_tokens,
            # do_sample=True,
            temperature=temp
        )

        response_content = self.tokenizer.decode(outputs[0])

        if print_response:
            print("\n", response_content, "\n")
        
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
