# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:56:54 2024

@author: klange01
"""

import tiktoken
from openai import AzureOpenAI
import time
from datetime import datetime
import pandas as pd
import json

class GPTAgent:
    
    def __init__(self):
        
        self.model_token_limits = {
            'gpt4model': 8192,  # Add other models and their token limits as needed
            # 'other-model': token_limit,
        }
        
        # gpt-4-8k-daal-sandbox-eus-01-01
        # gpt-4-32k-daal-sandbox-eus-01-02
        # gpt-35turbo16k-daal-sandbox-eus-01-01

        

        # Setting the API key from file and other related definitions
        with open(
                r'/azure openai.txt', 'r') as file:
            lines = file.readlines()
        api_key = lines[0].strip()
        endpoint = lines[1].strip()

        self.client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-02-15-preview"
        )



    def check_model (self, model_name):

        if model_name not in self.model_token_limits:
            raise ValueError (
                f"Model '{model_name}' not recognized. Please add it to model_token_limits.")


    def num_tokens(self, string: str) -> int:

        encoding = tiktoken.encoding_for_model ("gpt-4")
        return len (encoding.encode(string))


    def reduce_tokens(self, string: str, limit_tokens = None) -> str:

        if limit_tokens == None:
            limit_tokens = self.model_token_limits

        while self.num_tokens(string) > limit_tokens:
            
            string = '. '.join(string.split('. ')[:-1])
        
        return string


    def ask_gpt (self, prompt, temp=None, 
                max_tokens=None, 
                model='gpt-4-8k-daal-sandbox-eus-01-01', 
                print_response = True, 
                sleep = None, 
                save_file = None,
                messages = None):
        
        if sleep:
            time.sleep (sleep)
        
        if type (prompt) == str:
            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}]
        elif type (prompt) == list:
            messages = prompt
        else:
            print ("Prompt is not a list or str, breaking")
            return None
            
        completion = self.client.chat.completions.create(
          model = model, # model = "deployment_name"
          messages = messages,
          temperature=temp,
          max_tokens=max_tokens,
          # top_p=0.95,
          # frequency_penalty=0,
          # presence_penalty=0,
          # stop=None
        )
        
        response_content = completion.choices[0].message.content
        
        if print_response: print (
                "\n", response_content, "\n")
        
        df_one_note = pd.DataFrame (
            {"time": [str (datetime.now().time())],
             "model": [str (model)],
             "temperature": [str (temp)],
             "max_tokens": [str (max_tokens)],
             "messages": [str (messages)], 
             "completion": [str (completion)]})
        
        if save_file:
            df_one_note.to_csv (save_file)
        else:
            time_now = str (datetime.now().time())
            time_now = time_now.replace (":", ".")
            df_one_note.to_csv (
                f"d:/GPT_backup/GPT_{time_now}.csv")
        
        return response_content
    

if __name__ == "__main__":
    
    
    gpt_agent = GPTAgent()
    
    response = gpt_agent.ask_gpt (
        "Hello, please tell the world's shortest joke" 
        )
    
    
    