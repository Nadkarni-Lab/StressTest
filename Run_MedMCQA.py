import os

import pandas as pd

from datasets import load_dataset

ds = load_dataset("openlifescienceai/medmcqa")

df = pd.DataFrame (ds ["train"])

df = df [df ["choice_type"] == "single"].reset_index (drop = True)

df ["raw_question"] = df ["question"]

df ["question"] = df.apply (lambda row:f"""
{row ["raw_question"]}
A: {row ["opa"]}
B: {row ["opb"]}
C: {row ["opc"]}
D: {row ["opd"]}""", axis = 1)

df ["answer"] = df ["cop"].map ({0:"A", 1:"B", 2:"C", 3:"D"})

import random

random.seed(42)

numbers = list(range(0,df.shape [0]))


from call_Llama3 import GPTAgent

# model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

model = "Llama-3-70B"

gptagent = GPTAgent (model_id = model_id, cuda_gpus = "0,1")


def check_answers (df, questions, answer):
    
    json = gptagent.flex_json (answer)
    
    s = 0
    
    for i, item in enumerate (json):
        
        qst_num = int (item ["question"])
        
        single = item ["answer"]
        
        if qst_num == questions [i]:
            
            if single == df ["answer"][qst_num]:
                
                print (i, item, df ["answer"][qst_num])
                
                s = s + 1
        
    return s


def func_prompt (questions):
    
    concatenated = ""
    
    for q in questions:
        
        concatenated = concatenated + f"""Question number: {q}
Question: {df ["question"][q]}     
"""
        
    prompt = f"""
Following are single choice medical MCQ questions with their context:
{concatenated}

Please return A/B/C/D for each question in JSON format:

[{{"question": "<question_number>", "answer": "<A,B,C,D>"}},...]
"""    
    return prompt

"""
random_list = random.sample (numbers, 50)

prompt = func_prompt (random_list)

print (gptagent.num_tokens (prompt))

raw_json = gptagent.ask_gpt (prompt, 
                  max_tokens = 8192)

print (check_answers (df, random_list, raw_json))
"""


for test_size in [75, 50, 25, 5]:
    
    for experiment in range (50):
        
        if os.path.isfile(
            f"17.8.24/{model}_{test_size}_{experiment}.pkl"):

            print (f"17.8.24/{model}_{test_size}_{experiment}.pkl exits, moving on")
            
        else:
    
            random_list = random.sample (numbers, test_size)

            prompt = func_prompt (random_list)

            print ("num tokens:", gptagent.num_tokens (prompt))

            raw_json = gptagent.ask_gpt (prompt, max_tokens = 8192)

            try:
                print (f"Check answers {model}_{test_size}_{experiment}:", 
                       check_answers (df, random_list, raw_json))
            except:
                print (f"{model}_{test_size}_{experiment}: Failed JSON")

            df_temp = pd.DataFrame ({"model":[model],
                                    "test_size":[test_size],
                                    "experiment":[experiment],
                                    "questions":[random_list],
                                    "prompt":[prompt],
                                    "raw_json":[raw_json],})

            df_temp.to_pickle (
                f"17.8.24/{model}_{test_size}_{experiment}.pkl")
        
