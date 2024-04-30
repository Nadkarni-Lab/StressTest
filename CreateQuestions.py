# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:29:58 2024

@author: klange01
"""

import pandas as pd

from ETL import ETL

from call_GPT import GPTAgent

from Prompts import Prompts


if __name__ == "__main__":
    
    etl = ETL ()
    
    gptagent = GPTAgent ()
    
    prompts = Prompts ()
    
    
    df_notes, df_EncounterFact, df_PatientDim = etl.run ()
    
    df_notes.rename (inplace = True, columns = {"Clean_Text": 
                                                "Text"})

        
    # limit to 200 tokesn and up notes
    
    df_notes ["tokens"] = df_notes ["Text"].apply (
        lambda x: gptagent.num_tokens (x))
    
    
    df_notes = df_notes [df_notes ["tokens"] >= 200]
    
    print ("Limit to 200 tokens notes:", df_notes.shape)


    
        
    ####    

    def create_question_answer(note, questions=None):
        category_descriptions = {
            'FB': 'Fact-based questions requiring a direct answer from the text.',
            'TMP': 'Temporal questions related to the timing of events.',
            'NUM': 'Numerical questions asking for specific quantities.'
        }
    
        category_prompt = '\n'.join(
            [f"- {code}: {desc}" for code, desc in category_descriptions.items()])
    
        return f"""
    Given the following clinical note, generate one unequivocal question based on the information provided in the note, and categorize the type of question using the categories provided below:
    
    {category_prompt}
    
    Then, provide an answer to that question in the most concise form possible, adhering to the following guidelines:
    - If the answer consists of multiple items (e.g., symptoms, medications, findings), list the items separated by commas, exactly as mentioned in the note, without introductory phrases.
    - If the answer is a single fact or statement, copy it verbatim from the note without alteration or introduction.
    - Ensure all responses are factual, direct, and devoid of any additional context or commentary.
    
    Please return the output in JSON format with keys "question", "answer", and "category", ensuring the answer is presented according to these instructions.
    
    For example:
    
    {{
      "question": "Identify the blood pressure readings recorded during the visit.",
      "answer": "120/80 mmHg, 118/78 mmHg",
      "category": "NUM"
    }}
    
    {{
      "question": "What diagnosis was given for the patient's respiratory symptoms?",
      "answer": "chronic obstructive pulmonary disease",
      "category": "FB"
    }}
    
    {{
      "question": "What dietary advice was recommended to the patient?",
      "answer": "increase water intake, reduce sodium intake",
      "category": "FB"
    }}
    
    {{
      "question": "What is the duration of the prescribed antibiotic course?",
      "answer": "10 days",
      "category": "TMP"
    }}
    
    {{
      "question": "How does the patient describe their pain on a scale of 1 to 10?",
      "answer": "7",
      "category": "NUM"
    }}
    
    The question must be contextually different from the ones listed below:
    
    #### Previous questions:
    
    {questions if questions else 'None provided.'}
    
    #### Clinical Note:
    
    {note}
    """
    
    
    import json
    
    def extract_info(gpt_response):
        # Assuming gpt_response is a JSON string with "question", "answer", "category"
        try:
            # If the response is already a valid JSON string
            parsed_response = json.loads(gpt_response)
            question = parsed_response.get("question", "")
            answer = parsed_response.get("answer", "")
            category = parsed_response.get("category", "")

            return {
                "gpt4_response": gpt_response,
                "question": question,
                "answer": answer,
                "category": category
            }


        except json.JSONDecodeError:
            
            print (json.JSONDecodeError)
                        
            return {
                "gpt4_response" : gpt_response,
                "question": "Failed JSON",
                "answer": "Failed JSON",
                "category": "Failed JSON"
            }

    
    def generate_qa_pairs(note, num_pairs=20):
        
        questions_accumulated = ""
        
        qa_pairs = []
    
        for _ in range (num_pairs):
            
            # Generate the prompt with the current note and accumulated questions for uniqueness
            prompt = create_question_answer(note, 
                                            questions_accumulated)  # Assume create_question_answer is defined elsewhere
            
            # Simulate asking GPT and getting a response (you'll replace this with the actual call to GPT)
            gpt_response = gptagent.ask_gpt (prompt)  # Assume ask_gpt is a function you've defined to interact with GPT
            
            # Extract question, answer, category from GPT response
            extracted_info = extract_info (gpt_response)
            
            # Update the accumulated questions string for the next iteration
            questions_accumulated += f"{extracted_info ['question']}\n"
            
            # Append the extracted info to the qa_pairs list
            qa_pairs.append (extracted_info)
        
        return qa_pairs
    
    
    
    import os
 
    def save_row_to_file (i, row, 
                          directory=""):
        
        """Save each row to a separate file in the specified directory."""
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
               
        # Define file path using ClinicalNoteKey and timestamp for uniqueness
        file_path = os.path.join(
            directory, f"{row['ClinicalNoteKey']}_{i}.csv")
         
        # Convert row to DataFrame
        df_row = pd.DataFrame([row])
        
        # Save DataFrame to file
        df_row.to_csv(file_path, index=False)
        

        
    expanded_rows = []

    df_notes_500 = df_notes [
        df_notes ["tokens"].apply (
            lambda x: x>=400 and x<500)].reset_index (drop = True)

    df_notes_500 = df_notes_500.sample (frac = 1.0)

    for _, row in df_notes_500 [0:200].iterrows():
        
        qa_pairs = generate_qa_pairs (row['Text'], num_pairs=25)
        
        for i, pair in enumerate (qa_pairs):

            expanded_row = {
                "ClinicalNoteKey": row['ClinicalNoteKey'],
                "Text": row ["Text"],
                "GPT-4_response": pair ["gpt4_response"],
                "Question": pair ['question'],
                "Answer": pair ['answer'],
                "Category": pair ['category']
            }
    
            save_row_to_file (i, expanded_row)
            
            expanded_rows.append (expanded_row)
    
    # Convert the expanded list of rows into a DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    
    expanded_df.to_excel ("output.xlsx")

    
    
    