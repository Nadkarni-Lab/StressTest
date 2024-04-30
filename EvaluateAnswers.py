# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:50:46 2024

@author: klange01
"""


import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import json


import sys
import_folder_path = r"code"
sys.path.append (import_folder_path)

from call_GPT_research import GPTAgent

gptagent = GPTAgent ()





def sample_questions(df, n_questions_per_note):
    """
    Samples questions for each ClinicalNoteKey in the DataFrame. It samples without replacement
    if n_questions_per_note <= 10, ensuring uniqueness. If n_questions_per_note > 10,
    it allows repetition to meet the required count.

    :param df: DataFrame with 'ClinicalNoteKey', 'Note', and 'Question' columns.
    :param n_questions_per_note: Desired number of questions to sample per ClinicalNoteKey.
    :return: DataFrame containing the sampled questions.
    """
    # Function to handle sampling logic per group
    def sample_group(group, n):
        if n <= len(group):
            return group.sample(n=n, replace=False)
        else:
            # Sample up to the group size without replacement, then sample additional with replacement
            return pd.concat([group.sample(n=len(group), replace=False),
                              group.sample(n=n - len(group), replace=True)])
    
    # Apply the sampling function to each group
    sampled = df.groupby('ClinicalNoteKey').apply(sample_group, n_questions_per_note).reset_index(drop=True)
    
    return sampled


def prepare_prompt_and_df (df, n_notes=10, 
                           n_questions_per_note=2, 
                           randomize_questions=False):
    """
    Prepares a prompt for GPT-4 and a DataFrame containing the selected clinical notes and questions.
    Notes are introduced first with their content, followed by all questions at the end, requiring GPT-4 to match questions to notes by note_key.

    :param df: DataFrame with 'ClinicalNoteKey', 'Note', and 'Question' columns.
    :param n_notes: Number of notes to include in the prompt, default is 10.
    :param n_questions_per_note: Number of questions to include per note, default is 2.
    :return: A tuple containing the structured prompt for GPT-4 and a DataFrame of the selected notes and questions.
    """
    # Sample questions from the DataFrame
    selected_notes = sample_questions(df, n_questions_per_note)
    
    # Select a specified number of unique clinical notes
    selected_keys = np.random.choice(selected_notes['ClinicalNoteKey'].unique(), n_notes, replace=False)
    filtered_notes = selected_notes[selected_notes['ClinicalNoteKey'].isin(selected_keys)]
    
    # If randomizing, shuffle the questions within filtered_notes
    if randomize_questions:
        questions_only = filtered_notes.sample(
            frac=1).reset_index(drop=True)  # Shuffle questions
    else:
        questions_only = filtered_notes
    
    # Construct the prompt with notes first
    notes_prompt = ""
    questions_prompt = "### Questions:\n"
    for note_key in selected_keys:
        note = filtered_notes[filtered_notes['ClinicalNoteKey'] == note_key].iloc[0]['Text']
        notes_prompt += f"#### Note Key: {note_key} Note:\n{note}\n"
    
    # Add questions at the end
    for index, row in questions_only.iterrows():
        question_number = row["question_number"]
        note_key = row["ClinicalNoteKey"]
        question = row['Question']
        questions_prompt += f"- Note Key: {note_key}, Q{question_number}: {question}\n"
        
    # Instructions for answering in JSON format
    prompt = """
Given the following clinical notes and questions, provide answers as directly and concisely as possible. For multiple items (e.g., symptoms, medications, findings), list the items separated by commas exactly as mentioned in the note, without introductory phrases. For single facts or statements, copy them verbatim from the note without alteration or introduction. Ensure all responses are factual, direct, and devoid of any additional context or commentary.

Please return the answers in JSON format with the structure:
[{"question_number": "<question_number>", "answer": "<concise_answer>"},...]
"""

    # Combine notes and questions into the final prompt
    prompt += notes_prompt + questions_prompt + "\n---\n"

    return prompt, filtered_notes.reset_index(drop=True)


def prepare_prompt_and_df_note_questions_format (
        df, n_notes = 10, n_questions_per_note=2):
    """
    Prepares a prompt for GPT-4 and a DataFrame containing the selected clinical notes and questions.

    :param df: DataFrame with 'ClinicalNoteKey', 'Note', and 'Question' columns.
    :param n_questions_per_note: Number of questions to include per note, default is 2.
    :return: A tuple containing the structured prompt for GPT-4 and a DataFrame of the selected notes and questions.
    """
    # Randomly select 5 unique clinical notes
    #selected_notes = df.groupby('ClinicalNoteKey').apply(lambda x: x.sample(n=min(len(x), n_questions_per_note))).reset_index(drop=True)
    
    selected_notes = sample_questions(df, n_questions_per_note)
    
    selected_keys = np.random.choice(selected_notes['ClinicalNoteKey'].unique(), n_notes, replace=False)
    filtered_notes = selected_notes[selected_notes['ClinicalNoteKey'].isin(selected_keys)]
    
    # Begin constructing the prompt
    prompt = """
Given the following clinical notes and questions, provide answers as directly and concisely as possible. For multiple items (e.g., symptoms, medications, findings), list the items separated by commas exactly as mentioned in the note, without introductory phrases. For single facts or statements, copy them verbatim from the note without alteration or introduction.

Please return the answers in JSON format with the structure:
[{"quest": "<question_number>", "ans": "<concise_answer>"},...]

---
"""

    last_key = None
    for index, row in filtered_notes.iterrows():
        question_number = row["question_number"]
        note = row['Text']  # Assuming the correct column name for note content is 'Note'
        note_key = row ["ClinicalNoteKey"]
        question = row['Question']
        
        # Introduce the note content only when moving to a new note
        if note_key != last_key:
            prompt += f"#### Note:\n{note}\n"
            last_key = note_key
            
        # Directly attach the question number to each question
        prompt += f" Q{question_number}: {question}"
        
    prompt += "\n---\n"
    
    # Remove the last separator for cleanliness
    prompt = prompt.rstrip("---\n")

    return prompt, filtered_notes.reset_index(drop=True)


def update_df_with_json_answers (json_response, filtered_df):
    """
    Enhances the DataFrame update process to handle varying lengths of json_response.
    Creates a new DataFrame from the json_response, merges it with filtered_df based on question_number.

    :param json_response: JSON string containing the answers from GPT-4.
    :param filtered_df: DataFrame with questions and other relevant details.
    :return: Updated DataFrame with answers merged based on question_number.
    """
    try:
        answers_data = json.loads(json_response)

        # Create a new DataFrame from the JSON response
        new_df = pd.DataFrame(answers_data)

        # If new_df is empty, return original df with added columns indicating failure
        if new_df.empty:
            # filtered_df ["note_key"] = pd.NA
            filtered_df['returned_question_number'] = pd.NA
            filtered_df['returned_answer'] = "fail"
            return filtered_df
        
        # Process 'question_number' to remove "Q" and convert to int
        new_df['quest'] = new_df[
            'quest'].str.replace("Q", "").astype(int)

        # Drop duplicates based on 'question_number'
        new_df = new_df.drop_duplicates(subset=[
            'quest'])

        # Rename columns for clarity in merging
        new_df.rename(columns={'quest': 'returned_question_number', 'ans': 'returned_answer'}, inplace=True)

        # Merge the new_df with filtered_df based on question_number
        updated_df = pd.merge(filtered_df, new_df, left_on='question_number', right_on='returned_question_number', how='left')

        return updated_df

    except json.JSONDecodeError:
        print("JSON failed")
        # Mark the process as failed by adding specific columns
        # filtered_df ["note_key"] = pd.NA
        filtered_df['returned_question_number'] = pd.NA
        filtered_df['returned_answer'] = "JSON decode failed"
        return filtered_df

        

def compare_answers(row):

    answer, returned_answer = row['Answer'], row['returned_answer']

    original_question = row ["Question"]
    
    # Step 1: Check for exact match
    try: 
        if answer.lower () == returned_answer.lower ():
            return 1
    except:
        return 0
    
    # Step 2: Check for fuzzy match
    threshold = 90  # Adjust the threshold as needed
    if fuzz.ratio (answer.lower (), 
                   returned_answer.lower ()) >= threshold:
        return 1
    
    # Step 3: Employ GPT-4 for semantic similarity check
    prompt = f"""Below are two answers extracted from an EHR note by large language models in response to a specific question. The objective is to determine whether these answers convey similar clinical information, despite differences in their elaboration. This comparison is crucial for evaluating the reliability and consistency of automated data extraction and interpretation in healthcare settings.

Original Question: "{original_question}"

- Statement A: "{answer}"
- Statement B: "{returned_answer}"

Please evaluate their semantic equivalence in the context of the information needed to satisfactorily answer the original question. Assign a score of 1 if both answers present essentially the same clinical information, indicating that either would lead to identical conclusions by healthcare providers. Assign a score of 0 if there are significant differences in the content of the answers that could influence clinical decisions or interpretations.
Your answer must be either 1 or 0, without any other characters or word as we are structuring it.
"""
    
    result = gptagent.ask_gpt (prompt, 
                               max_tokens=1, 
                               sleep = 0.5,
                               model = "gpt-4-8k-daal-sandbox-eus-01-01")  # This is a placeholder. Implement your actual GPT-4 query function here.
    
    
    if result == "1": return 1
    else: return 0
    
    
    
#### experiment_1
# 500, 5 * 10

df = pd.read_excel (
    r"C:\Users\klange01\Desktop\Email Response\code\stress test 20.2.24\500 tokens 25 questions 25.2.24.xlsx")

df ["question_number"] = range (0, df.shape[0])


for qst, num_of_notes in zip ([25, 10, 5], [2, 5, 10]):


    updated_df_list = []
    
    
    import re
    
    def reduce_spaces(text):
        # Replace multiple spaces with a single space
        reduced_text = re.sub(r'\s+', ' ', text)
        return reduced_text
    
    
    for exp in range (30):
        
        prompt, filtered_df = prepare_prompt_and_df_note_questions_format (
                df, n_notes = num_of_notes, n_questions_per_note=qst)
    
        prompt = reduce_spaces (prompt)
    
            
        # prepare_prompt_and_df (
        #    df, n_notes = 5, n_questions_per_note=20,
        #    randomize_questions=True)
        
        filtered_df ["prompt"] = prompt
    
        filtered_df ["exp"] = exp
        
        print (prompt)
        
        print (gptagent.num_tokens (prompt))
        
        json_response = gptagent.ask_gpt (
            prompt, model = "gpt-35turbo16k-d3m-sandbox-swc-01-01")  

        "GPT-4-8K = gpt-4-8k-d3m-sandbox-swc-01-01"
        "GPT-4-32K = gpt-4-32k-d3m-sandbox-swc-01-01"
        "GPT-3.5 = gpt-35turbo16k-d3m-sandbox-swc-01-01"
        
        prompt_tokens = gptagent.num_tokens (prompt)
        
        json_tokens = gptagent.num_tokens (json_response)
        
        print (prompt_tokens + json_tokens)
        
        
        filtered_df ["json_response"] = json_response
            
        updated_df = update_df_with_json_answers (json_response, 
                                                   filtered_df)
        
        
        # Add a new column to the DataFrame based on the comparison
        if updated_df ["returned_answer"][0] != "JSON decode failed":
        
            updated_df['comparison_result'] = updated_df.apply(
                compare_answers, axis=1)
        
        else:
            
            updated_df ["comparison_result"] = 0 
    
        updated_df_list.append (updated_df)
    
        print ("Experiment:", exp, "Accuracy", 
               updated_df ["comparison_result"].mean ())
    
    df_final = pd.concat (updated_df_list)
    
    df_final.to_excel (
        f"results 16k/private 500 tokens {num_of_notes}x{qst} questions GPT-3-16k.xlsx")