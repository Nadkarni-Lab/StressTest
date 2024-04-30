# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:51:46 2024

@author: klange01
"""


import sys
import_folder_path = r"import/folder"
sys.path.append (import_folder_path)


from call_GPT import GPTAgent

gptagent = GPTAgent ()


import os
import pandas as pd


directory = ""

df_list = []

for item in [5, 15]:
    
    df = pd.read_excel (
        directory + f"/private 500 tokens 10x{item} questions GPT-3-32k.xlsx")

    df ["qst_n"] = item*10
    
    df_list.append (df)

df = pd.read_excel (directory + "/private 500 tokens 10x10 questions GPT-4-32k.xlsx")    

df ["qst_n"] = 100

df_list.append (df)
    
gpt4_df = pd.concat (df_list)
    
print (gpt4_df.groupby ("qst_n")["comparison_result"].mean ())





t = gpt4_df [
    gpt4_df ["returned_answer"] != "JSON decode failed"]

t = t [t ["Category"] != "Failed JSON"]

t [["qst_n", "Category", "comparison_result"]].to_excel (
    "results/failed.xlsx")




# 95% CI:
    
gpt4_df['Model'] = 'GPT-4'

def func_filter_returned_answer (x):
    
    if x == "JSON decode failed": return False
    
    return True

gpt4_df_filtered = gpt4_df [
    gpt4_df ["returned_answer"].apply (
        lambda x: func_filter_returned_answer (x))]


combined_df = pd.concat (
    [gpt4_df_filtered, 
     ])

combined_df ["comparison_result"] *=100

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6), dpi=300)
sns.set(style="whitegrid")
sns.set_context("talk")

palette = sns.color_palette("husl", 3)
sns.lineplot(x='qst_n', y='comparison_result', 
             hue='Model', style='Model',
             data=combined_df, 
             palette=palette, markers=True, 
             dashes=False, linewidth=2.5, ci=95)

plt.title('Model Accuracy by Number of Questions', fontsize=24)
plt.xlabel('Number of Questions', fontsize=22)
plt.ylabel('Accuracy (%)', fontsize=22)
plt.legend(title='Model', title_fontsize='20', fontsize='20')
plt.tight_layout()

plt.ylim (0,100)

plt.show()





##### Failed JSON


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to calculate the number of 'JSON decode failed' per experiment
def calculate_failures(df):
    failures = {}
    for qst_n in sorted(df['qst_n'].unique()):

        df ["JSON failed"] = df["returned_answer"].apply (
            lambda x: x == "JSON decode failed")        
        
        t = df [df ["qst_n"] == qst_n]["JSON failed"].sum () // qst_n
        
        failures.update ({qst_n: t})
        
    return failures

# Calculate the failures for each model
gpt4_failures = calculate_failures(gpt4_df)

# Prepare data for plotting
failure_data = pd.DataFrame({
    'Number of Questions': list(gpt4_failures.keys()), # + list(mixtral_failures.keys()), # + list(llama70_failures.keys()),
    'JSON Failures': list(gpt4_failures.values()), # + list(mixtral_failures.values()), # + list(llama70_failures.values()),
    'Model': ['GPT-4-32k']* len(gpt4_failures) #* len(gpt4_failures) + ['Mixtral'] * len(mixtral_failures) #+ ['Llama-2-70b'] * len(llama70_failures)
})


# Plotting the bar plot
plt.figure(figsize=(10, 6), dpi = 300)
sns.set(style="whitegrid")
sns.set_context("talk")

palette = sns.color_palette("husl", 3)
ax = sns.barplot(x='Number of Questions', y='JSON Failures', 
            hue='Model', data=failure_data, palette=palette)

plt.title('Failure of JSON load by Number of Questions', fontsize=24)
plt.xlabel('Number of Questions', fontsize=22)
plt.ylabel('Number of Failures', fontsize=22)
plt.legend(title='Model', title_fontsize='20', fontsize='20')
plt.tight_layout()

plt.ylim (0,30)

# Function to change bar width in vertical bar plots
def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # We change the bar width
        patch.set_width(new_value)

        # We recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

# Apply the function to adjust bar widths
change_width(ax, new_value=0.15)  
# Adjust 'new_value' to control the width of the bars

# Display the plot
plt.show()





##### Flesch

# 95% CI

import textstat

# Adjusted function to add Flesch scores and categories to the dataframe
def add_flesch_score_categories(df):
    df['flesch_score'] = df['Text'].apply(textstat.flesch_reading_ease)
    df['flesch_category'] = pd.cut(df['flesch_score'], bins=flesch_bins, labels=flesch_labels, right=False)
    df['flesch_category'] = pd.Categorical(df['flesch_category'], categories=flesch_labels, ordered=True)
    return df


# Define bins based on Flesch Reading Ease score ranges
flesch_bins = [0, 29, 49, 59, 69, 100]
flesch_labels = ['Very Difficult', 'Difficult', 'Fairly Difficult', 
                 'Plain English', 'Easy']


def plot_flesch_accuracy_comparison(
        gpt4_data): #llama70_data):
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set(style="whitegrid")
    sns.set_context("talk")

    sns.set_palette("husl", 3)

    # Combine the dataframes for easier plotting, ensuring unique indices
    combined_data = pd.concat(
        [gpt4_data.assign(Model='GPT-4-32k'), 
         ]).reset_index(drop=True)
         #llama70_data.assign (Model="Llama-2-70b"
                              

    # Ensure 'comparison_result' is expressed as a percentage
    combined_data ['comparison_result'] *= 100

    combined_data = combined_data [
        (combined_data ["returned_answer"] != "JSON decode failed")]


    # Plot with confidence intervals
    sns.lineplot(x='flesch_category', y='comparison_result', hue='Model', style='Model',
                 data=combined_data, markers=True, dashes=False, linewidth=2.5, ci=95, palette=palette)

    plt.title('Model Accuracy by Flesch Reading Ease Categories', fontsize=24)
    plt.xlabel('Flesch Reading Ease Category', fontsize=22)
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy (%)', fontsize=22)
    plt.legend(title='Model', title_fontsize='20', 
               fontsize='20')
    plt.tight_layout()

    plt.ylim (0,100)

    plt.show()



def analyze_and_compare_models (gpt4_df): #, llama70_df):
    # Add Flesch scores and categories without aggregating

    gpt4_data = add_flesch_score_categories(
        gpt4_df)


    # llama70_data = add_flesch_score_categories(
    #    llama70_df)

    # Reset indices after adding Flesch scores and categories
    gpt4_data.reset_index(drop=True, inplace=True)
    # llama70_data.reset_index(drop=True, inplace=True)


    # Now both dataframes are ready for plotting with CI
    plot_flesch_accuracy_comparison (
        gpt4_data) #, llama70_data)

    return gpt4_data #, llama70_data


# Assuming you have already defined gpt4_df and mixtral_df
gpt4_data = analyze_and_compare_models (
    gpt4_df) #, llama70_df)





##### Answer tokens


# 95% CI


def add_token_category(df):
    # Filter out "JSON decode failed" entries
    filtered_df = df[
        (df['returned_answer'] != 'JSON decode failed'
         )] 
    
    filtered_df ["answer_tokens"] = filtered_df [
        "Answer"].apply (
            lambda x: gptagent.num_tokens (str (x)))
    
    # Bin 'answer_tokens' into categories
    filtered_df['token_category'] = pd.cut(
        filtered_df['answer_tokens'],
        bins=[0, 5, 10, 15, 20, float('inf')],
        labels=['1-5', '6-10', '11-15', '16-20', '>20'],
        right=False
    )
    filtered_df['token_category'] = pd.Categorical(
        filtered_df['token_category'], 
        categories=['1-5', '6-10', '11-15', '16-20', '>20'], 
        ordered=True
    )
    return filtered_df

# Apply the function to both dataframes
gpt4_df_with_category = add_token_category(gpt4_df)

plt.figure(figsize=(10, 6), dpi=300)
sns.set(style="whitegrid")
sns.set_context("talk")

# Combine the modified dataframes with model labels
combined_data_with_category = pd.concat([
    gpt4_df_with_category.assign(Model='GPT-4-32k'),
    #llama70_df_with_category.assign (Model = "Llama-2-70b")
])

combined_data_with_category ["comparison_result"] *= 100

palette = sns.color_palette("husl", 3)
# Plot with confidence intervals
sns.lineplot(x='token_category', y='comparison_result', hue='Model', style='Model',
             data=combined_data_with_category, palette=palette, markers=True, dashes=False, linewidth=2.5, ci=95)

plt.title('Model Accuracy by Number of Tokens In Answer', fontsize=24)
plt.xlabel('Number of Tokens in Answer', fontsize=22)
plt.ylabel('Accuracy (%)', fontsize=22)
plt.legend(title='Model', title_fontsize='20', fontsize='20')
plt.tight_layout()

plt.ylim (0,100)

plt.show()







#### question number generation
# 95% CI


def prepare_data_with_matching_results(df):
    # Filter out "JSON decode failed" entries
    filtered_df = df[
        df['returned_answer'] != 'JSON decode failed']
    # Calculate match result as a new column
    filtered_df['match_percentage'] = (
        filtered_df['question_number'] == filtered_df[
            'returned_question_number']).astype(int) * 100
    return filtered_df


gpt4_prepared = prepare_data_with_matching_results(gpt4_df)

plt.figure(figsize=(10, 6), dpi=300)
sns.set(style="whitegrid")
sns.set_context("talk")

# Combine the prepared dataframes with model labels
combined_data = pd.concat([
    gpt4_prepared.assign(Model='GPT-4-32k'),
    # llama70_prepared.assign (Model = "Llama-2-70b")

])

palette = sns.color_palette("husl", 3)
# Plot with confidence intervals
sns.lineplot(x='qst_n', y='match_percentage', hue='Model', style='Model',
             data=combined_data, palette=palette, markers=True, dashes=False, linewidth=2.5, ci=95)

plt.title('Model Accuracy of Question Number Generation', fontsize=24)
plt.xlabel('Number of Questions Asked', fontsize=22)
plt.ylabel('Accuracy (%)', 
           fontsize=22)
plt.legend(title='Model', title_fontsize='20', fontsize='20')

plt.tight_layout()

plt.ylim (0,100)

plt.show()







# BLEU


import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction


def calculate_bleu_scores (df):
    
    # Tokenize the answers and returned_answers
    df['tokenized_answer'] = df['Answer'].apply(
        lambda x: word_tokenize(str (x).lower()))
    df['tokenized_returned_answer'] = df['returned_answer'].apply(
        lambda x: word_tokenize(str (x).lower()))
    
    # Calculate BLEU score for each row
    smoothie = SmoothingFunction().method4
    df['bleu_score'] = df.apply(
        lambda row: sentence_bleu(
            [row['tokenized_answer']], 
            row['tokenized_returned_answer'], 
            smoothing_function=smoothie), axis=1)
    
    return df


def prepare_and_visualize_bleu_scores_multiple_dfs(
        dfs, model_names):
    
    combined_data = pd.DataFrame()
    
    for df, model_name in zip(dfs, model_names):
        df_with_bleu = calculate_bleu_scores(df)
        df_with_bleu['Model'] = model_name  # Tag each row with the model name
        combined_data = pd.concat(
            [combined_data, df_with_bleu], ignore_index=True)
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set(style="whitegrid")
    sns.set_context("talk")

    palette = sns.color_palette("husl", 3)
    

    sns.lineplot(data=combined_data, x='qst_n', y='bleu_score', hue='Model', style='Model',
                 markers=True, dashes=False, linewidth=2.5, ci=95, palette=palette)
    
    plt.title('Average BLEU Score by Number of Questions for Different Models', fontsize=24)
    plt.xlabel('Number of Questions Asked', fontsize=22)
    plt.ylabel('Average BLEU Score', fontsize=22)
    plt.legend(title='Model', title_fontsize='20', fontsize='20')
    
    plt.ylim (0.1, 0.7)
    
    plt.tight_layout()
    plt.show()


def prepare_data_with_matching_results(df):

    def func_bleu (x):
        
        if x!=x: return True
        
        if "JSON decode failed" not in x: return True

        return False       
        
    
    df = df [df['returned_answer'].apply (
        lambda x: func_bleu (x))]
    
    return df


gpt4_prepared = prepare_data_with_matching_results (gpt4_df)

gpt4_prepared = calculate_bleu_scores (gpt4_prepared)

prepare_and_visualize_bleu_scores_multiple_dfs (
    [gpt4_prepared,], #, llama70_prepared],
    ["GPT-4-32k", ]) #, "Llama-2-70b"])





last_gpt4_df = pd.read_excel ("results/folder/tokens.xlsx")



# paper

gpt4_df_hold = gpt4_df.copy ()

gpt4_df = last_gpt4_df.copy ()



gpt4_df ["prompt_tokens"] = gpt4_df ["prompt"].apply (
    lambda x: gptagent.num_tokens (x))

temp = gpt4_df.drop_duplicates (subset = ["prompt"])


# Group by 'qst_n', then calculate mean and standard deviation of 'prompt_tokens'
grouped_stats = temp.groupby('qst_n')['prompt_tokens'].agg(['mean', 'std'])

# Round the mean and std to one decimal place and format them
formatted_stats = grouped_stats.apply(lambda x: f"{x['mean']:.1f} ± {x['std']:.1f}", axis=1)

# Convert the Series to a list of strings if you need a list specifically
formatted_stats_list = formatted_stats.tolist()

# Example output
for item in formatted_stats_list:
    print(item)


gpt4_df = gpt4_df_hold









for qst_n in [50, 100, 150]:
    
    print (
        gpt4_df [gpt4_df ["qst_n"] == qst_n][
            "returned_answer"].apply (
                lambda x: x == "JSON decode failed").sum ())


                
                




        
import scipy.stats as stats

# accuracies

def func_pear (df):
    
    df = df [df ["returned_answer"] != "JSON decode failed"]

    # Group by 'exp' and 'qst_n' and then calculate the mean 'comparison_result' for each group
    df_grouped = df.groupby(['exp', 'qst_n'])[
        'comparison_result'].mean().reset_index(name='accuracy')

    # Calculate the Pearson correlation for each model
    pearson_r, pearson_p = stats.spearmanr (df_grouped[
        'qst_n'], df_grouped['accuracy'])

    print (pearson_r, pearson_p)    
                
for item, name in zip ([gpt4_df], ["GPT-4-32k"]):
                                         
        print (name)
        func_pear (item)

# Accuracies
                
for df, name in zip ([gpt4_df], ["GPT-4-32k"]):
    item = df.copy ()
    item = item [item ["returned_answer"] != "JSON decode failed"]
                                         
    min_q = min (item ["qst_n"].unique ())
    mid_q = 100
    max_q = max (item ["qst_n"].unique ())
    print (name, "min qst_n:", min_q, "max qst_n:", max_q)
    print ("Accuracy in min_q:", item [
        item ["qst_n"] == min_q]["comparison_result"].mean ())
    print ("Accuracy in max_q:", item [
        item ["qst_n"] == max_q]["comparison_result"].mean ())                
    print ("Accuracy in max_q:", item [
        item ["qst_n"] == 100]["comparison_result"].mean ())                





directory = "analysis/folder"

gpt4_8k = pd.read_excel (directory + "/gpt4_df.xlsx")

mixtral_df = pd.read_excel (
    directory + "/mixtral_df fixed.xlsx")


def func_fix_value (x):
    
    if x != x:
        return x
    
    if type (x) != str:
        return x
    
    x = x.replace ("Value error", "JSON decode failed")
    
    return x


mixtral_df ["returned_answer"] = mixtral_df [
    "returned_answer"].apply (
        lambda x: func_fix_value (x))

        

directory = "analysis/folder"

gpt35 = pd.read_excel (directory + "/gpt3.5.xlsx")

mixtral2 = pd.read_excel (
    directory + "/mixtral fixed.xlsx")


def func_fix_value (x):
    
    if x != x:
        return x
    
    if type (x) != str:
        return x
    
    x = x.replace ("Value error", "JSON decode failed")
    
    return x


mixtral2 ["returned_answer"] = mixtral_df [
    "returned_answer"].apply (
        lambda x: func_fix_value (x))

        
directory_llama = "results/results llama70"

llama70_df = pd.read_excel (
    directory_llama + "/llama70 fixed.xlsx")



# Iterate over the DataFrame list
# for df in [gpt4_df, mixtral_df, gpt4_8k]:
for df in [gpt35, mixtral2, llama70_df]:
    # Filter out 'JSON decode failed'
    filtered_df = df[df[
        "returned_answer"] != "JSON decode failed"]
    
    # Calculate mean and standard deviation
    stats_df = filtered_df.groupby("Category")[
        "comparison_result"].agg(['mean', 'std'])
    
    stats_df['mean'] = stats_df['mean'] * 100
    stats_df['std'] = stats_df['std'] * 100
    
    # Print results in the specified format
    for category, row in stats_df.iterrows():
        print(f"{category}: {row['mean']:.1f} ± {row['std']:.1f}%")







# questions generation


def func_pear (df):
    
    df = df [df ["returned_answer"] != "JSON decode failed"]

    df ["comparison_qst"] = df ["question_number"] == df ["returned_question_number"]

    # Group by 'exp' and 'qst_n' and then calculate the mean 'comparison_result' for each group
    df_grouped = df.groupby(['exp', 'qst_n'])[
        'comparison_qst'].mean().reset_index(name='accuracy')

    # Calculate the Pearson correlation for each model
    pearson_r, pearson_p = stats.spearmanr (df_grouped[
        'qst_n'], df_grouped['accuracy'])

    print (pearson_r, pearson_p)    
                
for item, name in zip ([gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral], ["GPT-3.5", "Mixtral", "Llama-2-70b",
                            "GPT-4-8k", "Mixtral"]):
                                         
        print (name)
        func_pear (item)



# question generation Accuracies
                
for df, name in zip ([gpt4_df], ["GPT-4-32k"]):
    item = df.copy ()
    item = item [item ["returned_answer"] != "JSON decode failed"]
                                         
    item ["comparison_qst"] = item ["question_number"] == item ["returned_question_number"]
    
    min_q = min (item ["qst_n"].unique ())
    max_q = max (item ["qst_n"].unique ())
    mid_q = 100
    print (name, "min qst_n:", min_q, "max qst_n:", max_q)
    print ("Accuracy in min_q:", item [
        item ["qst_n"] == min_q]["comparison_qst"].mean ())
    print ("Accuracy in max_q:", item [
        item ["qst_n"] == max_q]["comparison_qst"].mean ())                
    print ("Accuracy in 100:", item [
        item ["qst_n"] == mid_q]["comparison_qst"].mean ())                




# sub-analyses





import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction


def calculate_bleu_scores (df):

    df = df [df ["returned_answer"] != "JSON decode failed"]
    
    # Tokenize the answers and returned_answers
    df['tokenized_answer'] = df['Answer'].apply(
        lambda x: word_tokenize(str (x).lower()))
    df['tokenized_returned_answer'] = df['returned_answer'].apply(
        lambda x: word_tokenize(str (x).lower()))
    
    # Calculate BLEU score for each row
    smoothie = SmoothingFunction().method4
    df['bleu'] = df.apply(
        lambda row: sentence_bleu(
            [row['tokenized_answer']], 
            row['tokenized_returned_answer'], 
            smoothing_function=smoothie), axis=1)
    
    return df

def calculate_token_number (df):
    
    df ["tokens"] = df ["Answer"].apply (
        lambda x: gptagent.num_tokens (str(x)))

    return df


# Add binned Flesch score categories to the dataframe
def calculate_flesch_score_categories(df):
    
    flesch_bins = [0, 29, 49, 59, 69, 100]
    flesch_labels = ['Very Difficult', 'Difficult', 'Fairly Difficult', 
                     'Plain English', 'Easy']

    df['flesch_score'] = df['Text'].apply(textstat.flesch_reading_ease)
    df['flesch'] = pd.cut(
        df['flesch_score'], bins=flesch_bins, labels=flesch_labels, right=True)
    
    return df

for df in [gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral]:
    
    df = calculate_bleu_scores (df)
    
    df = calculate_token_number (df)

    df = calculate_flesch_score_categories(df)






                                         
for df, name in zip ([gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral], ["GPT-3.5", "Mixtral", "Llama-2-70b",
                            "GPT-4-8k", "Mixtral"]):
    
    print (name)
                                         
    print (df.groupby (
        ["qst_n"])["bleu"].mean ())
    
    




                                         
                                         
for df, name in zip ([gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral], ["GPT-3.5", "Mixtral", "Llama-2-70b",
                            "GPT-4-8k", "Mixtral"]):
    
    print (name)
                                         
    print (df.groupby (["flesch"])["comparison_result"].mean ())
 
                                         
    
    
    
    
    
    
    
    
    
    
    
def add_token_category (df):
    
    df ["answer_tokens"] = df ["Answer"].apply (
        lambda x: gptagent.num_tokens (x) if x==x else 0)

    def func_token_cat (x):
        
        if x<=5: return 1
        if x<=10: return 2
        if x<=15: return 3
        if x<=20: return 4
        return 5

    # Bin 'answer_tokens' into categories
    df ['token_cat'] = df ['answer_tokens'].apply (lambda x: func_token_cat (x))

for item in [gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral]:
    
    item = add_token_category (item)
    
                               
                                         
target = "answer_tokens"
                                         
import textstat

def func_pear_ (df):
    
    df = df [df ["returned_answer"] != "JSON decode failed"]
    
    # Group by 'exp' and 'qst_n' and then calculate the mean 'comparison_result' for each group
    df_grouped = df.groupby (['exp', 'token_cat'])[
        "comparison_result"].mean().reset_index(name="accuracy")

    # Calculate the Pearson correlation for each model
    pearson_r, pearson_p = stats.pearsonr (df_grouped[
        'token_cat'], df_grouped["accuracy"])

    print (pearson_r, pearson_p)    
                
for item, name in zip ([gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral], ["GPT-3.5", "Mixtral", "Llama-2-70b",
                            "GPT-4-8k", "Mixtral"]):

    print (name)
    
    func_pear_ (item)




# question generation Accuracies
                
for df, name in zip ([gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral], ["GPT-3.5", "Mixtral", 
                                         "Llama-2-70b",
                            "GPT-4-8k", "Mixtral"]):
        
    item = df.copy ()
    item = item [item ["returned_answer"] != "JSON decode failed"]
                                                 
    min_q = min (item ["token_cat"].unique ())
    max_q = max (item ["token_cat"].unique ())
    print (name, "min token cat:", min_q, "max token cat:", max_q)
    print (target, " in min_q:", item [
        item ["qst_n"] == min_q][target].mean ())
    print (target, " in max_q:", item [
        item ["qst_n"] == max_q][target].mean ())                




def func_pear_ (df):
    
    df = df [df ["returned_answer"] != "JSON decode failed"]
    
    # Calculate the Pearson correlation for each model
    pearson_r, pearson_p = stats.pearsonr (df ["flesch_score"], 
                                           df ["comparison_result"])

    print (pearson_r, pearson_p)    
                
for item, name in zip ([gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral], ["GPT-3.5", "Mixtral", "Llama-2-70b",
                            "GPT-4-8k", "Mixtral"]):

    print (name)
    
    func_pear_ (item)




import numpy as np

def func_tokens (df):

    print (stats.spearmanr (df ["tokens"], df ["comparison_result"]))


for item, name in zip ([gpt4_df, mixtral_df, llama70_df,
                        gpt4, mixtral], ["GPT-3.5", "Mixtral", "Llama-2-70b",
                            "GPT-4-8k", "Mixtral"]):

    print (name)
    
    func_tokens (item)







# Old code


import medspacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the default MedSpaCy model
nlp = medspacy.load()

# Function to calculate the density of medical terms using MedSpaCy
def medical_term_density(text):
    doc = nlp(text)
    medical_terms = [ent.text for ent in doc.ents]  # Extract medical terms
    num_medical_terms = len(medical_terms)
    num_total_words = len(text.split())
    density = (num_medical_terms / num_total_words) if num_total_words else 0
    return density * 100  # As a percentage

# Add a column with medical term density to the DataFrame
def add_medical_term_density(df):
    df['medical_term_density'] = df['Text'].apply(medical_term_density)
    return df

# Bin the densities into categories
density_bins = [0, 25, 50, 75, 100]
density_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
def bin_density(df):
    df['density_category'] = pd.cut(df['medical_term_density'], bins=density_bins, labels=density_labels, include_lowest=True)
    return df

# Aggregate data by medical term density category
def aggregate_data_by_density(df):
    grouped = df.groupby('density_category')['comparison_result'].mean().reset_index()
    return grouped

# Plotting function for a single model
def plot_density_accuracy(df, title):
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x='density_category', y='comparison_result', data=df)
    
    plt.title(f'{title} Accuracy vs. Medical Term Density', fontsize=20)
    plt.xlabel('Medical Term Density', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.tight_layout()
    plt.show()

df = add_medical_term_density(gpt4_df)
df = bin_density(df)
aggregated_data = aggregate_data_by_density(df)
plot_density_accuracy(aggregated_data, 'Model Name')





# Flech kinkade on questions

import textstat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Define bins based on Flesch Reading Ease score ranges
flesch_bins = [0, 29, 49, 59, 69, 79, 89, 100]
flesch_labels = ['Very Difficult', 'Difficult', 'Fairly Difficult', 
                 'Plain English', 
                 'Fairly Easy', 'Easy', 'Very Easy']

# Define bins based on Flesch Reading Ease score ranges
flesch_bins = [0, 49, 69, 100]
flesch_labels = ['Very Difficult -\nDifficult', 
                 'Fairly Difficult -\nPlain English', 
                 'Fairly Easy -\nVery Easy']


# Add binned Flesch score categories to the dataframe
def add_flesch_score_categories(df):
    df['flesch_score'] = df['Text'].apply(textstat.flesch_reading_ease)
    df['flesch_category'] = pd.cut(
        df['flesch_score'], bins=flesch_bins, labels=flesch_labels, right=True)
    return df


# Adjust the aggregate_data function to use the new flesch_category
def aggregate_data(df):
    grouped = df.groupby(
        ['qst_n', 'flesch_category'])['comparison_result'].mean().reset_index()
    return grouped

# Plotting function
def plot_flesch_accuracy(df, title):
    plt.figure(figsize=(10, 6), dpi=300)
    sns.lineplot(x='flesch_category', 
                 y='comparison_result', hue='qst_n', data=df, markers=True)
    
    plt.title(f'{title} Accuracy vs. Flesch Score', fontsize=20)
    plt.xlabel('Flesch Score', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.legend(title='Experiment Number (qst_n)', title_fontsize='13', fontsize='12')
    plt.tight_layout()
    plt.show()

# Main analysis function
def analyze_model(df, model_name):
    df = add_flesch_score_categories(df)
    # Create bins for Flesch score for grouping
    df['flesch_score_bin'] = pd.cut(df['flesch_score'], bins=10)
    aggregated_data = aggregate_data(df)
    plot_flesch_accuracy(aggregated_data, model_name)
    return aggregated_data

# Example usage:
# t = analyze_model (gpt4_df, 'GPT-4')
t = analyze_model(mixtral_df, 'Mixtral')






# Define bins based on Flesch Reading Ease score ranges
flesch_bins = [0, 29, 49, 59, 69, 79, 89, 100]
flesch_labels = ['Very Difficult', 'Difficult', 'Fairly Difficult', 
                 'Plain English', 
                 'Fairly Easy', 'Easy', 'Very Easy']

# Define bins based on Flesch Reading Ease score ranges
flesch_bins = [0, 49, 69, 100]
flesch_labels = ['Very Difficult -\nDifficult', 
                 'Fairly Difficult -\nPlain English', 
                 'Fairly Easy -\nVery Easy']


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textstat

# Adjusted function to add Flesch scores and categories
def add_flesch_score_categories(df):
    df['flesch_score'] = df['Text'].apply(textstat.flesch_reading_ease)
    df['flesch_category'] = pd.cut(df['flesch_score'], bins=flesch_bins, labels=flesch_labels, right=True)
    return df

# New function to aggregate data by Flesch category only, without qst_n
def aggregate_data_by_flesch(df):
    grouped = df.groupby('flesch_category')['comparison_result'].mean().reset_index()
    return grouped

# New plotting function to plot for both models on the same graph
def plot_flesch_accuracy_comparison(gpt4_data, mixtral_data):
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot GPT-4 data
    sns.lineplot(x='flesch_category', y='comparison_result', data=gpt4_data,
                 label='GPT-4', markers=True)
    
    # Plot Mixtral data
    sns.lineplot(x='flesch_category', y='comparison_result', data=mixtral_data,
                 label='Mixtral', markers=True)
    
    plt.title('Accuracy vs. Flesch Score for GPT-4 and Mixtral', fontsize=20)
    plt.xlabel('Flesch Score', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.legend(title='Model', title_fontsize='13', fontsize='12')
    plt.tight_layout()
    # plt.savefig('flesch_accuracy_comparison.png', dpi=300)  # Save the figure
    plt.show()

# Main function to prepare data for both models and plot the comparison
def analyze_and_compare_models(gpt4_df, mixtral_df):
    gpt4_data = add_flesch_score_categories(gpt4_df)
    mixtral_data = add_flesch_score_categories(mixtral_df)
    
    gpt4_aggregated = aggregate_data_by_flesch(gpt4_data)
    mixtral_aggregated = aggregate_data_by_flesch(mixtral_data)
    
    plot_flesch_accuracy_comparison(gpt4_aggregated, mixtral_aggregated)


analyze_and_compare_models(gpt4_df, mixtral_df)
