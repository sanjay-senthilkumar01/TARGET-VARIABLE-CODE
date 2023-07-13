import tensorflow as tf
from keras.layers.experimental.preprocessing import TextVectorization
from keras import Sequential, Model, losses, metrics, optimizers
from keras import callbacks
from keras.layers import MultiHeadAttention, LayerNormalization
from keras.layers import Dense, Dropout, Input, Embedding
from dataclasses import dataclass
from multiprocessing import Pool, freeze_support
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import glob
import random
import tarfile
import requests
import os
import shutil

dataset_dir = '/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/code test set'
metadata_dir = '/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/metadata test'  
problem_desc_dir = '/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/problem description test'  

# Load the data

def load_dataset(dataset_dir, languages):
    dataset = []

    for problem_dir in os.listdir(dataset_dir):
        problem_path = os.path.join(dataset_dir, problem_dir)
        if os.path.isdir(problem_path):
            for language_dir in os.listdir(problem_path):
                language_path = os.path.join(problem_path, language_dir)
                if language_dir in languages and os.path.isdir(language_path):
                    for code_file in os.listdir(language_path):
                        code_file_path = os.path.join(language_path, code_file)
                        with open(code_file_path, "r") as file:
                            code = file.read()
                            dataset.append(code)

    return dataset

languages = [
    "Ada", "AWK", "Bash", "Brainfuck", "C", "C#", "C++", "Clojure", "COBOL", "Common Lisp",
    "D", "Dart", "Elixir", "Elm", "Erlang", "F#", "Forth", "Fortran", "Go", "Groovy", "Haskell",
    "HTML", "Java", "JavaScript", "Julia", "Kotlin", "Lua", "MATLAB", "Nim", "OCaml", "Octave",
    "Pascal", "Perl", "PHP", "PL/I", "PowerShell", "Prolog", "Python", "R", "Racket", "Ruby",
    "Rust", "Scala", "Scheme", "Shell", "SQL", "Swift", "Tcl", "TypeScript", "VB.NET",
    "Vim script", "Visual Basic", "Whitespace", "XQuery", "Zsh"
]

dataset = load_dataset(dataset_dir, languages)

print(f"Total number of code snippets loaded: {len(dataset)}")

print("Dataset loading completed")
print("-------------------------")

metadata = pd.read_csv(os.path.join(metadata_dir, 'problem_list.csv'))
problem_metadata = {}
for problem_csv_file in os.listdir(metadata_dir):
    if problem_csv_file.startswith('p') and problem_csv_file.endswith('.csv'):
        problem_id = problem_csv_file.split('.')[0]
        problem_metadata[problem_id] = pd.read_csv(os.path.join(metadata_dir, problem_csv_file))

problem_descriptions = {}
for problem_html_file in os.listdir(problem_desc_dir):
    if problem_html_file.startswith('p') and problem_html_file.endswith('.html'):
        problem_id = problem_html_file.split('.')[0]
        problem_desc_file_path = os.path.join(problem_desc_dir, problem_html_file)
        with open(problem_desc_file_path, 'r') as file:
            problem_description = file.read()
            problem_descriptions[problem_id] = problem_description

tokenized_dataset = [word_tokenize(code) for code in dataset]

char_encoded_dataset = []
for code in dataset:
    char_encoded_dataset.append([ord(char) for char in code])


print("Code snippet preprocessing completed.")
print("-------------------------------------")

for problem_id, problem_data in problem_metadata.items():
    for _, row in problem_data.iterrows():
        
        submission_id = row[0]  
        user_id = row[2] 
        date = row[3]  
        language = row[4]  
        original_language = row[5]  
        filename_ext = row[6]  
        status = row[7]  

        if 'cpu_time' in row.index:
            cpu_time = row['cpu_time']  
        else:
            cpu_time = None

        if 'memory' in row.index:
            memory = row['memory']  
        else:
            memory = None

        if 'code_size' in row.index:
            code_size = row['code_size']  
        else:
            code_size = None

        if 'accuracy' in row.index:
            accuracy = row['accuracy'] 
        else:
            accuracy = None


print("META DATA Information loading completed")
print("---------------------------------------")


 # (--------------------------target variables for each task------------)
code_translation_target = []
bug_detection_target = []
code_quality_assessment_target = []
code_completion_target = []
code_classification_target = []
code_recommendation_target = []

###@ (-------code translation target variable---------)

for problem_id, problem_data in problem_metadata.items():
    for _, row in problem_data.iterrows():
        submission_id = row[0]  
        source_language = row[4]  
        target_language = row[5]  

        problem_id = row[1]  
        html_file_path = f"/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/problem description test/{problem_id}.html"
        
        if os.path.exists(html_file_path):
            with open(html_file_path, 'r') as html_file:
                problem_description = html_file.read()
        else:
            problem_description = "No problem description available"

        translation_output = f"Translate problem description from {source_language} to {target_language}"
        

        code_translation_target.append((submission_id, problem_description, translation_output))


print("Code Translation target variable creation completed.")
print("----------------------------------------------------")

#(-----------bug detection target variable ----------------)

for problem_id, problem_data in problem_metadata.items():
    for _, row in problem_data.iterrows():
        submission_id = row[0]  
        status = row[7]  
        
        if status != "Accepted":
            bug_detection_output = f"Status: {status}"
            bug_detection_target.append((submission_id, bug_detection_output))


print("Bug detection target variable creation completed.")
print("-------------------------------------------------")

#(------------------code ---completion----------)

for problem_id, problem_data in problem_metadata.items():
    for _, row in problem_data.iterrows():
        submission_id = row[0]  
        problem_id = row[1]  

        if problem_id in problem_descriptions:
            problem_description = problem_descriptions[problem_id] 

            code_completion_output = problem_description
            code_completion_target.append((submission_id, code_completion_output))
        else:
            code_completion_target.append((submission_id, "No problem description available"))

print("Code completion target variable creation completed.")
print("----------------------------------------------------")

#(-------------------code_classification_target-----------------------)

for problem_id, problem_data in problem_metadata.items():
    for _, row in problem_data.iterrows():
        submission_id = row[0]  
        language = row[4]  

        code_classification_target.append((submission_id, language))

print("Code classification target variable creation completed.")
print("----------------------------------------------------")
 
 #(------------------------code recommendation------------------------)

for problem_id, problem_data in problem_metadata.items():
    for _, row in problem_data.iterrows():
        submission_id = row[0] 
        language = row[4]  

        if row[7] == 'Accepted':
            recommendation_output = f"Recommend well-structured and efficient code in {language}"
        else:
            recommendation_output = f"Avoid common bugs and improve code correctness in {language}"

        code_recommendation_target.append((submission_id, recommendation_output))
 
print("Code recommendation target variable creation completed.")
print("-------------------------------------------------------")
