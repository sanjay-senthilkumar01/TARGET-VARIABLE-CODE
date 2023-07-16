import tensorflow as tf
from keras.layers.experimental.preprocessing import TextVectorization
from keras import Sequential, Model, losses, metrics, optimizers
from keras.utils import Sequence
from keras import callbacks
from keras.layers import MultiHeadAttention, LayerNormalization
from keras.layers import Dense, Dropout, Input, Embedding
from dataclasses import dataclass
from multiprocessing import Pool, freeze_support
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction import FeatureHasher


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

dataset_dir = '/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/test dataset/code test set'
metadata_dir = '/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/metadata test'  
problem_desc_dir = '/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/problem description test'  

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
                            # Get the file name (submission id) without the file extension
                            submission_id = os.path.splitext(code_file)[0]
                            dataset.append((submission_id, code))

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
'''
first_few_lines = dataset[:5] 

for i, (submission_id, code) in enumerate(first_few_lines, start=1):
    print(f"Submission ID: {submission_id}")
    print(f"Code snippet {i}:\n{code}\n")'''

print("Dataset loading completed")
print("-------------------------")

tokenized_dataset_with_id = []

for idx, (submission_id, code) in enumerate(dataset):
    tokens = word_tokenize(code)
    tokenized_dataset_with_id.append((submission_id, tokens))

# Print the first few entries in the tokenized_dataset_with_id
'''for i, (submission_id, tokens) in enumerate(tokenized_dataset_with_id[:5], start=1):
    print(f"Submission ID: {submission_id}")
    print(f"Tokenized Code {i}:")
    print(tokens)
    print()'''

#WOrd2vec model word embedding
all_tokens = [tokens for _, tokens in tokenized_dataset_with_id]

word2vec_model = Word2Vec(sentences=all_tokens, vector_size=50, window=5, min_count=1, sg=0)  

embedded_dataset_with_id = []

for submission_id, tokens in tokenized_dataset_with_id:
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    embedded_dataset_with_id.append((submission_id, embeddings))


# Print the first few entries in the embedded_dataset_with_id
'''num_examples_to_print = 1  

for i, (submission_id, embeddings) in enumerate(embedded_dataset_with_id[:num_examples_to_print], start=1):
    print(f"Example {i}")
    print("Submission ID:", submission_id)
    print("Word Embeddings:")
    for embedding in embeddings:
        print(embedding)  # Replace this with a prettier representation if needed
    print()'''
    

 #to file the submission id word embedding to check it worked correctly or ot   

'''def find_code_by_submission_id(embedded_dataset_with_id, target_submission_id):
    for submission_id, embeddings in embedded_dataset_with_id:
        if submission_id == target_submission_id:
            return embeddings
    return None  

target_submission_id = "s041116657" 
embeddings = find_code_by_submission_id(embedded_dataset_with_id, target_submission_id)

if embeddings is not None:
    
    print(f"Word embeddings for submission_id '{target_submission_id}':")
    print(embeddings)
else:
    print(f"Submission_id '{target_submission_id}' not found.")'''

print("Word embedding with submission id completed!")
print("--------------------------------------------")


#Metadat loading



# Step 1: Load the Dataset-Level Metadata
dataset_metadata = pd.read_csv('/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/test dataset/metadata test/problem_list.csv')

# Create a dictionary to store dataset-level metadata
dataset_metadata_dict = {
    'problem_ids': dataset_metadata['id'].tolist(),
    'names': dataset_metadata['name'].tolist(),
    'datasets': dataset_metadata['dataset'].tolist(),
    'time_limits': dataset_metadata['time_limit'].tolist(),
    'memory_limits': dataset_metadata['memory_limit'].tolist(),
    'ratings': dataset_metadata['rating'].tolist(),
    'tags': dataset_metadata['tags'].tolist(),
    'complexity': dataset_metadata['complexity'].tolist()
}

# Step 2: Load the Problem-Level Metadata
problem_ids = dataset_metadata['id'].tolist()  # Assuming you have the problem ids from the dataset-level metadata

# Create a dictionary to store problem-level metadata
problem_metadata_dict = {}

for problem_id in problem_ids:
    problem_file_path = f'/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/Project_CodeNet/metadata/{problem_id}.csv'
    problem_df = pd.read_csv(problem_file_path)
    problem_metadata_dict[problem_id] = {
        'submission_ids': problem_df['submission_id'].tolist(),
        'user_ids': problem_df['user_id'].tolist(),
        'dates': problem_df['date'].tolist(),
        'languages': problem_df['language'].tolist(),
        'original_languages': problem_df['original_language'].tolist(),
        'filename_exts': problem_df['filename_ext'].tolist(),
        'statuses': problem_df['status'].tolist(),
        'cpu_times': problem_df['cpu_time'].tolist(),
        'memories': problem_df['memory'].tolist(),
        'code_sizes': problem_df['code_size'].tolist(),
        'accuracies': problem_df['accuracy'].tolist()
    }

#TO check the metadata has loaded correctly (give submission id and get metadata info)
def get_submission_metadata_by_id(submission_id, problem_metadata_dict):
    for problem_id, metadata in problem_metadata_dict.items():
        if submission_id in metadata['submission_ids']:
            
            submission_index = metadata['submission_ids'].index(submission_id)

            
            submission_metadata = {
                'problem_id': problem_id,
                'submission_id': metadata['submission_ids'][submission_index],
                'user_id': metadata['user_ids'][submission_index],
                'date': metadata['dates'][submission_index],
                'language': metadata['languages'][submission_index],
                'original_language': metadata['original_languages'][submission_index],
                'filename_ext': metadata['filename_exts'][submission_index],
                'status': metadata['statuses'][submission_index],
                'cpu_time': metadata['cpu_times'][submission_index],
                'memory': metadata['memories'][submission_index],
                'code_size': metadata['code_sizes'][submission_index],
                'accuracy': metadata['accuracies'][submission_index]
            }

            return submission_metadata


    return None



submission_id_to_find = 's311084484'
submission_metadata = get_submission_metadata_by_id(submission_id_to_find, problem_metadata_dict)

if submission_metadata is not None:
    print("Submission ID:", submission_metadata['submission_id'])
    print("Problem ID:", submission_metadata['problem_id'])
    print("User ID:", submission_metadata['user_id'])
    print("Date:", submission_metadata['date'])
    print("Language:", submission_metadata['language'])
    print("Original Language:", submission_metadata['original_language'])
    print("Filename Extension:", submission_metadata['filename_ext'])
    print("Status:", submission_metadata['status'])
    print("CPU Time:", submission_metadata['cpu_time'], "milliseconds")
    print("Memory Used:", submission_metadata['memory'], "KB")
    print("Code Size:", submission_metadata['code_size'], "Bytes")
    print("Accuracy:", submission_metadata['accuracy'])
else:
    print("Submission ID not found.")


#preprocess the metadata


languages = [
    "Ada", "AWK", "Bash", "Brainfuck", "C", "C#", "C++", "Clojure", "COBOL", "Common Lisp",
    "D", "Dart", "Elixir", "Elm", "Erlang", "F#", "Forth", "Fortran", "Go", "Groovy", "Haskell",
    "HTML", "Java", "JavaScript", "Julia", "Kotlin", "Lua", "MATLAB", "Nim", "OCaml", "Octave",
    "Pascal", "Perl", "PHP", "PL/I", "PowerShell", "Prolog", "Python", "R", "Racket", "Ruby",
    "Rust", "Scala", "Scheme", "Shell", "SQL", "Swift", "Tcl", "TypeScript", "VB.NET",
    "Vim script", "Visual Basic", "Whitespace", "XQuery", "Zsh"
]

updated_problem_metadata_dict = {}


for problem_id, problem_data in problem_metadata_dict.items():
    problem_languages = problem_data['languages']
    one_hot_languages = pd.get_dummies(problem_languages, columns=['language'], prefix='', prefix_sep='')
    
    
    for language in languages:
        problem_data[language] = one_hot_languages[language].tolist()

    
    updated_problem_metadata_dict[problem_id] = problem_data


submission_id_to_query = 's311084484'


problem_id_for_submission = None
for problem_id, problem_data in updated_problem_metadata_dict.items():
    if submission_id_to_query in problem_data['submission_ids']:
        problem_id_for_submission = problem_id
        break


if problem_id_for_submission is not None:
    submission_info = updated_problem_metadata_dict[problem_id_for_submission]
    print(f"Submission ID: {submission_id_to_query}")
    print(f"Problem ID: {problem_id_for_submission}")
    print(f"User IDs: {submission_info['user_ids']}")
    print(f"Dates: {submission_info['dates']}")
    print(f"Languages: {submission_info['languages']}")
    print(f"Original Languages: {submission_info['original_languages']}")
    print(f"Filename Extensions: {submission_info['filenafme_exts']}")
    print(f"Statuses: {submission_info['statuses']}")
    print(f"CPU Times: {submission_info['cpu_times']}")
    print(f"Memories: {submission_info['memories']}")
    print(f"Code Sizes: {submission_info['code_sizes']}")
    print(f"Accuracies: {submission_info['accuracies']}")
else:
    print(f"Submission ID '{submission_id_to_query}' not found in the metadata.")
