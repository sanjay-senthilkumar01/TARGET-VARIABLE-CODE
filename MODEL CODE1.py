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
from sklearn.preprocessing import LabelEncoder


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

# Print the first few entries in the tokenized_dataset_with_id(🔍)
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


# Print the first few entries in the embedded_dataset_with_id(🔍)
'''num_examples_to_print = 1  

for i, (submission_id, embeddings) in enumerate(embedded_dataset_with_id[:num_examples_to_print], start=1):
    print(f"Example {i}")
    print("Submission ID:", submission_id)
    print("Word Embeddings:")
    for embedding in embeddings:
        print(embedding)  # Replace this with a prettier representation if needed
    print()'''
    

 #to file the submission id word embedding to check it worked correctly or ot   

def find_code_by_submission_id(embedded_dataset_with_id, target_submission_id):
    for submission_id, embeddings in embedded_dataset_with_id:
        if submission_id == target_submission_id:
            return embeddings
    return None  

target_submission_id = "s785268131" 
embeddings = find_code_by_submission_id(embedded_dataset_with_id, target_submission_id)

if embeddings is not None:
    
    print(f"Word embeddings for submission_id '{target_submission_id}':")
    print(embeddings)
else:
    print(f"Submission_id '{target_submission_id}' not found.")

print("Word embedding with submission id completed!")
print("--------------------------------------------")


#Metadat loading(🔄)



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


problem_ids = dataset_metadata['id'].tolist()  
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

#TO check the metadata has loaded correctly (give submission id and get metadata info)(🔍)
'''def get_submission_metadata_by_id(submission_id, problem_metadata_dict):
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



submission_id_to_find = 's001601661'
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
    print("Submission ID not found.")'''

print("metadata loading completed!")
print("---------------------------")

#Process the metadata
label_encoder = LabelEncoder()

updated_metadata_dict = {}


for problem_id, problem_data in problem_metadata_dict.items():
    
    updated_languages = label_encoder.fit_transform(problem_data['languages'])
    updated_original_languages = label_encoder.fit_transform(problem_data['original_languages'])
    updated_filename_exts = label_encoder.fit_transform(problem_data['filename_exts'])

   
    updated_metadata_dict[problem_id] = {
        'submission_ids': problem_data['submission_ids'],
        'user_ids': problem_data['user_ids'],
        'dates': problem_data['dates'],
        'languages': updated_languages.tolist(),
        'original_languages': updated_original_languages.tolist(),
        'filename_exts': updated_filename_exts.tolist(),
        'statuses': problem_data['statuses'],
        'cpu_times': problem_data['cpu_times'],
        'memories': problem_data['memories'],
        'code_sizes': problem_data['code_sizes'],
        'accuracies': problem_data['accuracies']
    }
#TO check the updated metadata encoding(🔍)
def get_metadata_by_submission_id(submission_id):
    for problem_id, problem_data in updated_metadata_dict.items():
        if submission_id in problem_data['submission_ids']:
            index = problem_data['submission_ids'].index(submission_id)
            metadata = {
                'problem_id': problem_id,
                'submission_id': submission_id,
                'user_id': problem_data['user_ids'][index],
                'date': problem_data['dates'][index],
                'language': problem_data['languages'][index],
                'original_language': problem_data['original_languages'][index],
                'filename_ext': problem_data['filename_exts'][index],
                'status': problem_data['statuses'][index],
                'cpu_time': problem_data['cpu_times'][index],
                'memory': problem_data['memories'][index],
                'code_size': problem_data['code_sizes'][index],
                'accuracy': problem_data['accuracies'][index]
            }
            return metadata
   
    return None


submission_id_to_find = 's785268131'
metadata = get_metadata_by_submission_id(submission_id_to_find)

if metadata:
    print("Metadata for Submission ID:", submission_id_to_find)
    for key, value in metadata.items():
        print(f"{key}: {value}")
else:
    print("Submission ID not found.")

print("encoding of metadata completed!")
print("-----------------------------------")

#
#PRoblem ddescription

import os
import pandas as pd
import spacy
from bs4 import BeautifulSoup
import re


def read_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content

def preprocess_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()

    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', clean_text)

    clean_text = clean_text.lower()

    doc = nlp(clean_text)
    tokens = [token.text for token in doc]

    return tokens

nlp = spacy.load("en_core_web_sm")

html_folder = '/Users/sanjaysenthilkumar/Documents/NI project/code doc & code arch/data set/test dataset/problem description test'
html_data = {}

for file_name in os.listdir(html_folder):
    file_path = os.path.join(html_folder, file_name)
    
    if os.path.isfile(file_path) and file_name.endswith('.html'):
        tracker = os.path.splitext(file_name)[0]
        
        html_content = read_html_file(file_path)
        
        tokenized_content = preprocess_text(html_content)
        
        html_data[tracker] = tokenized_content


trackers = list(html_data.keys())
tokenized_contents = list(html_data.values())

'''desired_tracker = "p00002"  

if desired_tracker in html_data:
    print(f"Tracker: {desired_tracker}")
    print("Tokenized Content:")
    print(html_data[desired_tracker])
else:
    print(f"Tracker '{desired_tracker}' not found in the html_data dictionary.")
'''

#Problem description encoding

import numpy as np

def load_glove_format(file_path):
    embedding_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict


glove_path = '/Users/sanjaysenthilkumar/Documents/NI project/glove pretrained model/glove.6B/glove.6B.100d.txt'  # Update with the path to the downloaded GloVe file


glove_model = load_glove_format(glove_path)


glove_vectors = []
for tokens in tokenized_contents:
    vectors = [glove_model[word] for word in tokens if word in glove_model]
    if vectors:
        mean_vector = np.mean(vectors, axis=0)
    else:
        mean_vector = np.zeros(100)  # Replace 100 with the correct dimension GloVe embeddings
    glove_vectors.append(mean_vector)

vocabulary = list(glove_model.keys())
glove_word_vectors = list(glove_model.values())

# To check()
'''desired_tracker = "p00002"  
if desired_tracker in trackers:
    index = trackers.index(desired_tracker)
    print(f"GloVe representation for Tracker '{desired_tracker}':")
    print(glove_vectors[index])
else:
    print(f"Tracker '{desired_tracker}' not found in the 'trackers' list.")'''


print('word embedding of problem description is completed')
print('--------------------------------------------------')




  
#
#         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Target variable for the model👍 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
#TO DO LIST
#1. code completion to assist developers in writing code
#2. bug detection using machine learning algorithms(error too)
#3.Code classification using deep learning techniques for organizing codebases
#4. Code Security Analysis to identify vulnerabilities and threats
#5.Code Debugging
#6.Code clone detection to identify duplicated code fragments
#7.Automatic code generation based on high-level specifications or templates
#8.code summarization for generating concise documentation
#9.code smell detection for identifying poor coding practices
#10. AUTOmated code suggestion for data anlaytics


combined_data = []


for submission_id, embeddings in embedded_dataset_with_id:
    for problem_id, problem_data in updated_metadata_dict.items():
        if submission_id in problem_data['submission_ids']:
            encoded_languages = problem_data['languages'][problem_data['submission_ids'].index(submission_id)]
            combined_data.append((submission_id, embeddings, encoded_languages))
            break


desired_submission_id = "s785268131"  

for submission_id, embeddings, encoded_languages in combined_data:
    if submission_id == desired_submission_id:
        print("Submission ID:", submission_id)
        print("Embeddings:", embeddings)
        print("Encoded Languages:", encoded_languages)
        break
else:
    print("Submission ID not found.")

#
#      <-------------------------------------------parallel hybrid model of transformers and LSTM based RNN models👍-------------------------->
#
#TO DO LIST
#1.shared input layer 
#2. Bidirectional Transformer Model
#3.LSTM Model
#4. Merge Layers
#5.Task-specific Output Layers
# 6.Task-specific Output Layers

import tensorflow as tf
from keras.layers import Input, Concatenate, Dense
max_sequence_length =2
# Define input shape and other hyperparameters
input_shape = (max_sequence_length,)  # The shape of your input sequences
vocab_size = ...  # The size of the vocabulary (number of unique tokens in your data)

# Shared input layer
shared_input_layer = Input(shape=input_shape)

# Define the Bidirectional Transformer model
transformer_model = ...  # Create the Bidirectional Transformer model using Keras or other libraries

# Define the LSTM model
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.Embedding(vocab_size, num_lstm_units, input_length=max_sequence_length))
lstm_model.add(tf.keras.layers.LSTM(num_lstm_units))

# Get representations from both models
transformer_rep = transformer_model(shared_input_layer)
lstm_rep = lstm_model(shared_input_layer)

# Merge representations (you can use different methods, like concatenation)
merged_rep = Concatenate()([transformer_rep, lstm_rep])

# Interconnected layers (optional)
interconnected_layer = Dense(128, activation='relu')(merged_rep)
# Add more interconnected layers as needed

# Task-specific output layers for each task
num_classes_task1 = ...  # Number of classes for task 1 (classification)
num_classes_task2 = ...  # Number of classes for task 2 (classification)

task1_output = Dense(num_classes_task1, activation='softmax')(interconnected_layer)
task2_output = Dense(num_classes_task2, activation='sigmoid')(interconnected_layer)

# Create the multi-task hybrid model
multi_task_model = tf.keras.Model(inputs=shared_input_layer, outputs=[task1_output, task2_output])

# Compile the model and define task-specific losses, metrics, and optimizer
multi_task_model.compile(optimizer='adam',
                         loss={'task1_output': 'categorical_crossentropy', 'task2_output': 'binary_crossentropy'},
                         metrics={'task1_output': 'accuracy', 'task2_output': 'accuracy'},
                         loss_weights={'task1_output': 1.0, 'task2_output': 0.5})














###(---------------------------------Transformer model----------------------------------------------------)
import keras
from keras.layers import Input, Dense, Dropout, LayerNormalization, Embedding, concatenate

def positional_encoding(max_sequence_length, d_model):
    # Compute positional encodings for the input sequences
    # Replace this implementation with your preferred positional encoding mechanism
    positions = tf.range(max_sequence_length, dtype=tf.float32)[:, tf.newaxis]
    dimensions = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angles = 1 / tf.pow(10000, 2 * dimensions / d_model)
    positional_encodings = positions * angles
    positional_encodings = tf.stack([tf.sin(positional_encodings[:, dim::d_model]) for dim in range(d_model)], axis=-1)
    return positional_encodings

def create_code_completion_model(max_sequence_length, vocab_size, num_transformer_heads, d_model, dff, num_encoder_layers, num_decoder_layers):
    # Input layer for the full code sequence
    input_sequence = Input(shape=(max_sequence_length,))

    # Embedding layer for the input tokens
    embedding_layer = Embedding(vocab_size, d_model)(input_sequence)

    # Add positional encoding to the input embeddings
    positional_encodings = positional_encoding(max_sequence_length, d_model)
    encoder_output = embedding_layer + positional_encodings

    # Bidirectional Encoder Layers with Masked Attention
    for _ in range(num_encoder_layers):
        # Masked Multi-head self-attention (using TensorFlow's MultiHeadAttention as a custom layer)
        masked_multi_head_attention = keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=num_transformer_heads, dropout=0.1)
        encoder_output = masked_multi_head_attention([encoder_output, encoder_output], attn_mask=None)
        encoder_output = LayerNormalization(epsilon=1e-6)(encoder_output + embedding_layer)  # Add residual connection

        # Optional: Add dropout to the encoder output
        encoder_output = Dropout(0.1)(encoder_output)

        # Feed-forward neural network
        feed_forward_output = Dense(dff, activation='relu')(encoder_output)
        feed_forward_output = Dense(d_model)(feed_forward_output)
        encoder_output = LayerNormalization(epsilon=1e-6)(encoder_output + feed_forward_output)  # Add residual connection

        # Optional: Add dropout to the encoder output
        encoder_output = Dropout(0.1)(encoder_output)

    # Decoder Layers with Masked Attention
    decoder_input = Input(shape=(max_sequence_length,))

    # Embedding layer for the target tokens (decoder input)
    decoder_embedding_layer = Embedding(vocab_size, d_model)(decoder_input)

    # Concatenate the encoder output and decoder input
    decoder_input_with_context = concatenate([encoder_output, decoder_embedding_layer], axis=-1)

    # Decoder layers similar to the encoder
    decoder_output = decoder_input_with_context
    for _ in range(num_decoder_layers):
        # Masked Multi-head self-attention (using TensorFlow's MultiHeadAttention as a custom layer)
        masked_multi_head_attention = keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=num_transformer_heads, dropout=0.1)
        decoder_output = masked_multi_head_attention([decoder_output, decoder_output], attn_mask=None)
        decoder_output = LayerNormalization(epsilon=1e-6)(decoder_output + decoder_embedding_layer)  # Add residual connection

        #  Add dropout to the decoder output
        decoder_output = Dropout(0.1)(decoder_output)

        # Feed-forward neural network
        feed_forward_output = Dense(dff, activation='relu')(decoder_output)
        feed_forward_output = Dense(d_model)(feed_forward_output)
        decoder_output = LayerNormalization(epsilon=1e-6)(decoder_output + feed_forward_output)  # Add residual connection

        # Add dropout to the decoder output
        decoder_output = Dropout(0.1)(decoder_output)

    # Output layer for code token prediction (code completion)
    output_layer = Dense(vocab_size, activation='softmax')(decoder_output)

    # Create the model with both encoder and decoder inputs
    model = keras.models.Model(inputs=[input_sequence, decoder_input], outputs=output_layer)

    return model

# Parameters
max_sequence_length = ... 
vocab_size = ... 
num_transformer_heads = ...  
d_model = ...
dff = ...  
num_encoder_layers = ...  
num_decoder_layers = ...  
model = create_code_completion_model(max_sequence_length, vocab_size, num_transformer_heads, d_model, dff, num_encoder_layers, num_decoder_layers)

