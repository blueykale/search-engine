import flask
import panel as pn
from flask import Flask, render_template
from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import TextInput, Button
from bokeh.plotting import curdoc
from bokeh.themes import Theme
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
import numpy as np
import pandas as pd
import seaborn as sns
import string
import re
import os
import json
from PIL import Image
from IPython.display import display, clear_output, HTML, Markdown
import ipywidgets as widgets
import textwrap
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist, euclidean
import torch
import nltk
import ssl
from nltk.tokenize import word_tokenize
import openai
pn.extension()
import gensim.downloader as api
import spacy
from spacy.lang.en import English
from nltk.corpus import stopwords as nltk_stopwords
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from nltk.stem import WordNetLemmatizer

# ntlk download certificate error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')





# Combine stopwords from NLTK and SpaCy
STOP_WORDS = set(nltk_stopwords.words('english')).union(spacy_stopwords)

# Add custom word to the stopwords list
STOP_WORDS.add('learn')

def preprocess_user_input(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove stop words using combined stopwords list 
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    text = ' '.join(words)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatization - using NLTK's lemmatisation library 
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    
    return text

# Load BERT pre-trained model and tokenizer
model_bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

# # Function to calculate BERT embeddings
def calculate_bert_embedding(text):
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model_bert(**inputs)
    embeddings = outputs.last_hidden_state.mean(axis=1).detach().numpy()
    return embeddings[0]


# Load RoBERTa pre-trained model and tokenizer
model_roberta = RobertaModel.from_pretrained('roberta-base')
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')

# Function to calculate RoBERTa embeddings
def calculate_roberta_embedding(text):
    inputs = tokenizer_roberta(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model_roberta(**inputs)
    embeddings = outputs.last_hidden_state.mean(axis=1).detach().numpy()
    return embeddings[0]

def get_relevant_courses(prompt, df):
    """
    This function returns the relevant courses based on the user's input.
    """
    prompt = prompt.lower()
    relevant_courses = df[df['skill name'].str.lower().str.contains(prompt, case=False) | 
                          df['topic'].str.lower().str.contains(prompt, case=False) | 
                          df['description'].str.lower().str.contains(prompt, case=False)]
    return relevant_courses

# Define stuff that we require
chat_history = pn.Column()
inp = pn.widgets.TextInput(placeholder='Enter a skill...')
df_final = pd.read_csv('FINAL (processed bert embedded).csv')
# List of phrases that are considered as thanking
thank_you_phrases = ["thank you", "thanks", "appreciate it", "grateful", "much obliged", 
                        "cheers", "much appreciated", "thanks for the assistance"]
thank_you_phrases_preprocessed = []
for word in thank_you_phrases:
    thank_you_phrases_preprocessed.append(preprocess_user_input(word))

def collect_messages(button):
    """
    This function collects and processes the user messages.
    """
    print("Collect message")

    prompt = inp.value
    inp.value = ''

    # Preprocess user input
    user_input_processed = preprocess_user_input(prompt)    #### used to use preprocess or preprocess_glove 

    print("User input processed:", user_input_processed)
    # Check if the user_input_processed is in the list of thank you phrases. If it is, say 'you're welcome'
    if any(word in user_input_processed.lower() for word in thank_you_phrases_preprocessed):
        response = "You're welcome! I hope you find them useful in your learning journey!"
        
    # Initial check for input prompt
    elif not prompt.strip():
        response = "Welcome to IBM Skills Build!\nWhat would you like to learn more about?"
    
    else:
        # Check for direct matches in the dataset
        relevant_courses = get_relevant_courses(user_input_processed, df_final)
        
        # Generate the chatbot response based on direct matches
        if not relevant_courses.empty:
            response = generate_direct_matches_response(relevant_courses)
        
        else:
            # Calculate user embedding
            print("User input processed:", user_input_processed)
            user_embedding = calculate_bert_embedding(user_input_processed)

            if np.isnan(user_embedding).any():
                response = "Sorry, I couldn't understand your input. Please try again."
            else:
                # Calculate Euclidean distance between user input embedding and row embeddings
                topic_distance = cdist(df_final[[f"embedding_topic_{i}" for i in range(768)]], [user_embedding], metric='euclidean')
                type_distance = cdist(df_final[[f"embedding_type_{i}" for i in range(768)]], [user_embedding], metric='euclidean')
                skill_distance = cdist(df_final[[f"embedding_skill_{i}" for i in range(768)]], [user_embedding], metric='euclidean')
                description_distance = cdist(df_final[[f"embedding_description_{i}" for i in range(768)]], [user_embedding], metric='euclidean')

                # Combine distance scores (take the negative because we want to sort in descending order later)
                total_distance = -(topic_distance + type_distance + skill_distance + description_distance)

                # Retrieve most similar rows based on similarity scores
                top_indices = total_distance.argsort(axis=0)[-10:][::-1]  # Retrieve top 10 matches in descending order

                # Generate the chatbot response
                response = generate_similarity_response(top_indices, df_final)

    # print('response:', response)
    panels = []
    # Append the "User" row only when there is a non-empty prompt
    # if prompt.strip():
    #     panels.append(pn.Row('User:', pn.pane.Markdown(prompt, width=800)))
    
    # panels.append(
    #     pn.Row('Assistant:', pn.pane.HTML(response, width=800, style={'background-color': '#e8f5f8'})))

    if prompt.strip():
        chat_history.append(pn.Row('User:', pn.pane.Markdown(prompt, width=800)))

    chat_history.append(
        pn.Row('Assistant:', pn.pane.HTML(response, width=800, style={'background-color': '#e8f5f8'}))
    )

    return pn.Column(*panels)


def generate_similarity_response(top_indices, df_final, max_responses=10):
    """
    Generates the chatbot response for the input that doesn't directly match something in the dataset, based on similarity scores.
    """
    response = "Based on your interests, here are the top 10 skills we think you'll find useful:<br>Click (opens in new tab) on the skill name or the course name for the relevant IBM Skilld Build website!<br><br>"
    num_responses = min(max_responses, len(top_indices.flatten()))
    for i, idx in enumerate(top_indices.flatten(), 1):
        topic = df_final.loc[idx]['topic']
        skill_name = df_final.loc[idx]['skill name']
        link = df_final.loc[idx]['link']
        type_ = df_final.loc[idx]['type']
        description = df_final.loc[idx]['description']
        specific_link = df_final.loc[idx]['specific link']

        response += f'{i}. <a href="{link}" target="_blank">{topic}</a> - <a href="{specific_link}" target="_blank">{skill_name}</a> ({type_}):'
        response += f'   {description}<br><br>'
    return response


def generate_direct_matches_response(relevant_courses, max_responses=10):
    """
    Generates the chatbot response for the directly matched courses.
    """
    response = "Based on your interests, here are some relevant courses we think you'll find useful:<br> Click (opens in new tab) on the skill name or the course name for the relevant IBM Skilld Build website!<br><br>"
    
    num_responses = min(max_responses, len(relevant_courses))
    for i, course in enumerate(relevant_courses.iterrows(), 1):
        if i > num_responses:
            break
        topic = course[1]['topic']
        skill_name = course[1]['skill name']
        link = course[1]['link']
        type_ = course[1]['type']
        description = course[1]['description']
        specific_link = course[1]['specific link']

        response += f"{i}. <a href='{link}' target='_blank'>{topic}</a> - <a href='{specific_link}' target='_blank'>{skill_name}</a> ({type_}):"
        response += f"   {description}<br><br>"
    return response


app = Flask(__name__)

# Define a function to create a document
# The Panel application will be served by this
def create_document(doc):
    # The original Panel application code goes here

    panels = [] # collect display 

    # Initialize "Chat!" button 
    button_conversation = pn.widgets.Button(name="Chat!")
    # button_conversation.on_click(collect_messages)
    interactive_conversation = pn.bind(collect_messages, button_conversation)

    dashboard = pn.Column(
        inp,                             #input box
        pn.Row(button_conversation),     #followed by the button "Chat!" 
        chat_history,                    # display chat history
        pn.panel(interactive_conversation, loading_indicator=True, height=400)    #styling
    )

    doc.theme = Theme(json={
    "attrs": {
        "Plot": {"toolbar_location": None},
        "Grid": {"grid_line_color": None},
        "Axis": {
            "axis_line_color": None,
            "major_label_text_color": None,
            "major_tick_line_color": None,
            "minor_tick_line_color": None,
        }
    }
    })
    
    # Convert the Panel object to a Bokeh model
    bokeh_dashboard = dashboard.get_root()
    # Add the Bokeh model to the current Bokeh document
    doc.add_root(bokeh_dashboard)

@app.route('/', methods=['GET'])
# def bkapp_page():
#     script = server_document('http://localhost:5006/bkapp')
#     return render_template("embed.html", script=script, template="Flask")
def bkapp_page():
    # Use the SERVER_URL environment variable or default to localhost:5006
    server_url = os.environ.get("SERVER_URL", "https://localhost:5006")
    script = server_document(f'{server_url}/bkapp')
    return render_template("embed.html", script=script, template="Flask")

# def bk_worker():
#     # Can't pass num_procs > 1 in this configuration. If you need to run multiple
#     # processes, see e.g. flask_gunicorn_embed.py
#     server = Server({'/bkapp': create_document}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8080"])
#     server.start()
#     server.io_loop.start()
def bk_worker():
    # Use the WEBAPP_ORIGIN environment variable or default to localhost:8080
    webapp_origin = os.environ.get("WEBAPP_ORIGIN", "127.0.0.1:8080")
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': create_document}, io_loop=IOLoop(), allow_websocket_origin=[webapp_origin])
    server.start()
    server.io_loop.start()

from threading import Thread
Thread(target=bk_worker).start()

if __name__ == '__main__':
    print('Opening single process Flask app with embedded Bokeh application on http://localhost:8080/')
    print()
    print('Multiple connections may block the Bokeh app in this configuration!')
    print('See "flask_gunicorn_embed.py" for one way to run multi-process')
    # app.run(port=8080)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
