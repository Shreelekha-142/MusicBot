from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import random
import csv
import spacy
from bot_responses import responses

app = Flask(__name__)

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load songs from CSV files
songs = {
    'positive': [],
    'negative': [],
    'neutral': []
}

def load_songs_from_csv(csv_filename, sentiment, language):
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            song = {
                'name': row['Name'].strip(),
                'artist': row['Artist'].strip(),
                'language': language
            }
            songs[sentiment].append(song)

load_songs_from_csv('English-pop.csv', 'positive', 'english')
load_songs_from_csv('English-unwind.csv', 'negative', 'english')
load_songs_from_csv('Hindi-pop.csv', 'positive', 'hindi')
load_songs_from_csv('Hindi-unwind.csv', 'negative', 'hindi')
load_songs_from_csv('kannada.csv', 'positive', 'kannada')
load_songs_from_csv('Punjabi.csv', 'positive', 'punjabi')
load_songs_from_csv('telugu.csv', 'positive', 'telugu')

# Counter for sentiment analysis
count = 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    global count
    user_text = request.args.get('msg')

    # Preprocess user input using BERT tokenizer
    inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    sentiment_logits = outputs.logits
    sentiment = torch.argmax(sentiment_logits, dim=1).item()

    # Extract keywords from user input using spaCy
    keywords = extract_keywords(user_text)

    # Get the chatbot response
    chatbot_response = generate_chatbot_response(user_text)

    # Check if the user wants song recommendations
    if any(keyword in user_text.lower() for keyword in ['pop songs', 'english songs', 'hindi songs', 'unwind', 'kannada songs', 'punjabi songs', 'telugu songs', 'songs']):
        return get_song_playlist(keywords)
    else:
        return chatbot_response

@app.route("/get-songs")
def get_songs():
    user_text = request.args.get('msg')

    # Preprocess user input using BERT tokenizer
    inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    sentiment_logits = outputs.logits
    sentiment = torch.argmax(sentiment_logits, dim=1).item()

    # Determine emotion based on sentiment
    if sentiment == 1:
        emotion = 'positive'
    else:
        emotion = 'negative'

    # Fetch songs based on the detected emotion
    if emotion == 'positive':
        with open('happy.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            recommended_songs = [{'name': row['Name'], 'album': row['Album'], 'artist': row['Artist']} for row in reader]
    else:
        with open('sad.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            recommended_songs = [{'name': row['Name'], 'album': row['Album'], 'artist': row['Artist']} for row in reader]

    # Shuffle the list of recommended songs and select first 5
    random.shuffle(recommended_songs)
    selected_songs = recommended_songs[:5]

    # Generate HTML code for table with styling and interactivity
    playlist_html = """
    <style>
    /* CSS styling for playlist */
    </style>
    """

    playlist_html += "<h1>Recommended Songs</h1>"
    playlist_html += "<table>"
    playlist_html += "<tr><th>Song Name</th><th>Album</th><th>Artist</th></tr>"
    for song in selected_songs:
        # Check if 'album' key exists in the song dictionary before accessing it
        album = song.get('album', '')  # If 'album' key doesn't exist, set it to empty string
        playlist_html += f"<tr><td>{song['name']}</td><td>{album}</td><td>{song['artist']}</td></tr>"
    playlist_html += "</table>"

    return playlist_html

def extract_keywords(text):
    keywords = []
    doc = nlp(text.lower())
    for token in doc:
        if token.text in ['pop', 'english', 'hindi', 'unwind', 'punjabi', 'kannada', 'telugu']:
            keywords.append(token.text)
    return keywords

def generate_chatbot_response(user_text):
    # Check if the user input matches any key in the responses dictionary
    for key in responses:
        if key.lower() in user_text.lower():
            # If a matching key is found, return a random response from its corresponding list
            return random.choice(responses[key])

    # If no matching key is found, return a default response
    default_responses = ["I'm sorry, I didn't understand that.",
                         "Could you please rephrase your question?",
                         "Let me think about that..."]
    return random.choice(default_responses)

def get_song_playlist(keywords):
    # Determine user preferences based on keywords
    preference = ""
    if 'pop' in keywords:
        preference += "positive"  # Assuming pop songs are generally positive
    else:
        preference += "negative"  # Assuming other songs are generally negative

    if 'english' in keywords:
        language = 'english'
    elif 'hindi' in keywords:
        language = 'hindi'
    elif 'kannada' in keywords:
        language = 'kannada'
    elif 'punjabi' in keywords:
        language = 'punjabi'
    elif 'telugu' in keywords:
        language = 'telugu'
    else:
        language = 'english'  # Default language

    if 'unwind' in keywords:
        sentiment = 'negative'
    else:
        sentiment = 'positive'

    # Get the corresponding songs based on user preferences and language
    recommended_songs = [song for song in songs[preference] if song.get('language') == language]

    # Shuffle the list of recommended songs and select first 5
    random.shuffle(recommended_songs)
    selected_songs = recommended_songs[:5]

    # Generate HTML code for table with styling and interactivity
    playlist_html = """
    <style>
    /* CSS styling for playlist */
    </style>
    """

    playlist_html += "<h1>Recommended Songs</h1>"
    playlist_html += "<table>"
    playlist_html += "<tr><th>Song Name</th><th>Album</th><th>Artist</th></tr>"
    for song in selected_songs:
        # Check if 'album' key exists in the song dictionary before accessing it
        album = song.get('album', '')  # If 'album' key doesn't exist, set it to empty string
        playlist_html += f"<tr><td>{song['name']}</td><td>{album}</td><td>{song['artist']}</td></tr>"
    playlist_html += "</table>"

    return playlist_html

if __name__ == '__main__':
    app.run()
