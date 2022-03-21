from flask import Flask
from flask import request
import sys
import json
import requests
from spell import spell_corrector
import pickle
import nltk

import random
import torch
from models import NeuralNet
from nltk_utils import tokenize, bag_of_words

# ----------for chat bot engine----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()
#--------------------------------------------------

file1 = open('word_dictionary', "rb")
word_dictionary = pickle.load(file1)
file1.close()

app = Flask(__name__)

# put your facebook page access token and verify token here

PAGE_ACCESS_TOKEN = ''
VERIFY_TOKEN = ''


@app.route('/', methods=['GET'])
def handle_verification():
    if (request.args.get('hub.verify_token', '') == VERIFY_TOKEN):
        print("Verified")
        return request.args.get('hub.challenge', '')
    else:
        print("Wrong token")
        return "Error, wrong validation token"


@app.route('/', methods=['POST'])
def handle_message():
    '''
    Handle messages sent by facebook messenger to the applicaiton
    '''
    data = request.get_json()

    if data["object"] == "page":
        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:
                if messaging_event.get("message"):

                    sender_id = messaging_event["sender"]["id"]
                    recipient_id = messaging_event["recipient"]["id"]
                    message_text = messaging_event["message"]["text"]
                    send_message_response(
                        sender_id, parse_user_text(message_text))

    return "ok"


def send_message(sender_id, message_text):
    '''
    Sending response back to the user using facebook graph API
    '''
    r = requests.post("https://graph.facebook.com/v2.6/me/messages",
                      params={"access_token": PAGE_ACCESS_TOKEN},
                      headers={"Content-Type": "application/json"},
                      data=json.dumps({
                          "recipient": {"id": sender_id},
                          "message": {"text": message_text}
                      }))


def parse_user_text(user_text):
    '''
    use chatbot and return response
    '''
    user_text = user_text.lower()
    response = ''
    # spelling correction
    tokens = nltk.word_tokenize(user_text)
    sentance = spell_corrector(tokens, word_dictionary)

    # chat bot engine response
    tokenized_sentance = tokenize(sentance)
    X = bag_of_words(tokenized_sentance, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent["responses"])
    else:
        response = 'I do not understand.....'

    return (response)


def send_message_response(sender_id, message_text):
    sentenceDelimiter = ". "
    messages = message_text.split(sentenceDelimiter)

    for message in messages:
        send_message(sender_id, message)


if __name__ == '__main__':
    app.run(debug=True)
