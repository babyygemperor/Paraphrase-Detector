from flask import Flask, request, render_template

from parrot import Parrot
import torch
import warnings
import spacy

warnings.filterwarnings("ignore")

app = Flask(__name__)

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")


def paraphrases(phrases):
    result = []
    for _, phrase in phrases.items():
        para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
        result.append(para_phrases)

    return result[0], result[1]


def find_similarity(phrases):
    p_1, p_2 = paraphrases(phrases)
    nlp = spacy.load("en_core_web_lg")

    maximum_similarity = 0.0
    for line_1 in p_1:
        for line_2 in p_2:
            similarity = nlp(str(line_1)).similarity(nlp(str(line_2)))
            if similarity > maximum_similarity:
                maximum_similarity = similarity

    return round(maximum_similarity, 6) * 100


@app.route('/predict', methods=["POST"])
def post_predict():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        json = request.json
        if 'sent_1' in json.keys() and 'sent_2' in json.keys():
            similarity = {'similarity': find_similarity(json), 'sent_1': json['sent_1'], 'sent_2': json['sent_2']}
            return similarity
        return "No requests found! Please put your prediction requests as {'sent_1' : 'Test string to predict', " \
               "'sent_2' : 'Test string 2 to predict'} "
    return 'Content-Type not supported!'


@app.route('/')
def my_form():
    return render_template('my-form.html')


@app.route('/', methods=["POST"])
def post_gui_predict():
    sent_1 = request.form['sent_1']
    sent_2 = request.form['sent_2']
    input_vals = {'sent_1': sent_1, 'sent_2': sent_2}
    return render_template('my-form.html', prediction="Prediction", result=f"{find_similarity(input_vals)}%:"
                                                                           f" \"{sent_1}\"; \"{sent_2}\"")
