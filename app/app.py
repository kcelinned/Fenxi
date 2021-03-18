from flask import Flask, request ,jsonify
import twitter
from analyser import get_predictions, get_results
import json

app = Flask(__name__)

@app.route('/')
def index():
    return "This is the Home Page"

@app.route('/predict', methods=['POST'])
def sentimetntAnalysis():
    input = request.form['query'] # will change to form 
    df = twitter.get_tweets(input)
    predicted_df = get_predictions(df)

    return jsonify(get_results(predicted_df))


if __name__ == 'main':
    app.run(debug = True)