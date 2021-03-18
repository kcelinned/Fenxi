from flask import Flask, request 
import twitter
from analyser import get_predictions, get_results
import json

app = Flask(__name__)

@app.route('/')
def index():
    return "This is the Home Page"

@app.route('/', methods=['POST'])
def sentimetntAnalysis():
    input = request.args['query'] # will change to form 
    df = twitter.get_tweets(input)
    predicted_df = get_predictions(df)

    return json.dumps(get_results(predicted_df))


if __name__ == 'main':
    app.run(debug = True)