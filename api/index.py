from flask import Flask
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    df = pd.read_csv('diet_case_base.csv')
    return df.to_html()

@app.route('/about')
def about():
    return 'About'