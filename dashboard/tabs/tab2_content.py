# IMPORT LOCAL
from models.production.generate_perf_report import generate_perf_report
from models.production.vectorize_data import vectorize_data
from models.production.preprocess_data import preprocess_data

# IMPORT EXTERNAL
import time
from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import auc
import plotly.figure_factory as ff
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from dash import Dash, html, Output, Input
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import requests
from bs4 import BeautifulSoup
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import string
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import json
import re
import emoji
import itertools
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from scipy.stats import kstest

# LOAD FILES
df = pd.read_csv("data/original/train.csv")
text = df[['text']]
dfm = " ".join(df[df.target == 1].text)
TRAIN_DATA_PATH = r"data\original\train.csv"
# IMPORT DUMMY DATA FOR BAR CHART
dummy_class = pd.read_csv("data\class_chart\class_chart.csv")


# TAB 2: CLASSIFICATION ##############################################################################################################
tab2_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    html.H2("INPUTS"),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("Data Cleaning", style={
                                            "fontSize": 20}),
                                    dbc.Checklist(
                                        options=[
                                            {"label": "Remove hashes", "value": 1},
                                            {
                                                "label": "Remove HTML special entities",
                                                "value": 2,
                                            },
                                            {"label": "Remove tickers", "value": 3},
                                            {"label": "Remove hyperlinks",
                                                "value": 4},
                                            {"label": "Remove whitespaces",
                                                "value": 5},
                                            {
                                                "label": "Remove URL, RT, mention(@)",
                                                "value": 6,
                                            },
                                            {
                                                "label": "Remove no BMP characters",
                                                "value": 7,
                                            },
                                            {
                                                "label": "Remove misspelled words",
                                                "value": 8,
                                            },
                                            {"label": "Remove emojis", "value": 9},
                                            {"label": "Remove Mojibake",
                                                "value": 10},
                                            {
                                                "label": "Tokenize & Lemmatize",
                                                "value": 11,
                                            },
                                            {"label": "Leave only nouns",
                                                "value": 12},
                                            {
                                                "label": "Spell check",
                                                "value": 13,
                                                "disabled": True,
                                            },
                                        ],
                                        id="preprocessing-checklist",
                                    ),
                                ]
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("Vectorization", style={
                                            "fontSize": 20}),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Count", "value": "count"},
                                            {"label": "TF-IDF", "value": "tfidf"},
                                            {
                                                "label": "Word2Vec ",
                                                "value": "W2V",
                                                "disabled": True,
                                            }
                                            # TODO: implement this
                                        ],
                                        value="tfidf",
                                        id="vectorization-radio-items",
                                    ),
                                ]
                            )
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("Model", style={"fontSize": 20}),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "SVC", "value": "SVC"},
                                            {"label": "Logistic",
                                                "value": "Logistic"},
                                            {"label": "Naive Bayes",
                                                "value": "Bayes"},
                                            {
                                                "label": "LSTM ANN model",
                                                "value": "LSTM",
                                                "disabled": True,
                                            },
                                            {
                                                "label": "BERT model",
                                                "value": "BERT",
                                                "disabled": True,
                                            },
                                            {
                                                "label": "roBERTa model",
                                                "value": "roBERTa",
                                                "disabled": True,
                                            },
                                            # TODO: add other models
                                        ],
                                        value="Logistic",
                                        id="model-radio-items",
                                    ),
                                ]
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Button(
                                        "Run",
                                        color="success",
                                        outline=True,
                                        id="run",
                                        style={
                                            "borderRadius": "50%",
                                            "height": 110,
                                            "width": 110,
                                            "marginBottom": 20,
                                        },
                                    ),
                                    dbc.Button(
                                        "Reset",
                                        color="secondary",
                                        outline=True,
                                        id="reset",
                                        style={
                                            "borderRadius": "50%",
                                            "height": 110,
                                            "width": 110,
                                            "marginBottom": 20,
                                        },
                                    ),
                                    dbc.Button(
                                        "Save",
                                        color="primary",
                                        outline=True,
                                        id="save",
                                        style={
                                            "borderRadius": "50%",
                                            "height": 110,
                                            "width": 110,
                                            "marginBottom": 20,
                                        },
                                    ),
                                ],
                                className="d-grid gap-2",
                            ),
                        ],
                        width=3,
                    ),
                ]
            ),
            dbc.Row(
                [
                    html.H2("OUTPUTS", style={"marginTop": 25}),
                    dbc.Col(
                        [
                            html.Div(id="performance-metrics-accuracy-text"),
                            html.H3(
                                "Fig 1. Confusion Matrix data", style={"fontSize": 20}
                            ),
                            dcc.Graph(id="class_barchart"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(id="performance-metrics-precison-text"),
                            html.H3(
                                "Fig 2. Performance Metrics", style={"fontSize": 20}
                            ),
                            html.Div(
                                [
                                    html.Div(id="output-datatable"),
                                    dcc.Store(id="intermediate-value"),
                                ]
                            ),
                            html.P(id="performance-metrics-recall-text"),
                        ],
                        width=4,
                    ),
                    # dbc.Col([html.H3("Fig 2. Confusion Matrix", style={"fontSize": 20}),
                    #          dcc.Graph(id="confusion-matrix-graph")], width=4),
                    dbc.Col(
                        [
                            html.P(
                                "Fig 3. shows the performance of the classification model at all classification thresholds."
                            ),
                            html.H3("Fig 3. ROC & AUC",
                                    style={"fontSize": 20}),
                            dcc.Graph(id="roc-graph"),
                        ],
                        width=4,
                    ),
                ]
            ),
        ]
    ),
    className="mt-3",
)