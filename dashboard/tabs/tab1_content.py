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

# TAB 1: EXPLORATORY DATA ANALYSIS ###################################################################################################
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.H2("Introduction"),
            html.P(
                "We have at our disposal multivariate data consisting of 3 potential explonatory variables (keyword, location, text) and response variable (target). Variables 'keyword', 'location' and 'text' are nominal. The outcome variable 'target' is of binary type."
            ),
            # This is not quite right, that's just an estimate
            html.P("In total we have a sample of 10,000."),
            # DATA QUALITY ISSUES ##############################################################
            html.H2("Data Quality Issues"),
            html.P("We were able to discover the following data quality issues:"),
            html.Ul(
                [
                    html.Li("Duplicated rows with opposing target values"),
                    html.Li("Missing responses in column 'location'"),
                    html.Li("Fake responses in column 'location'"),
                    html.Li("Location appearing in multipe formats: country, city"),
                    html.Li("Missing values in column 'keyword'"),
                    html.Li("Strange characters appering in 'keyword' and 'text'"),
                ]
            ),
            # KEYWORD #########################################################################
            # html.H2("Keyword"),
            # html.P("Description of variable 'keyword'."),
            # html.P("WORDCLOUD"),

            # LOCATION #########################################################################
            html.H2("Location"),
            # What does the variable 'location' represent?
            html.P(
                "The variable 'location' represents the responses that Twitter users gave about where they were from. Out of 10,000 users, 7467 submited any response at all giving the response rate of 74.7%. One would expect structured data in the form 'country-city', but that is not the case. Users submitted their location data with the help of a text input. So, apart from responses like 'France', 'Germany' or 'USA' the column contains values like 'milky way', 'Worldwide' or 'Your Sister's Bedroom'. When geographical name is given at all, it often isn't given in a standard format. For example, the variable contains values like 'Jaipur, India' (city, country), 'bangalore' (just city), 'Indonesia' (just country) and many other variations of country, city, province or other administrative divison name in different combinations and no one uniform order."
            ),
            # How we will transform the data to make it useful for analysis
            html.P(
                "Nevertheless, thanks to so some data crunching we were able to make use of the largest possible subset of responses that contained geohraphical names."
            ),
            # How many non-null responses contained geographical names that we could link with specific countries?
            html.P(
                "We were able to obtain the total of X number of legitimate country responses."
            ),
            html.P(
                "Since the response rate in the variable 'keyword' (that contains names of disasters) was 99.4%, we have the types of a catastrophy for almost all countries. The map below represents the most disastered countries in total and by type* of a catastrophy."
            ),
            html.P(
                "*It is worth noting, that the number of distinct values in the column 'keyword' was almost X. This is not because there are so many types of disasters out there. One of the reasons, why that's the case is because sometimes the same type of disaster like for example 'fire' is given in different ways like 'explosion', 'fire responders' or 'flames'. Hence we combined values in the following groups: ()."
            ),
            html.P(
                "Map 1. Total number of disasters by country and types of disasters"
            ),
            dbc.Row([
                dbc.Col([
                    html.H3("Select disaster type",
                            style={"fontSize": 20}),
                    dbc.RadioItems(
                        options=[
                            {"label": "All types", "value": "all"},
                            {"label": "Fire", "value": "fire"},
                            {"label": "Explosion",
                             "value": "explosion"},
                            {"label": "Transport",
                             "value": "transport"},
                            {"label": "Terrorism",
                             "value": "terrorism"},
                            {"label": "Construction",
                             "value": "construction"},
                            {"label": "Wind", "value": "wind"},
                            {"label": "Flooding", "value": "flooding"},
                            {"label": "Hot weather", "value": "hot"},
                            {"label": "Tectonics",
                             "value": "tectonics"},
                            {"label": "Famine", "value": "famine"},
                            {"label": "Errosion", "value": "errosion"},
                            {"label": "Lightening",
                             "value": "lightening"},
                            {"label": "Mass murder", "value": "mass"},
                            {"label": "Nuclear", "value": "nuclear"},
                            {"label": "Industrial",
                             "value": "industrial"},
                            {"label": "Disease", "value": "disease"},
                            {"label": "Riot", "value": "riot"},
                            {"label": "War", "value": "war"},
                            {"label": "Unidentified",
                             "value": "Unidentified"},
                        ],
                        value="all",
                        id="location-radio-items",
                    )], width=2),
                dbc.Col([dcc.Graph(id="map_from_pgo")], width=10),
            ]),
            # TEXT #################################################################################################################
            html.H2("Text"),
            html.P("The column 'text' takes a special place in our analysis, because it is the only explanatory variable, that is, used in our NLP classification model. Values are strings. Each cell represents a separate tweet, so a short chunk of text, as Twitter limits the number of characters in a miniblog to 280."),
            # WORD FREQUENCY
            html.H3("WORD FREQUENCY", style={
                    "fontSize": 20}),
            html.P(
                "We can start, as is typical in analysing text data, from word frequency."),
            html.P("'Corpus' is an another name for a pool of all the words in a text file, column of table, or a paragraph of text. The pool of all words present in column 'text' has 31 924 distinct values. Values are in vast majority words but can also mean dashes or other punctuation characters."),
            html.P("Distributiom is right-skewed and resambles a Pareto distribution, otherwise known as a power-law distribution, where top words are much more frequent than the next most frequent words."),
            html.P("Use the slider to load the first X number of most frequent words."),
            dcc.Slider(0, 300, 10, value=170, id='n_of_words'),
            html.P("Fig 1. Most frequent words and their counts in raw data"),
            dcc.Graph(id="freq_w_stopwords"),
            html.P("Taking a first glance at the word frequency distribution, we discover that stopwords take up the first 50 most frequent words. Stopwords represent words like 'the', 'a', 'to', 'in', 'of', 'and', etc. This type of words is common in all kinds of text irregardles of the meaning. Therefore, we decided to perform the following data manipulations:"),
            html.Ul(
                [
                    html.Li("Change all text to lower case"),
                    html.Li(
                        "Splitting contractions e.g. you're -> you are, weren't -> were not"),
                    html.Li(
                        "Remove punctuation characters e.g. '!', '%', '&', '+', ',', '-', '.' etc."),
                    html.Li(
                        "Remove stopwords e.g. 'the', 'is', 'in', 'very', 'such', etc."),
                    html.Li("Remove hashtag while keeping hashtag text"),
                    html.Li("Remove HTML special entities (e.g. &amp;)"),
                    html.Li("Remove tickers"),
                    html.Li("Remove hyperlinks"),
                    html.Li("Remove whitespace (including new line characters)"),
                    html.Li("Remove URL, RT, mention(@)"),
                    html.Li(
                        "Remove characters beyond Basic Multilingual Plane (BMP) of Unicode"),
                    html.Li("Remove Remove emoji"),
                    html.Li("Remove mojibake (also extra spaces)"),
                ]
            ),
            html.P("Hence, we obtaine the following data:"),
            dcc.Slider(0, 300, 10, value=100, id='n_of_words_cleaned'),
            html.P("Fig 2. Most frequent words and their counts in cleaned data"),
            dcc.Graph(id="freq_w_cleaned"),
            html.P("Fig 1. Word distribution of values of variable 'text' combined'"),
            html.Div([
                html.Img(id="image_wc", width="100%"),
            ]),
            html.P("What are the most common words in either of classes?"),

            # VERSION 2.0
            # # PARTS OF SPEECH
            # html.H3("PARTS OF SPEECH", style={
            #     "fontSize": 20}),

            # # SENTIMENT ANALYSIS
            # html.H3("SENTIMENT ANALYSIS", style={
            #     "fontSize": 20}),
            # html.P(
            #     "Sentiment analysis is a very common natural language processing task in which we determine if the text is positive, negative or neutral. It could be possible that tweets relating to distasters are more negative. The way to check that is by looking at a metric called 'polarity'. Polarity is a floating-point number that lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement."),
            # html.P("What is the polarity of entire dataset?"),
            # html.P("How does polarity differ between groups?"),
            # html.P("DATA VIZ FOR POLARITY",),

            # # SPELLING ERRORS
            # html.H3("SPELL ERRORS", style={
            #     "fontSize": 20}),

            # # LEMMATIZATION
            # html.H3("SENTIMENT ANALYSIS", style={
            #     "fontSize": 20}),

            # # NAMED ENTITY RECOGNITION
            # html.H3("NAMED ENTITY RECOGNITION", style={
            #     "fontSize": 20}),

            # # CHUNKING
            # html.H3("CHUNKING", style={
            #     "fontSize": 20}),

            # # UNSUPERVISED LEARNING
            # html.H3("UNSUPERVISED LEARNING", style={
            #     "fontSize": 20}),

            # RESPONSE VARIABLE
            html.H2("Dataset Balance"),
            html.P("Is the dataset balanced?"),
            html.P("BAR CHART"),
        ]
    ),
    className="mt-3",
)
