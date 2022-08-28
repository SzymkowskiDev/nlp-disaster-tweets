# IMPORT LOCAL
from dashboard.src.models.production.generate_perf_report import generate_perf_report
from dashboard.src.models.production.vectorize_data import vectorize_data
#from models.production.preprocess_data import preprocess_data

# IMPORT EXTERNAL
import time
from dash import Dash, html, Input, Output, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import auc
import plotly.figure_factory as ff
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import io
import requests
from bs4 import BeautifulSoup
import re
import nltk
import seaborn as sns
import string
import json
import string
import emoji
import itertools
from collections import Counter
from scipy.stats import kstest

# TAB 1: EXPLORATORY DATA ANALYSIS ###################################################################################################
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.H2("Introduction"),
            html.P(
                "We have at our disposal multivariate data consisting of 3 potential explonatory variables ('keyword', 'location', 'text') and a response variable ('target')."),
            html.P("Variables 'keyword', 'location' and 'text' are categorical. The outcome variable 'target' is of binary type."),
            # This is not quite right, that's just an estimate
            html.P("In total we have a sample of 10 873 tweets. This is split into train dataset of 7 614 records and a validation set of 3 264 rows. Giving roughly a 70-30 ratio."),
            html.P(
                "As is adviced by related literarture, our EDA focuses solely on the train set."),
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
            html.H2("Keyword"),
            html.P("The column 'keyword' represents keywords associated with individual tweets. Each tweet is accompanied by only one keyword. In our dataset, keywords relate to disasters, so for example we have there values like 'fire', 'earthquake', 'airplane accident' or smilar."),
            html.P(
                "There are only 61 null values in the column, making it 99.2% complete."),
            html.P("There are 221 distinct values in the variable 'keyword'. So, for the purpose of summary, we were able to come up with the following 18 logical groupings:"),
            dbc.Row([
                dbc.Col(
                    [
                        html.H2("🔥 FIRE", style={
                                "color": "#32FBE2", "fontSize": "18px"}),
                        html.P("ablaze arson arsonist arson arsonist buildings burning buildings on fire burned burning burning buildings  bush fires fire fire truck first responders flames forest fire forest fires wild fires wildfire engulfed hellfire",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("💥 EXPLOSION",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("blew up blown up bomb bombed bombing detonate detonation explode exploded explosion loud bang",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("🚗 TRANSPORT",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("airplane accident   collide collided collision crash crashed wreck wreckage wrecked derail derailed derailment sinking sunk",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("🧨 TERRORISM",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("bioterrorism bioterror hijack hijacker  hijacking hostage hostages suicide bomb suicide bomber suicide bombing terrorism terrorist",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
            ]),
            dbc.Row([
                dbc.Col(
                    [
                        html.H2("🏗️ CONSTRUCTION", style={
                            "color": "#32FBE2", "fontSize": "18px"}),
                        html.P("bridge collapse collapse  collapsed structural failure",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("💨 WIND",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("cyclone  hurricane rainstorm snowstorm storm tornado typhoon whirlwind windstorm blizzard hail hailstorm sandstorm dust storm violent storm",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("🏔️ ERROSION",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("landslide  mudslide sinkhole avalanche cliff fall",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("☀️ HOT WEATHER",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("drought heat wave",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
            ]),
            dbc.Row([
                dbc.Col(
                    [
                        html.H2("🌋 TECTONICS", style={
                            "color": "#32FBE2", "fontSize": "18px"}),
                        html.P("earthquake epicentre lava seismic volcano",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("🌽 FAMINE",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("famine",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("🌊 FLOOD",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("deluge deluged drown drowned drowning  flooding floods flood tsunami",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),

                dbc.Col(
                    [
                        html.H2("⚡ LIGHTENING",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("lightning thunder thunderstorm",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
            ]),
            dbc.Row([
                dbc.Col(
                    [
                        html.H2("🩸 MASS MURDER", style={
                            "color": "#32FBE2", "fontSize": "18px"}),
                        html.P("mass murder mass murderer",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("☢️ NUCLEAR",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("meltdown military natural disaster nuclear disaster nuclear reactor radiation emergency",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("🏭 INDUSTRIAL",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("oil spill electrocute electrocuted chemical emergency",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("🦠 DISEASE",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("outbreak quarantine quarantined",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
            ]),
            dbc.Row([
                dbc.Col(
                    [
                        html.H2("👥 RIOT", style={
                            "color": "#32FBE2", "fontSize": "18px"}),
                        html.P("riot rioting",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                        html.H2("⚔️ WAR",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("war zone weapon weapons army battle refugees",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
                dbc.Col(
                    [
                    ], width=3),

                dbc.Col(
                    [
                        html.H2("",
                                style={"color": "#32FBE2", "fontSize": "18px"}),
                        html.P("",
                               style={"color": "#32FBE2", "fontSize": "14px"})
                    ], width=3),
            ]),
            html.P("We have created an additional category for all the generic words that relate to distasters like 'emergency' while not indicating any specific type:"),
            html.H2("🚨 UNIDENTIFIED",
                    style={"color": "#32FBE2", "fontSize": "18px"}),
            html.P("accident aftershock ambulance annihilated annihilation apocalypse armageddon  attack attacked   blaze blazing bleeding blight blood bloody body bag body bagging body bags casualties casualty catastrophe catastrophic crush crushed curfew  damage danger dead death  deaths debris demolish demolished demolition   desolate desolation destroy destroyed destruction devastated devastation disaster displaced  emergency emergency plan emergency services evacuate evacuated evacuation  eyewitness fatal fatalities fatality fear flattened   harm hazard hazardous   injured injuries injury inundated inundation  massacre mayhem obliterate obliterated  obliteration   pandemonium panic panicking  police  razed rescue rescued rescuers   rubble ruin  screamed screaming screams  siren sirens smoke  stretcher   survive survived survivors  threat tragedy trapped trauma traumatised trouble  twister  upheaval wounded wounds",
                   style={"color": "#32FBE2", "fontSize": "14px"}),
            html.P("When the generic category is excluded, we see that the most common types of disasters are accidents in all means of transportation (car, train, plane, marine) as well as related to the effects of wind, fires, floodings, terrorism and explosions."),
            dbc.RadioItems(
                options=[
                    {"label": "All groups", "value": 1},
                    {"label": "Exclude 'UNIDENTIFIED'", "value": 2},
                ],
                value=2,
                id="groups-input",
                inline=True,
                style={"float": "right"}
            ),
            html.P("Fig 1. Frequencies of disasters by category"),
            dbc.Spinner([dcc.Graph(id="barplot_groups")],
                        color="success", spinner_style={"width": "8rem", "height": "8rem"}),
            # LOCATION #########################################################################
            html.H2("Location"),
            html.P(
                "The variable 'location' represents the responses that Twitter users gave about where they were from. Out of 7 614 users considered, only 5 081 submited any response at all giving the response rate of 66.7%."),
            html.P("One would expect structured data in the form 'country-city', but that is not the case. Users submitted their location data with the help of a text input. So, apart from legitimate responses like 'France', 'Germany' or 'USA' the column contains also entries like 'milky way', 'Worldwide' or 'Your Sister's Bedroom'. When a geographical name is given at all, it often isn't given in a standard format. For example, the variable contains values like 'Jaipur, India' (city, country), 'bangalore' (just city), 'Indonesia' (just country) and many other variations of country, city, province or other administrative divison name in different combinations and no one uniform order."),
            html.P(
                "Nevertheless, thanks to so some data crunching we were able to make use of the largest possible subset of responses that contained geohraphical names."
            ),
            html.P(
                "Out of 5 081 non-null responses we were able to obtain the total of 4 132 of legitimate country responses (81.3%). So, for the entire dataset 54.3% (4132/7614) of tweets had legitimate location given."
            ),
            html.P("Fig 2. Proportions of null/non-null and legitimate/fake responses"),
            # This input is hidden, it is here to make sankey work
            dbc.RadioItems(
                options=[
                    {"label": "All groups", "value": 1},
                    {"label": "Exclude 'UNIDENTIFIED'", "value": 2},
                ],
                value=2,
                id="sankey-input",
                inline=True,
                style={'display': 'none'}
            ),
            dbc.Spinner([dcc.Graph(id="sankey-legit-location")],
                        color="success", spinner_style={"width": "8rem", "height": "8rem"}),
            # BLOCKED: by Stim
            html.P("By further filtering our sample of 4 132 tweets to include only those where disastered did happen ('target' = 1) we obtain a sample of X."),
            html.P("This allows us to make the observation that the highest number of disasters were tweeted from: A, B, C, D."),
            html.P(
                "Map 3. Frequencies of disaster types by country"
            ),
            dbc.Row([
                dbc.Col([
                    html.H3("Select disaster type",
                            style={"fontSize": 20}),
                    dbc.RadioItems(
                        options=[
                            {"label": "All types", "value": "total"},
                            {"label": "🔥 Fire", "value": "fire"},
                            {"label": "💥 Explosion",
                             "value": "explosion"},
                            {"label": "🚗 Transport",
                             "value": "transport"},
                            {"label": "🧨 Terrorism",
                             "value": "terrorism"},
                            {"label": "🏗️ Construction",
                             "value": "construction"},
                            {"label": "💨 Wind", "value": "wind"},
                            {"label": "🌊 Flooding", "value": "flooding"},
                            {"label": "☀️ Hot weather", "value": "hot_weather"},
                            {"label": "🌋 Tectonics",
                             "value": "tectonics"},
                            {"label": "🌽 Famine", "value": "famine"},
                            {"label": "🏔️ Errosion", "value": "errosion"},
                            {"label": "⚡ Lightening",
                             "value": "lightening"},
                            {"label": "🩸 Mass murder", "value": "mass_murder"},
                            {"label": "☢️ Nuclear", "value": "nuclear"},
                            {"label": "🏭 Industrial",
                             "value": "industrial"},
                            {"label": "🦠 Disease", "value": "disease"},
                            {"label": "👥 Riot", "value": "riot"},
                            {"label": "⚔️ War", "value": "war"},
                            {"label": "🚨 Unidentified",
                             "value": "unidentified"},
                        ],
                        value="total",
                        id="location-radio-items",
                    )], width=2),
                dbc.Col([dbc.Spinner([dcc.Graph(id="map_from_pgo")],
                        color="success", spinner_style={"width": "8rem", "height": "8rem"}), ], width=10),
                html.P(
                    "Caveat 1. The above map represents data where 'target'=1, so records labelled as actual disasters."),
                html.P("Caveat 2. Locations should be thought of as approximates. That's because we take into account location of the profile posting a tweet. The tweet itself could, nevertheless, relate to a disaster happening in another country.")
            ]),
            # TEXT #################################################################################################################
            html.H2("Text"),
            html.P("The column 'text' takes a special place in our analysis. This is because it is the only explanatory variable, that is used in our NLP classification model. Each value represents a separate tweet, so a short chunk of text, as Twitter limits the number of characters in a miniblog to 280."),
            # WORD FREQUENCY
            html.H3("WORD FREQUENCY", style={
                    "fontSize": 20}),
            html.P(
                "We can start, as is typical in analysing text data, from word frequency."),
            html.P("'Corpus' is an another name for a pool of all the words in a text file, column of table, or a paragraph of text. The pool of all words present in column 'text' totals has 31 924 distinct values. Values are in vast majority words but can also mean dashes or other punctuation characters."),
            html.P("Distributiom is right-skewed and resambles a Pareto distribution, otherwise known as a power-law distribution, where top words are much more frequent than the next most frequent words."),
            html.P("Choose number of top words to display"),
            dcc.Slider(0, 300, 10, value=60, id='n_of_words'),
            html.P("Fig 4. Most frequent words and their counts in raw data"),
            dbc.Spinner([dcc.Graph(id="freq_w_stopwords")], color="success", spinner_style={
                        "width": "8rem", "height": "8rem"}),
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
            html.P("This resulted in the following data:"),
            dcc.Slider(0, 300, 10, value=100, id='n_of_words_cleaned'),
            html.P("Fig 5. Most frequent words and their counts in cleaned data"),
            dbc.Spinner([dcc.Graph(id="freq_w_cleaned")], color="success", spinner_style={
                        "width": "8rem", "height": "8rem"}),
            html.P(
                "Alternatively, we can get the sense of most common words by looking at the wordcloud below:"),
            html.P("Fig 6. Most common words in cleaned text data"),
            html.Div([
                html.Img(id="image_wc", width="100%"),
            ]),
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
            html.H2("Target"),
            html.P("Finally, we come to the response variable 'target'. It is a labelling variable that designates rows to one of two classes."),
            html.P("Most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class."),
            html.P(
                "The ratio of classes is 57% to 43%, so we can call this dataset balanced."),
            html.P("Fig 7. Dataset Balance"),
            # This input is hidden, it is here to make balance output work
            dbc.RadioItems(
                options=[
                    {"label": "All groups", "value": 1},
                    {"label": "Exclude 'UNIDENTIFIED'", "value": 2},
                ],
                value=2,
                id="balance-input",
                inline=True,
                style={'display': 'none'}
            ),
            dbc.Spinner([dcc.Graph(id="balance-output")], color="success",
                        spinner_style={"width": "8rem", "height": "8rem"}),
        ]
    ),
    className="mt-3",
)
