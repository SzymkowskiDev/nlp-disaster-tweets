# IMPORT LOCAL
from dashboard.src.tabs.tab1_content import tab1_content
from dashboard.src.tabs.tab2_content import tab2_content
from dashboard.src.tabs.tab3_content import tab3_content
from dashboard.src.tabs.tab4_content import tab4_content
from dashboard.src.tabs.tab5_content import tab5_content
from dashboard.src.tabs.tab6_content import tab6_content
from dashboard.src.models.production.generate_perf_report import generate_perf_report
from dashboard.src.models.production.vectorize_data import vectorize_data
from dashboard.src.models.production.preprocess_data import preprocess_data
from dashboard.src.models.production.make_a_prediction import make_a_prediction

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
import os
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

## PATHS ##
# TRAIN_DATA_PATH = r"data/original/train.csv"
# CLASS_CHART_PATH = r"data/class_chart/class_chart.csv"
# WORD_FREQ_LEMMATIZED_PATH = r"data/word_frequencies/word-freq_lemmatized.csv"
# WORD_FREQ_PATH = r"data/keyword/group_frequencies.csv"


TRAIN_DATA_PATH = os.path.join(
    os.getcwd(), "dashboard", "src", "data", "original", "train.csv")
CLASS_CHART_PATH = os.path.join(
    os.getcwd(), "dashboard", "src", "data", "class_chart", "class_chart.csv")
WORD_FREQ_LEMMATIZED_PATH = os.path.join(os.getcwd(
), "dashboard", "src", "data", "word_frequencies", "word-freq_lemmatized.csv")
WORD_FREQ_PATH = os.path.join(
    os.getcwd(), "dashboard", "src", "data", "keyword", "group_frequencies.csv")


## FUNCTIONS ##
def get_corpus(text):
    words = []
    for i in text:
        i = str(i)
        for j in i.split():
            words.append(j.strip())
    return words


# IMPORT DATA
df = pd.read_csv(TRAIN_DATA_PATH)
dummy_class = pd.read_csv(CLASS_CHART_PATH)
word_freqs_l = pd.read_csv(WORD_FREQ_LEMMATIZED_PATH)
word_freqs_g = pd.read_csv(WORD_FREQ_PATH)

# get corpus of list
corpus = get_corpus(df.text)
nlp_words = nltk.FreqDist(corpus)
list_of_words = list(nlp_words.keys())
list_of_counts = list(nlp_words.values())

# WORDCLOUD DATA
word_freqs = pd.DataFrame({'word': list_of_words, 'freq': list_of_counts})
word_freqs = word_freqs.sort_values(by=["freq"], ascending=False)


# APP SATRT
app = Dash(__name__, title="NLP Disaster Tweets", external_stylesheets=[
    dbc.themes.VAPOR, dbc.icons.BOOTSTRAP], meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5"},
])


# TAB 1: EXPLORATORY DATA ANALYSIS ###################################################################################################

# SANKEY diagram
@app.callback(Output("sankey-legit-location", "figure"), Input("sankey-input", "value"))
def update_sankey_chart(value):
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="perpendicular",
                node=dict(
                    pad=40,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[
                        "Total",
                        "Non-null",
                        "Null",
                        "Legitimate",
                        "Not legitimate",
                        "Null",
                    ],
                    color=[
                        "#636EFA",
                        "#48EF7B",
                        "#D85360",
                        "#48EF7B",
                        "#D85360",
                        "#D85360",
                    ],
                ),
                link=dict(
                    source=[0, 0, 1, 1, 2],
                    target=[1, 2, 3, 4, 5],
                    value=[5081, 2533, 4132, 949, 2533],
                    color=["#48EF7B", "#D85360",
                           "#48EF7B", "#D85360", "#D85360"],
                ),
            )
        ]
    )

    fig.update_layout(
        hovermode="x",
        margin=dict(t=0, l=50),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=14, color="black"),
    )

    return fig


# WORDCLOUD (looks a bit difffrent from other outputs)
def plot_wordcloud(data):
    d = {a: x for a, x in data.values}

    wc = WordCloud(
        background_color="black",
        width=800,
        height=360,
        max_words=1000,
        colormap="plasma",
    )

    wc.fit_words(d)
    return wc.to_image()


# TODO:???
@app.callback(Output('image_wc', 'src'), [Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    plot_wordcloud(data=word_freqs_l).save(img, format="PNG")
    return "data:image/png;base64,{}".format(base64.b64encode(img.getvalue()).decode())


# BARCHART 1
@app.callback(Output("freq_w_stopwords", "figure"), Input("n_of_words", "value"))
def update_bar_chart(value):
    ndf = word_freqs.iloc[:value]
    fig = px.bar(
        ndf,
        x="word",
        y="freq",
        labels={
            "freq": "Frequency",
            "word": "",
        },
        text_auto=True,
    )
    fig.update_layout(
        margin=dict(t=0, l=50),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=14, color="#32FBE2"),
    )
    return fig


# BARCHART 2
@app.callback(Output("freq_w_cleaned", "figure"), Input("n_of_words_cleaned", "value"))
def update_bar_chart(value):
    ndf = word_freqs_l.iloc[:value]
    fig = px.bar(
        ndf,
        x="word",
        y="freq",
        text_auto=True,
        labels={
            "freq": "Frequency",
            "word": "",
        },
    )
    fig.update_layout(
        margin=dict(t=0, l=50),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=14, color="#32FBE2"),
    )
    return fig


# LOCATION MAP
@app.callback(Output("map_from_pgo", "figure"), Input("location-radio-items", "value"))
def update_location_map(value):
    df2 = pd.read_csv(
        os.path.join(os.getcwd(), "dashboard", "src", "data",
                     "location", "location_totals.csv")
    )
    df2 = df2[["country", value]]

    # DATA TRANSFORMATIONS TO FEED MAP
    fig = go.Figure(
        data=go.Choropleth(
            locations=df2["country"],
            z=df2[value],
            text=df2["country"],
            colorscale="Plasma",
            autocolorscale=False,
            reversescale=True,
            marker_line_color="darkgray",
            marker_line_width=0.5,
        )
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="orthographic",
            bgcolor="rgba(0,0,0,0)",
            lakecolor="#17082D",
            showocean=True,
            oceancolor="#17082D",
        ),
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14, color="#32FBE2"),
    )
    fig.update_traces(showscale=False)

    return fig


# FIRST BAR CHART
@app.callback(Output("barplot_groups", "figure"), Input("groups-input", "value"))
def update_bar_chart(value):
    ndf = word_freqs_g

    if value == 1:
        include = [
            "UNINDENTIFIED",
            "TRANSPORT",
            "WIND",
            "FIRE",
            "FLOODING",
            "TERRORISM",
            "EXPLOSION",
            "WAR",
            "TECTONICS",
            "ERROSION",
            "DISEASE",
            "LIGHTENING",
            "CONSTRUCTION",
            "RIOT",
            "NUCLEAR",
            "INDUSTRIAL",
            "FAMINE",
            "HOT WEATHER",
        ]
        ndf = ndf[ndf["word"].isin(include)]
    elif value == 2:
        exclude = [
            "TRANSPORT",
            "WIND",
            "FIRE",
            "FLOODING",
            "TERRORISM",
            "EXPLOSION",
            "WAR",
            "TECTONICS",
            "ERROSION",
            "DISEASE",
            "LIGHTENING",
            "CONSTRUCTION",
            "RIOT",
            "NUCLEAR",
            "INDUSTRIAL",
            "FAMINE",
            "HOT WEATHER",
        ]
        ndf = ndf[ndf["word"].isin(exclude)]

    fig = px.bar(
        ndf,
        x="word",
        y="freq",
        text_auto=True,
        color="freq",
        color_continuous_scale="plasma",
        labels={
            "freq": "Frequency",
            "word": "",
        },
    )
    fig.update_layout(
        margin=dict(t=0, l=50),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=14, color="#32FBE2"),
    )
    return fig


# BALANCE PIE CHART
@app.callback(Output("balance-output", "figure"), Input("balance-input", "value"))
def update_pie_chart(value):
    # DATA
    # df = pd.read_csv(
    #     os.path.join(os.getcwd(), "dashboard", "src", "data", "original", "train.csv")
    #     )

    count_0 = df["target"].value_counts().loc[0].item()
    count_1 = df["target"].value_counts().loc[1].item()

    dfp = pd.DataFrame(
        {"Class": ["Class 0", "Class 1"], "Count": [count_0, count_1]})

    fig = px.pie(
        dfp,
        values="Count",
        names="Class",
        color_discrete_sequence=["#D85360", "#48EF7B"],
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        margin=dict(t=0, l=50),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=14, color="#32FBE2"),
    )

    return fig


# TAB 2: CLASSIFICATION ##############################################################################################################

# BARCHART
@app.callback(Output("class_barchart", "figure"), Input("intermediate-value", "data"))
def update_bar_chart(data):
    dff = pd.read_json(data, typ="series")
    tn, fp, fn, tp = np.array(dff.get("Confusion Matrix")).ravel()
    df = pd.DataFrame(
        {
            "Disaster": ["No", "Yes", "No", "Yes"],
            "Actual": ["TRUE", "TRUE", "FALSE", "FALSE"],
            "Count": [tn, tp, fn, fp],
        }
    )
    fig = px.bar(
        df,
        x="Disaster",
        y="Count",
        color="Actual",
        barmode="group",
        text_auto=True,
        color_discrete_sequence=["#48EF7B", "#D85360", "#48EF7B", "#D85360"],
    )
    fig.update_layout(
        margin=dict(t=0, l=50),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=14, color="#32FBE2"),
    )

    return fig


# INTERIM DATA
@app.callback(
    Output("intermediate-value", "data"),
    [
        Input("preprocessing-checklist", "value"),
        Input("vectorization-radio-items", "value"),
        Input("model-radio-items", "value"),
    ],
)
def our_function(preprocessing_checklist, vectorization, model):
    tic = time.time()
    global df
    df_train = df.copy()

    # Data Cleaning
    df_train = preprocess_data(df_train, preprocessing_checklist)

    # Vectorization
    X = vectorize_data(df_train, vectorization)
    y = df_train["target"].copy()

    # Model
    if model == "Logistic":
        clf = LogisticRegression
    elif model == "SVC":
        clf = SVC
    elif model == "Bayes":
        clf = MultinomialNB

    series = generate_perf_report(X, y, clf=clf())
    toc = time.time()
    # series = series.append({"Time": (toc - tic)})
    series = pd.concat([series, pd.Series({"Time": (toc - tic)})])

    print("Time: ", "{:.3f}".format(toc - tic), " seconds")
    return series.to_json(date_format="iso")


# PERFORMANCE METRICS TABLE
@app.callback(Output("output-datatable", "children"), Input("intermediate-value", "data"))
def update_datatable(data):
    dff = pd.read_json(data, typ="series")
    return html.Div(
        [
            html.Table(
                [
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Th(
                                        "Accuracy", style={"padding": "8px 8px 4px 8px"}
                                    ),
                                    html.Th(
                                        round(dff.get("Accuracy"), 2),
                                        style={"width": 50},
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Th(
                                        "Precision",
                                        style={"padding": "8px 8px 4px 8px"},
                                    ),
                                    html.Th(round(dff.get("Precision"), 2)),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Th(
                                        "Recall", style={"padding": "8px 8px 4px 8px"}
                                    ),
                                    html.Th(round(dff.get("Recall"), 2)),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Th(
                                        "F1 Score", style={"padding": "8px 8px 4px 8px"}
                                    ),
                                    html.Th(round(dff.get("F1 Score"), 2)),
                                ]
                            ),
                        ]
                    )
                ],
                style=dict(
                    textAlign="left",
                    padding=15,
                    margin=10,
                    border="1px solid #32FBE2",
                    backgroundColor="#3F3B96",
                    fontWeight="bold",
                    color="white",
                    width="85%",
                ),
            )
        ]
    )
    pass


# PERFORMANCE METRICS TEXT - ACCURACY
@app.callback(Output("performance-metrics-accuracy-text", "children"), Input("intermediate-value", "data"),
              )
def update_performance_metrics_accuracy_text(data):
    dff = pd.read_json(data, typ="series")
    tn, fp, fn, tp = np.array(dff.get("Confusion Matrix")).ravel()
    return (
        html.P(
            f"Your customized model has been tried out on a test sample of {fn + fp + tn + tp} tweets in {round(dff.get('Time'), 4)} seconds. "
            f"It correctly classified {tn + tp} of records, while the remaining {fp + fn} were assigned to a wrong class. "
            f"This means that the accuracy is {dff.get('Accuracy')*100:.2f}%."
        ),
    )


# PERFORMANCE METRICS TEXT - PRECISION
@app.callback(Output("performance-metrics-precison-text", "children"), Input("intermediate-value", "data"),)
def update_performance_metrics_precision_text(data):
    dff = pd.read_json(data, typ="series")
    tn, fp, fn, tp = np.array(dff.get("Confusion Matrix")).ravel()
    return html.P(
        f"The classifier has marked a total of {fp + tp} tweets as those that relate to natural disasters (class 1). Out of these"
        f", {tp} were actually in this group. This gives precision of {dff.get('Precision')*100:.2f}%"
    )


# PERFORMANCE METRICS TEXT - RECALL
@app.callback(Output("performance-metrics-recall-text", "children"), Input("intermediate-value", "data"),)
def update_performance_metrics_recall_text(data):
    dff = pd.read_json(data, typ="series")
    tn, _, fn, _ = np.array(dff.get("Confusion Matrix")).ravel()
    return html.P(
        f"The sample contained a total of {tn + fn} negatives. "
        f"Out of them, {tn} were correctly predicted by the model. Hence, the recall is {dff.get('Recall')*100:.2f}  %. "
        f"Finally, the harmonic mean of precison and recall is {dff.get('F1 Score')*100:.2f}%."
    )


# ROC GRAPH
@app.callback(Output("roc-graph", "figure"), Input("intermediate-value", "data"))
def update_roc(data):
    dff = pd.read_json(data, typ="series")
    fpr, tpr, _ = dff.get("Roc curve")
    fig = px.area(
        x=fpr, y=tpr, labels=dict(
            x="False Positive Rate", y="True Positive Rate")
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", color="red"), x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    fig.update_layout(
        margin=dict(t=10, l=50),
        height=350,
        width=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14, color="#32FBE2"),
    )

    fig.add_trace(
        go.Scatter(
            x=[0.7],
            y=[0.3],
            mode="text",
            text=[f"AUC = {auc(fpr, tpr):.4f}"],
            textposition="bottom center",
        )
    )

    fig.update_layout(showlegend=False)

    return fig


# TAB 3 - SOME FUNNY TEXT   ###########################################################################################################


# TAB 4 - MAKE A PREDICTION ###########################################################################################################

@app.callback(Output("output-outcome-of-prediction", "children"), Input("input-tweet-to-predict", "value"))
def on_text_input(value):

    prediction = make_a_prediction(df, value)

    # Conditional display on outcome
    if prediction == 0:
        return html.P("This is likely not about a disaster, no need for an alert.", style={"color": "#49EF7B"})
    elif prediction == 1:
        return html.P("This is likely an emergency, call 112.", style={"color": "#DA525E"})
    else:
        return html.P("Outcome will appear here.", style={"color": "#32FBE2"})


# TAB 5 - TWITTER BOT ANALYTICS ########################################################################################################

@app.callback(
    Output('analytics-output', 'children'),
    Input('analytics-input-dummy', 'value')
)
def get_analytics_data(value):

    # HTTP request to the REST API of the bot will go here

    # api_data = requests.get('https://API_DOMAIN_NAME/')
    # or
    # api_data = requests.get('https://API_DOMAIN_NAME/recent/{seconds}')

    api_data = 100

    return 'The data from the server is: \n{}'.format(api_data)


# FIRST BAR CHART


@app.callback(Output("bot-timeseries", "figure"), Input("bot-timeseries-input", "value"))
def update_bar_chart(value):

    # Define data for the data viz
    df = px.data.stocks()  # replace with your own data source
    fig = px.line(df, x='date', y="AMZN")

    # fig = px.bar(
    #     ndf,
    #     x="word",
    #     y="freq",
    #     text_auto=True,
    #     color="freq",
    #     color_continuous_scale="plasma",
    #     labels={
    #         "freq": "Frequency",
    #         "word": "",
    #     },
    # )
    fig.update_layout(
        margin=dict(t=0, l=50),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=14, color="#32FBE2"),
    )
    return fig


# TABS SETUP ###########################################################################################################################
tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Exploratory Data Analysis", tab_id="tab-1"),
        dbc.Tab(tab2_content, label="Classification", tab_id="tab-2"),
        dbc.Tab(
            tab3_content,
            label="Best performing",
            tab_id="tab-3",
        ),
        dbc.Tab(
            tab4_content,
            label="Make a prediction",
            tab_id="tab-4",
        ),
        dbc.Tab(
            tab5_content,
            label="Twitter BOT Analytics",
            tab_id="tab-5",
        ),
        dbc.Tab(
            tab6_content,
            label="About",
            tab_id="tab-6",
        ),
    ],
    active_tab="tab-2",
)

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# LAYOUT
app.layout = dbc.Container(
    [html.H1([html.I(className="bi bi-twitter me-2"), "NLP Disaster Tweets"]), tabs]
)

if __name__ == "__main__":
    app.run_server(debug=True)
