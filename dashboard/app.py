# IMPORT LOCAL
from dashboard.tabs.tab1_content import tab1_content
from dashboard.tabs.tab2_content import tab2_content
# from dashboard.tabs.tab3_content import tab3_content
# from dashboard.tabs.tab4_content import tab4_content
# from dashboard.tabs.tab5_content import tab5_content
# from dashboard.tabs.tab6_content import tab6_content
from dashboard.tabs.tab7_content import tab7_content
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

# APP
app = Dash(external_stylesheets=[dbc.themes.VAPOR, dbc.icons.BOOTSTRAP])

# IMPORT DATA
df = pd.read_csv("dashboard/data/original/train.csv")
text = df[['text']]
dfm = " ".join(df[df.target == 1].text)
TRAIN_DATA_PATH = r"dashboard\data\original\train.csv"
dummy_class = pd.read_csv("dashboard\data\class_chart\class_chart.csv")

word_freqs_l = pd.read_csv(
    "dashboard/data/word_frequencies/word-freq_lemmatized.csv")

word_freqs_g = pd.read_csv(
    "dashboard/data/keyword/group_frequencies.csv")

# TAB 1: EXPLORATORY DATA ANALYSIS ###################################################################################################

# WORDCLOUD DATA


def get_corpus(text):
    words = []
    for i in text:
        i = str(i)
        for j in i.split():
            words.append(j.strip())
    return words


corpus = get_corpus(df.text)
nlp_words = nltk.FreqDist(corpus)
list_of_words = list(nlp_words.keys())
list_of_counts = list(nlp_words.values())

word_freqs = pd.DataFrame({'word': list_of_words, 'freq': list_of_counts})
word_freqs = word_freqs.sort_values(by=["freq"], ascending=False)

# WORDCLOUD (looks a bit difffrent from other outputs)


def plot_wordcloud(data):
    d = {a: x for a, x in data.values}

    wc = WordCloud(background_color='black', width=800,
                   height=360, max_words=1000, colormap="plasma")

    wc.fit_words(d)
    return wc.to_image()


@app.callback(Output('image_wc', 'src'), [Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    plot_wordcloud(data=word_freqs_l).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# BARCHART 1


@app.callback(Output("freq_w_stopwords", "figure"), Input("n_of_words", "value"))
def update_bar_chart(value):
    ndf = word_freqs.iloc[:value]
    fig = px.bar(
        ndf,
        x="word",
        y="freq",
        # color="Actual",
        # barmode="group",
        text_auto=True
        # color_discrete_sequence=["#48EF7B", "#D85360", "#48EF7B", "#D85360"],
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
        # color="Actual",
        # barmode="group",
        text_auto=True
        # color_discrete_sequence=["#48EF7B", "#D85360", "#48EF7B", "#D85360"],
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


@app.callback(
    Output("map_from_pgo", "figure"), Input("location-radio-items", "value"))
def update_location_map(value):
    df2 = pd.read_csv("data/location/totals.csv")
    df2 = df2[["country", value]]
    # DATA TRANSFORMATIONS TO FEED MAP
    fig = go.Figure(data=go.Choropleth(
        locations=df2['country'],
        z=df2[value],
        text=df2['country'],
        colorscale='Plasma',
        autocolorscale=False,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        # colorbar_tickprefix='$',
        colorbar_title='Number of disasters',))
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="orthographic",
            bgcolor='rgba(0,0,0,0)',
            lakecolor="#17082D",
            showocean=True,
            oceancolor="#17082D"
        ),
        height=600, margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            size=14,
            color="#32FBE2"
        ),
        annotations=[dict(
            x=0.55,
            y=0.1,
            xref='paper',
            yref='paper',
            text='Source: Twitter',
            showarrow=False
        ),
        ]
    )

    return fig


# DATA VIZ:
@app.callback(Output("barplot_groups", "figure"), Input("groups", "value"))
def update_bar_chart(value):

    ndf = word_freqs_g.iloc[:value]

    fig = px.bar(
        ndf,
        x="word",
        y="freq",
        text_auto=True,
        color="freq",
        color_continuous_scale='plasma',
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
    df_train = pd.read_csv(TRAIN_DATA_PATH)

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


@app.callback(
    Output("output-datatable", "children"), Input("intermediate-value", "data")
)
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
@app.callback(
    Output("performance-metrics-accuracy-text", "children"),
    Input("intermediate-value", "data"),
)
def update_performance_metrics_accuracy_text(data):
    dff = pd.read_json(data, typ="series")
    tn, fp, fn, tp = np.array(dff.get("Confusion Matrix")).ravel()
    return (
        html.P(
            f"Your customized model has been tried out on a test sample of {fn+fp+fn+tp} tweets in {round(dff.get('Time'),4)} seconds. "
            f"It correctly classified {tn+tp} of records, while the remaining {fp+fn} were assigned to a wrong class. "
            f"This means that the accuracy is {round(dff.get('Accuracy'), 3)*100}%."
        ),
    )


# PERFORMANCE METRICS TEXT - PRECISION
@app.callback(
    Output("performance-metrics-precison-text", "children"),
    Input("intermediate-value", "data"),
)
def update_performance_metrics_precision_text(data):
    dff = pd.read_json(data, typ="series")
    tn, fp, fn, tp = np.array(dff.get("Confusion Matrix")).ravel()
    return html.P(
        f"The classifier has marked a total of {fp+tp} tweets as those that relate to natural disasters (class 1). Out of these"
        f", {tp} were actually in this group. This gives precision of {round(dff.get('Precision'), 3)*100}%"
    )


# PERFORMANCE METRICS TEXT - RECALL
@app.callback(
    Output("performance-metrics-recall-text", "children"),
    Input("intermediate-value", "data"),
)
def update_performance_metrics_recall_text(data):
    dff = pd.read_json(data, typ="series")
    tn, _, fn, _ = np.array(dff.get("Confusion Matrix")).ravel()
    return html.P(
        f"The sample contained a total of {tn+fn} true negatives. "
        f"Out of them, {tn} were correctly predicted by the model. Hence, the recall is {round(dff.get('Recall'), 3)*100}%. "
        f"Finally, the harmonic mean of precison and recall is {round(dff.get('F1 Score'), 3)*100}%."
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


# PROGRESS BAR


@ app.callback(
    [Output("progress", "value"), Output("progress", "label")],
    [Input("progress-interval", "n_intervals")],
)
def update_progress(n):
    # check progress of some background process, in this example we'll just
    # use n_intervals constrained to be in 0-100

    progress = min(n % 110, 100)
    # only add text after 5% progress to ensure text isn't squashed too much
    return progress, f"{progress} %" if progress >= 5 else ""


# TABS SETUP
tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Exploratory Data Analysis", tab_id="tab-1"),
        dbc.Tab(tab2_content, label="Classification", tab_id="tab-2"),
        dbc.Tab(
            "This tab's content is never seen",
            label="Compare runs",
            disabled=True,
            tab_id="tab-3",
        ),
        dbc.Tab(
            "This tab's content is never seen",
            label="Custom data upload",
            disabled=True,
            tab_id="tab-4",
        ),
        dbc.Tab(
            "This tab's content is never seen",
            label="Twitter API Calls",
            disabled=True,
            tab_id="tab-5",
        ),
        dbc.Tab(
            "This tab's content is never seen",
            label="Community labelling",
            disabled=True,
            tab_id="tab-6",
        ),
        dbc.Tab(tab7_content, label="About", tab_id="tab-7"),
    ],
    active_tab="tab-1",
)

# LAYOUT
app.layout = dbc.Container(
    [html.H1([html.I(className="bi bi-twitter me-2"), "NLP Disaster Tweets"]), tabs]
)


if __name__ == "__main__":
    app.run_server(debug=True)
