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
from models.production.generate_perf_report import generate_perf_report
from models.production.vectorize_data import vectorize_data
from models.production.preprocess_data import preprocess_data
import plotly.figure_factory as ff
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from dash import Dash, html, Output, Input
import plotly.express as px
import plotly.graph_objects as go


# IMPORT DUMMY DATA FOR BAR CHART
dummy_class = pd.read_csv("data\class_chart.csv")

app = Dash(external_stylesheets=[dbc.themes.VAPOR, dbc.icons.BOOTSTRAP])

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

            # html.H2("Text"),
            # html.P("Description of variable 'text'"),
            # html.P("WORDCLOUD"),
            # html.H2("Dataset Balance"),
            # html.P("Is the dataset balanced?"),
            # html.P("BAR CHART"),
        ]
    ),
    className="mt-3",
)

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
                                    html.H3("Data Cleaning", style={"fontSize": 20}),
                                    dbc.Checklist(
                                        options=[
                                            {"label": "Remove hashes", "value": 1},
                                            {
                                                "label": "Remove HTML special entities",
                                                "value": 2,
                                            },
                                            {"label": "Remove tickers", "value": 3},
                                            {"label": "Remove hyperlinks", "value": 4},
                                            {"label": "Remove whitespaces", "value": 5},
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
                                            {"label": "Remove Mojibake", "value": 10},
                                            {
                                                "label": "Tokenize & Lemmatize",
                                                "value": 11,
                                            },
                                            {"label": "Leave only nouns", "value": 12},
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
                                    html.H3("Vectorization", style={"fontSize": 20}),
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
                                            {"label": "Logistic", "value": "Logistic"},
                                            {"label": "Naive Bayes", "value": "Bayes"},
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
                            html.H3("Fig 3. ROC & AUC", style={"fontSize": 20}),
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


# LOCATION MAP
@app.callback(
    Output("map_from_pgo", "figure"), Input("location-radio-items", "value"))
def update_location_map(value):

    df2 = pd.read_csv("data/totals.csv")
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
        # title_text='2014 Global GDP',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            # projection_type='equirectangular',
            projection_type="orthographic",
            bgcolor='rgba(0,0,0,0)',
            lakecolor="#17082D",
            showocean=True,
            oceancolor="#17082D"
            # showrivers =True,
            # rivercolor = "red",
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


@app.callback(
    Output("performance-metrics-accuracy-text", "children"),
    Input("intermediate-value", "data"),
)
def update(data):
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
        x=fpr, y=tpr, labels=dict(x="False Positive Rate", y="True Positive Rate")
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


# TAB 7: ABOUT ######################################################################################################################
tab6_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)

# TABS SETUP ########################################################################################################################

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
        dbc.Tab(tab6_content, label="About", tab_id="tab-7"),
    ],
    active_tab="tab-1",
)

# LAYOUT ##############################################################################################################################
app.layout = dbc.Container(
    [html.H1([html.I(className="bi bi-twitter me-2"), "NLP Disaster Tweets"]), tabs]
)


if __name__ == "__main__":
    app.run_server(debug=True)
