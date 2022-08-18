from dash import Dash, html, Input, Output, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from models.production.generate_perf_report import generate_perf_report
from models.production.vectorize_data import vectorize_data
import plotly.figure_factory as ff
from turtle import width
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash import Dash, html, Output, Input
from dash_extensions.javascript import arrow_function
import plotly.express as px
import plotly.graph_objects as go

# map with plotly.graph_objects
df2 = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

# DATA TRANSFORMATIONS TO FEED MAP


map_from_pgo = go.Figure(data=go.Choropleth(
    locations=df2['CODE'],
    z=df2['GDP (BILLIONS)'],
    text=df2['COUNTRY'],
    colorscale='Plasma',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix='$',
    colorbar_title='Number of disasters',
))

map_from_pgo.update_layout(
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
    annotations=[dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        # text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
        #     CIA World Factbook</a>',
        showarrow=False
    )]
)

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
            html.Ul([html.Li("Duplicated rows with opposing target values"),
                     html.Li("Missing responses in column 'location'"),
                     html.Li("Fake responses in column 'location'"),
                     html.Li(
                         "Location appearing in multipe formats: country, city"),
                     html.Li("Missing values in column 'keyword'"),
                     html.Li("Strange characters appering in 'keyword' and 'text'")
                     ]),

            # html.H2("Keyword"),
            # html.P("Description of variable 'keyword'."),
            # html.P("WORDCLOUD"),

            # LOCATION #########################################################################
            html.H2("Location"),

            # What does the variable 'location' represent?
            html.P("The variable 'location' represents the responses that Twitter users gave about where they were from. Out of 10,000 users, 7467 submited any response at all giving the response rate of 74.7%. One would expect structured data in the form 'country-city', but that is not the case. Users submitted their location data with the help of a text input. So, apart from responses like 'France', 'Germany' or 'USA' the column contains values like 'milky way', 'Worldwide' or 'Your Sister's Bedroom'. When geographical name is given at all, it often isn't given in a standard format. For example, the variable contains values like 'Jaipur, India' (city, country), 'bangalore' (just city), 'Indonesia' (just country) and many other variations of country, city, province or other administrative divison name in different combinations and no one uniform order."),

            # How we will transform the data to make it useful for analysis
            html.P("Nevertheless, thanks to so some data crunching we were able to make use of the largest possible subset of responses that contained geohraphical names."),

            # How many non-null responses contained geographical names that we could link with specific countries?
            html.P(
                "We were able to obtain the total of X number of legitimate country responses."),

            html.P("Since the response rate in the variable 'keyword' (that contains names of disasters) was 99.4%, we have the types of a catastrophy for almost all countries. The map below represents the most disastered countries in total and by type* of a catastrophy."),
            html.P("*It is worth noting, that the number of distinct values in the column 'keyword' was almost X. This is not because there are so many types of disasters out there. One of the reasons, why that's the case is because sometimes the same type of disaster like for example 'fire' is given in different ways like 'explosion', 'fire responders' or 'flames'. Hence we combined values in the following groups: ()."),

            html.P(
                "Map 1. Total number of disasters by country and types of disasters"),
            dbc.Row([
                dbc.Col([
                    html.H3("Select disaster type",
                            style={"fontSize": 20}),
                    dbc.RadioItems(
                        options=[
                            {"label": "All types", "value": "all"},
                            {"label": "Fire", "value": "fire"},
                            {"label": "Explosion", "value": "explosion"},
                            {"label": "Transport", "value": "transport"},
                            {"label": "Terrorism", "value": "terrorism"},
                            {"label": "Construction", "value": "construction"},
                            {"label": "Wind", "value": "wind"},
                            {"label": "Flooding", "value": "flooding"},
                            {"label": "Hot weather", "value": "hot"},
                            {"label": "Tectonics", "value": "tectonics"},
                            {"label": "Famine", "value": "famine"},
                            {"label": "Errosion", "value": "errosion"},
                            {"label": "Lightening", "value": "lightening"},
                            {"label": "Mass murder", "value": "mass"},
                            {"label": "Nuclear", "value": "nuclear"},
                            {"label": "Industrial", "value": "industrial"},
                            {"label": "Disease", "value": "disease"},
                            {"label": "Riot", "value": "riot"},
                            {"label": "War", "value": "war"},
                            {"label": "Unidentified", "value": "Unidentified"},
                        ],
                        value="all",
                        id="location-radio-items",
                    )], width=2),
                dbc.Col([dcc.Graph(figure=map_from_pgo)], width=10),
            ]),




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
            dbc.Row([
                html.H2("INPUTS"),
                dbc.Col([html.Div([
                    html.H3("Data Cleaning", style={"fontSize": 20}),
                    dbc.Checklist(
                        options=[
                            {"label": "Remove hashes", "value": 1},
                            {"label": "Remove duplicate records", "value": 2},
                            {"label": "Lemmatize", "value": 3},
                            {"label": "Remove HTML special entities", "value": 4},
                            {"label": "Remove tickers", "value": 5},
                            {"label": "Remove hyperlinks", "value": 6},
                            {"label": "Remove whitespaces", "value": 7},
                            {"label": "Remove URL, RT, mention(@)",
                             "value": 8},
                            {"label": "Remove no BMP characters", "value": 9},
                            {"label": "Remove misspelled words", "value": 10},
                            {"label": "Remove emojis", "value": 11},
                            {"label": "Remove Mojibake", "value": 12},
                            {"label": "Lemmatize", "value": 13},
                            {"label": "Leave only nouns", "value": 14},
                            {"label": "Spell check", "value": 15}],
                        id="preprocessing-checklist"
                    )
                ])], width=3),
                dbc.Col([html.Div([
                    html.H3("Vectorization", style={"fontSize": 20}),
                    dbc.RadioItems(
                        options=[
                            {"label": "Count", "value": "Count"},
                            {"label": "TF-IDF", "value": "TF-IDF"},
                            {"label": "Bag of Words", "value": "BoW"},
                            {"label": "Word2Vec ", "value": "W2V"}
                        ],
                        value=1,
                        id="vectorization-radio-items",
                    )
                ])], width=2),
                dbc.Col([html.Div([
                    html.H3("Model", style={"fontSize": 20}),
                    dbc.RadioItems(
                        options=[
                            {"label": "SVC", "value": "SVC"},
                            {"label": "Logistic", "value": "Logistic"}
                        ],
                        value=1,
                        id="model-radio-items",
                    ),
                ])], width=3),
                dbc.Col([
                    html.Div(
                        [
                            dbc.Button("Run", color="success", outline=True,
                                       id="run", style={"borderRadius": "50%", "height": 110, "width": 110, "marginBottom": 20}),
                            dbc.Button("Reset", color="secondary", outline=True,
                                       id="reset", style={"borderRadius": "50%", "height": 110, "width": 110, "marginBottom": 20}),
                            dbc.Button("Save", color="primary", outline=True,
                                       id="save", style={"borderRadius": "50%", "height": 110, "width": 110, "marginBottom": 20})
                        ],
                        className="d-grid gap-2"
                    ),
                ],
                    width=3)
            ]),
            dbc.Row([
                html.H2("OUTPUTS", style={"marginTop": 25}),
                dbc.Col([
                    html.P("Your customized model has been tried out on a test sample of 1002 tweets. It correctly classified 732 of records, while the remaining 270 were assigned to a wrong class. This means that the accuracy is 81.45%."),
                    html.H3("Fig 1. Confusion Matrix data",
                            style={"fontSize": 20}),
                    dcc.Graph(id="class_barchart"),
                ], width=4),
                dbc.Col([
                        html.P("The classifier has marked a total of 632 tweets as those that relate to natural disasters (class 1). Out of these, 547 were actually in this group. This gives precision of 83.26%."),
                        html.H3("Fig 2. Performance Metrics",
                                style={"fontSize": 20}),
                        html.Div([
                            html.Div(id="output-datatable"),
                            dcc.Store(id="intermediate-value"),
                        ]),
                        html.P(
                            "The sample contained a total of 652 true positives. Out of them, 610 were correctly predicted by the model. Hence, the recall is 66.87%. Finally, the harmonic mean of precison and recall is 77.34%."),
                        ], width=4),
                # dbc.Col([html.H3("Fig 2. Confusion Matrix", style={"fontSize": 20}),
                #          dcc.Graph(id="confusion-matrix-graph")], width=4),
                dbc.Col([
                    html.P(
                        "Fig 3. shows the performance of the classification model at all classification thresholds."),
                    html.H3("Fig 3. ROC & AUC", style={"fontSize": 20}),
                    dcc.Graph(id="roc-graph")], width=4)
            ]),
        ]
    ),
    className="mt-3",
)

# BARCHART


@app.callback(
    Output("class_barchart", "figure"), Input("intermediate-value", "data"))
def update_bar_chart(data):
    df = dummy_class
    fig = px.bar(df, x="Disaster", y="Count",
                 color="Actual", barmode="group", text_auto=True, color_discrete_sequence=['#48EF7B', '#D85360', '#48EF7B', '#D85360'])
    fig.update_layout(
        margin=dict(t=0, l=50),
        height=350,
        # width="100%",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(
            size=14,
            color="#32FBE2"
        ))

    return fig

# INTERIM DATA


@ app.callback(
    Output("intermediate-value", "data"),
    [
        Input("preprocessing-checklist", "value"),
        Input("vectorization-radio-items", "value"),
        Input("model-radio-items", "value"),
    ],


)
def our_function(preprocessing_checklist, vectorization, model):
    df_train = pd.read_csv(r"data\original\train.csv")
    tfidf_vect = TfidfVectorizer(max_features=5000)
    X = tfidf_vect.fit_transform(df_train["text"])
    y = df_train["target"].copy()
    # vectorize_data(data, method)
    series = generate_perf_report(X, y, clf=LogisticRegression())
    # TODO: complete this function
    return series.to_json(date_format="iso")

# PERFORMANCE METRICS TABLE


@ app.callback(
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
                                [html.Th("Accuracy", style={"padding": "8px 8px 4px 8px"}),
                                 html.Th(
                                    round(dff.get("Accuracy"), 2), style={"width": 50})]
                            ),
                            html.Tr(
                                [html.Th("Precision", style={"padding": "8px 8px 4px 8px"}), html.Th(
                                    round(dff.get("Precision"), 2))]
                            ),
                            html.Tr(
                                [html.Th("Recall", style={"padding": "8px 8px 4px 8px"}), html.Th(round(dff.get("Recall"), 2))]),
                            html.Tr(
                                [html.Th("F1 Score", style={"padding": "8px 8px 4px 8px"}), html.Th(
                                    round(dff.get("F1 Score"), 2))]
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
                    width="85%"
                )
            )
        ]
    )
    # TODO: complete this function, e.g. add 'readable' layout
    pass

# # CONFUSION MATRIX GRAPH
# @ app.callback(
#     Output("confusion-matrix-graph",
#            "figure"), Input("intermediate-value", "data")
# )
# def update_confusion_matrix(data):
#     dff = pd.read_json(data, typ="series")
#     z = dff.get("Confusion Matrix")
#     x = ["0", "1"]
#     y = ["1", "0"]

#     # change each element of z to type string for annotations
#     z_text = [[str(y) for y in x] for x in z]

#     # set up figure
#     conf_matrix = ff.create_annotated_heatmap(
#         z, x=x, y=y, annotation_text=z_text, colorscale='plasma', font_colors=["black", "white"])

#     # add custom xaxis title
#     conf_matrix.add_annotation(dict(font=dict(color="white", size=18),
#                                     x=0.5,
#                                     y=-0.15,
#                                     showarrow=False,
#                                     text="Predicted value",
#                                     xref="paper",
#                                     yref="paper"))

#     # add custom yaxis title
#     conf_matrix.add_annotation(dict(font=dict(color="white", size=18),
#                                     x=-0.15,
#                                     y=0.5,
#                                     showarrow=False,
#                                     text="Real value",
#                                     textangle=-90,
#                                     xref="paper",
#                                     yref="paper"))

#     # adjust margins to make room for yaxis title
#     conf_matrix.update_layout(
#         margin=dict(t=0, l=50),
#         height=350,
#         width=350,
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font=dict(
#             family="Courier New, monospace",
#             size=22,
#             color="white"
#         ))

#     # add colorbar
#     conf_matrix['data'][0]['showscale'] = True

#     # TODO: complete this function
#     return conf_matrix

# ROC GRAPH


@ app.callback(Output("roc-graph", "figure"), Input("intermediate-value", "data"))
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
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            size=14,
            color="#32FBE2"
        ))

    fig.add_trace(go.Scatter(
        x=[0.7],
        y=[0.3],
        mode="text",
        text=[f'AUC = {auc(fpr, tpr):.4f}'],
        textposition="bottom center"
    ))

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
