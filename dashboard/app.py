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

# # map with plolty.express
# df = px.data.gapminder().query("year==2007")
# map_from_px = px.choropleth(df, locations="iso_alpha",
#                             color="lifeExp",  # lifeExp is a column of gapminder
#                             hover_name="country",  # column to add to hover information
#                             color_continuous_scale=px.colors.sequential.Plasma)


# map with plotly.graph_objects
df2 = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

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
    #title_text='2014 Global GDP',
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


# # map with leaflet


# url = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png'
# attribution = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '

# let's try mapbox? no example with world choropleth

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
            html.P("Duplicated rows with opposing target values"),
            html.P("Missing responses in column 'location'"),
            html.P("Fake responses in column 'location'"),
            html.P("Location appearing in multipe formats: country, city"),
            html.P("Missing values in column 'keyword"),
            html.P("Strange characters appering in 'keyword' and 'text'"),


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
            # dl.Map(dl.TileLayer(), style={
            #        'height': '500px'})

            # 'width': '100%' is okey
            # how do I set starting coordinate views?
            # How do I set starting zoom view?
            # What is better interactive map or interactive globe?
            # If interactive map: choropleth vs charts
            # If interactive map: which provider: leaflet, mapbox, plotly.express as px, import plotly.graph_objects as go
            # html.Div([
            #     dl.Map(dl.TileLayer(url=url, attribution=attribution))
            # ], style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block", "position": "relative"}),
            # dcc.Graph(figure=map_from_px),

            dcc.Graph(figure=map_from_pgo)






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
                    dcc.Checklist(
                        [
                            "Remove hashes",
                            "Remove duplicates",
                            "Translate emojis",
                            # TODO: add more preprocessing ideas
                        ],
                        value=["Remove hashes",
                               "Remove duplicates"],
                        labelStyle={"display": "block"},
                        style={
                            "height": 200,
                            "width": 200,
                            "overflow": "auto",
                        },
                        inputStyle={"marginRight": "12px"},
                        id="preprocessing-checklist",
                    )
                ])], width=4),
                dbc.Col([html.Div([
                    html.H3("Vectorization", style={"fontSize": 20}),
                    dcc.RadioItems(
                        ["Count", "TF-IDF"],
                        value="TF-IDF",
                        labelStyle={"display": "block"},
                        style={
                            "height": 200,
                            "width": 200,
                            "overflow": "auto",
                        },
                        inputStyle={"marginRight": "12px"},
                        id="vectorization-radio-items",
                    )
                ])], width=4),
                dbc.Col([html.Div([
                    html.H3("Model", style={"fontSize": 20}),
                    dcc.RadioItems(
                        [
                            "SVC",
                            "Logistic",
                            # TODO: add other classifiers
                        ],
                        value="SVC",
                        labelStyle={"display": "block"},
                        style={
                            "height": 200,
                            "width": 200,
                            "overflow": "auto",
                        },
                        inputStyle={"marginRight": "12px"},
                        id="model-radio-items",
                    ),
                    dbc.Button("Run", color="success", className="me-1")
                ])], width=4)
            ]),
            dbc.Row([
                html.H2("OUTPUTS"),
                dbc.Col([html.P(
                    "Your customized classification correctly predicted 732 responses out of 1002, which amounts to the accuracy rate of 0.81. Other metrics are shown below."),
                    html.H3("Fig 1. Performance Metrics",
                            style={"fontSize": 20}),
                    html.Div([
                        html.Div(id="output-datatable"),
                        dcc.Store(id="intermediate-value"),
                        html.P('Run generated on 2022-07-29 00:17:16')
                    ])], width=4),
                dbc.Col([html.H3("Fig 2. Confusion Matrix", style={"fontSize": 20}),
                         dcc.Graph(id="confusion-matrix-graph")], width=4),
                dbc.Col([html.H3("Fig 3. ROC & AUC", style={"fontSize": 20}),
                         dcc.Graph(id="roc-graph")], width=4)
            ]),
        ]
    ),
    className="mt-3",
)


@app.callback(
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
                                [html.Th("Accuracy  "), html.Th(
                                    dff.get("Accuracy"))]
                            ),
                            html.Tr(
                                [html.Th("Precision  "), html.Th(
                                    dff.get("Precision"))]
                            ),
                            html.Tr(
                                [html.Th("Recall  "), html.Th(dff.get("Recall"))]),
                            html.Tr(
                                [html.Th("F1 Score  "), html.Th(
                                    dff.get("F1 Score"))]
                            ),
                        ]
                    )
                ],
                style=dict(
                    textAlign="left",
                    padding="5px",
                    border="1px solid grey",
                    backgroundColor="#3F3B96",
                    fontWeight="bold",
                    color="white",
                )
            )
        ]
    )
    # TODO: complete this function, e.g. add 'readable' layout
    pass


@app.callback(
    Output("confusion-matrix-graph",
           "figure"), Input("intermediate-value", "data")
)
def update_confusion_matrix(data):
    dff = pd.read_json(data, typ="series")
    z = dff.get("Confusion Matrix")
    x = ["0", "1"]
    y = ["1", "0"]

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure
    conf_matrix = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=z_text, colorscale='blues', font_colors=["red", "red"])

    # add custom xaxis title
    conf_matrix.add_annotation(dict(font=dict(color="white", size=18),
                                    x=0.5,
                                    y=-0.15,
                                    showarrow=False,
                                    text="Predicted value",
                                    xref="paper",
                                    yref="paper"))

    # add custom yaxis title
    conf_matrix.add_annotation(dict(font=dict(color="white", size=18),
                                    x=-0.15,
                                    y=0.5,
                                    showarrow=False,
                                    text="Real value",
                                    textangle=-90,
                                    xref="paper",
                                    yref="paper"))

    # adjust margins to make room for yaxis title
    conf_matrix.update_layout(
        margin=dict(t=0, l=50),
        height=350,
        width=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Courier New, monospace",
            size=22,
            color="white"
        ))

    # add colorbar
    conf_matrix['data'][0]['showscale'] = True

    # TODO: complete this function
    return conf_matrix


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
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="white"
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
