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
from dash import Dash, html


app = Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

### TAB 1: EXPLORATORY DATA ANALYSIS
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.H2("Introduction"),
            html.P(
                "We have at our disposal multivariate data consisting of 3 potential explonatory variables (keyword, location, text) and response variable (target). Variables 'keyword', 'location' and 'text' are nominal. The outcome variable 'target' is of binary type."
            ),
            html.P("In total we have a sample of 10,000."),
            html.H2("Data Quality Issues"),
            html.P(
                "Missing values are found in the column 'keyword', where there were only 61 missing records (out of 10,000) and in the column 'location', where a total of 2533 values were missing (over 25%)."
            ),
            html.H2("Keyword"),
            html.P("Description of variable 'keyword'."),
            html.P("WORDCLOUD"),
            html.H2("Location"),
            html.P("In the variable 'location' there are 3341 distinct values."),
            html.P("INTERACTIVE MAP"),
            html.H2("Text"),
            html.P("Description of variable 'text'"),
            html.P("WORDCLOUD"),
            html.H2("Dataset Balance"),
            html.P("Is the dataset balanced?"),
            html.P("BAR CHART"),
        ]
    ),
    className="mt-3",
)


# TAB 2: CLASSIFICATION
tab2_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("Preprocessing", style={"fontSize": 20}),
                                    dcc.Checklist(
                                        [
                                            "Remove hashes",
                                            "Remove duplicates",
                                            "Translate emojis",
                                            # TODO: add more preprocessing ideas
                                        ],
                                        value=["Remove hashes", "Remove duplicates"],
                                        labelStyle={"display": "block"},
                                        style={
                                            "height": 200,
                                            "width": 200,
                                            "overflow": "auto",
                                        },
                                        inputStyle={"marginRight": "12px"},
                                        id="preprocessing-checklist",
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
                                    dbc.Button(
                                        "Run", color="success", className="me-1"
                                    ),
                                ]
                            )
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.H2("OUTPUTS"),
                            html.P(
                                "Your customized classification correctly predicted 732 responses out of 1002, which ammounts to the accuracy rate of 0.81. Other metrics are shown below."
                            ),
                            html.H3(
                                "Fig 1. Performance Metrics", style={"fontSize": 20}
                            ),
                            html.Div(id="output-datatable"),
                            dcc.Store(id="intermediate-value"),
                            html.H3("Fig 2. Confusion Matrix", style={"fontSize": 20}),
                            dcc.Graph(id="confusion-matrix-graph"),
                            html.H3("Fig 3. ROC & AUC", style={"fontSize": 20}),
                            dcc.Graph(id="roc-graph"),
                            html.Div(id="roc-info"),
                        ],
                        width=6,
                    ),
                ]
            )
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
                                [html.Th("Accuracy  "), html.Th(dff.get("Accuracy"))]
                            ),
                            html.Tr(
                                [html.Th("Precision  "), html.Th(dff.get("Precision"))]
                            ),
                            html.Tr([html.Th("Recall  "), html.Th(dff.get("Recall"))]),
                            html.Tr(
                                [html.Th("F1 Score  "), html.Th(dff.get("F1 Score"))]
                            ),
                        ]
                    )
                ],
                style=dict(
                    textAlign="left",
                    padding="5px",
                    border="1px solid grey",
                    backgroundColor="#1E1E1E",
                    fontWeight="bold",
                    color="white",
                ),
                # style_header=dict(
                #     backgroundColor="#1E1E1E", fontWeight="bold", color="white"
                # ),
                # style_data=dict(backgroundColor="#323232", color="white"),
            )
        ]
    )
    # TODO: complete this function, e.g. add 'readable' layout
    pass


@app.callback(
    Output("confusion-matrix-graph", "figure"), Input("intermediate-value", "data")
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
        z,
        x=x,
        y=y,
        annotation_text=z_text,
        colorscale="greys",
        font_colors=["red", "red"],
    )

    # add custom xaxis title
    conf_matrix.add_annotation(
        dict(
            font=dict(color="white", size=18),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom yaxis title
    conf_matrix.add_annotation(
        dict(
            font=dict(color="white", size=18),
            x=-0.15,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    # adjust margins to make room for yaxis title
    conf_matrix.update_layout(
        margin=dict(t=0, l=50),
        height=500,
        width=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Courier New, monospace", size=22, color="white"),
    )
    # add colorbar
    conf_matrix["data"][0]["showscale"] = True
    # TODO: complete this function
    return conf_matrix


# TAB 6: ABOUT
tab6_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)


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
        height=500,
        width=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Courier New, monospace", size=18, color="white"),
    )
    return fig


@app.callback(Output("roc-info", "children"), Input("intermediate-value", "data"))
def update_roc_info(data):
    dff = pd.read_json(data, typ="series")
    roc_auc_score = dff.get("Roc_auc_score")
    return (
        html.P(
            f"AUC = {roc_auc_score:.4f}",
            style={
                "fontFamily": "Courier New, monospace",
                "fontSize": 20,
                "color": "red",
            },
            id="roc-paragraph",
        ),
    )


tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Exploratory Data Analysis", tab_id="tab-1"),
        dbc.Tab(tab2_content, label="Classification", tab_id="tab-2"),
        dbc.Tab(
            "This tab's content is never seen",
            label="Custom data upload",
            disabled=True,
            tab_id="tab-3",
        ),
        dbc.Tab(
            "This tab's content is never seen",
            label="Twitter API Calls",
            disabled=True,
            tab_id="tab-4",
        ),
        dbc.Tab(
            "This tab's content is never seen",
            label="Community labelling",
            disabled=True,
            tab_id="tab-5",
        ),
        dbc.Tab(tab6_content, label="About", tab_id="tab-6"),
    ],
    active_tab="tab-2",
)

### LAYOUT
app.layout = dbc.Container(
    [html.H1([html.I(className="bi bi-twitter me-2"), "NLP Disaster Tweets"]), tabs]
)


if __name__ == "__main__":
    app.run_server(debug=True)
