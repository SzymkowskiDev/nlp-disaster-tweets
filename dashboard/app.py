from turtle import width
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, Input, Output, dash_table
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification


app = Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

#### OUTPUTS ####
## METRICS TABLE
metrics_df = pd.DataFrame(
    {
        "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
        "Value": [0.875, 0.679208, 0.764771, 0.815236],
    }
)

## HEATMAP
z = [[156, 310], [602, 74]]
x = ["0", "1"]
y = ["1", "0"]

# change each element of z to type string for annotations
z_text = [[str(y) for y in x] for x in z]

# set up figure
conf_matrix = ff.create_annotated_heatmap(
    z, x=x, y=y, annotation_text=z_text, colorscale="greys", font_colors=["red", "red"]
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

## ROC & AUC
X, y = make_classification(n_samples=500, random_state=0)

model = LogisticRegression()
model.fit(X, y)
y_score = model.predict_proba(X)[:, 1]

fpr, tpr, thresholds = roc_curve(y, y_score)

fig = px.area(
    x=fpr, y=tpr, labels=dict(x="False Positive Rate", y="True Positive Rate")
)
fig.add_shape(type="line", line=dict(dash="dash", color="red"), x0=0, x1=1, y0=0, y1=1)

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
                                        id="preprocessing_checklist",
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
                                        id="vectorization_radio_items",
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
                                        id="model_radio_items",
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
                            html.Div(
                                [
                                    dash_table.DataTable(
                                        id="table",
                                        columns=[
                                            {"name": i, "id": i}
                                            for i in metrics_df.columns
                                        ],
                                        data=metrics_df.to_dict("records"),
                                        style_cell=dict(
                                            textAlign="left",
                                            padding="5px",
                                            border="1px solid grey",
                                        ),
                                        style_header=dict(
                                            backgroundColor="#1E1E1E",
                                            fontWeight="bold",
                                            color="white",
                                        ),
                                        style_data=dict(
                                            backgroundColor="#323232", color="white"
                                        ),
                                    ),
                                    html.P("Run generated on 2022-07-29 00:17:16"),
                                ]
                            ),
                            html.H1(id="intermediate-value"),
                            html.H3("Fig 2. Confusion Matrix", style={"fontSize": 20}),
                            dcc.Graph(figure=conf_matrix),
                            html.H3("Fig 3. ROC & AUC", style={"fontSize": 20}),
                            dcc.Graph(figure=fig),
                            html.P(
                                f"AUC = {auc(fpr, tpr):.4f}",
                                style={
                                    "fontFamily": "Courier New, monospace",
                                    "fontSize": 20,
                                    "color": "red",
                                },
                            ),
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
        Input("preprocessing_checklist", "value"),
        Input("vectorization_radio_items", "value"),
        Input("model_radio_items", "value"),
    ],
)
def our_function(value_1, value_2, value_3):
    return value_1


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
