from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

tab1_content = dbc.Card(
    dbc.CardBody(
    [
        dbc.Row(
            [
                dbc.Col([
                    html.Div([
                        html.H3("Preprocessing", style={"fontSize": 20}),
                        dcc.Checklist(
                            ['Remove hashes', 'Remove duplicates', 'Translate emojis'], ['Remove hashes', 'Remove duplicates'],
                            labelStyle={'display': 'block'},
                            style={"height":200, "width":200, "overflow":"auto"},
                            inputStyle={"marginRight": "12px"})
                    ])

                ],
                width = 2),
                dbc.Col([
                    html.Div([
                        html.H3("Vectorization", style={"fontSize": 20}),
                        dcc.RadioItems(['Count', 'TF-IDF'], 'TF-IDF',
                        labelStyle={'display': 'block'},
                        style={"height":200, "width":200, "overflow":"auto"},
                        inputStyle={"marginRight": "12px"})
                    ])

                ],
                width = 2),
                dbc.Col([
                    html.Div([
                        html.H3("Model", style={"fontSize": 20}),
                        dcc.RadioItems(['SVC', 'Ridge', 'Logistic', 'SGD', 'Perceptron', 'PAA'], 'SVC',
                        labelStyle={'display': 'block'},
                        style={"height":200, "width":200, "overflow":"auto"},
                        inputStyle={"marginRight": "12px"}),
                    dbc.Button("Run", color="success", className="me-1")
                    ])

                ],
                width = 2),

                dbc.Col(
                    html.H2("OUTPUTS"),
                    width = 6
                ),
            ]
        )
    ]
    ),
    className="mt-3",
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)

tab5_content = dbc.Card(
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
        dbc.Tab(tab1_content, label="Classification"),
        dbc.Tab(tab2_content, label="Exploratory Data Analysis"),
        dbc.Tab("This tab's content is never seen", label="Custom data upload", disabled=True),
        dbc.Tab("This tab's content is never seen", label="Twitter API Calls", disabled=True),
        dbc.Tab("This tab's content is never seen", label="Community labelling", disabled=True),
        dbc.Tab(tab5_content, label="About"),
    ]
)

### LAYOUT
app.layout = dbc.Container([
    html.H1([html.I(className="bi bi-twitter me-2"), "NLP Disaster Tweets"]),
    tabs
])

if __name__ == '__main__':
    app.run_server(debug=True)
