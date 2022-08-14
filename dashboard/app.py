from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

### TAB 1: EXPLORATORY DATA ANALYSIS
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.H2("Introduction"),
            html.P("We have at our disposal multivariate data consisting of 3 potential explonatory variables (keyword, location, text) and response variable (target). Variables 'keyword', 'location' and 'text' are nominal. The outcome variable 'target' is of binary type."),
            html.P("In total we have a sample of 10,000."),

            html.H2("Data Quality Issues"),
            html.P("Missing values are found in the column 'keyword', where there were only 61 missing records (out of 10,000) and in the column 'location', where a total of 2533 values were missing (over 25%)."),
            
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
                    [html.H2("OUTPUTS"),
                    html.P("PARAMETRIZED TEXT SUMMARIZING THE OUTCOME OF RUN"),
                    html.P("TABLE OF PERFROMANCE METRICS"),
                    html.P("CONFUSION MATRIX"),
                    html.P("ROC & AUC")                    
                    ],
                    width = 6
                ),
            ]
        )
    ]
    ),
    className="mt-3",
)

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
        dbc.Tab("This tab's content is never seen", label="Custom data upload", disabled=True, tab_id="tab-3"),
        dbc.Tab("This tab's content is never seen", label="Twitter API Calls", disabled=True, tab_id="tab-4"),
        dbc.Tab("This tab's content is never seen", label="Community labelling", disabled=True, tab_id="tab-5"),
        dbc.Tab(tab6_content, label="About", tab_id="tab-6")
    ],
    active_tab="tab-1"
)

### LAYOUT
app.layout = dbc.Container([
    html.H1([html.I(className="bi bi-twitter me-2"), "NLP Disaster Tweets"]),
    tabs
])

if __name__ == '__main__':
    app.run_server(debug=True)
