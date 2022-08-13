from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 1!", className="card-text"),
            dbc.Button("Click here", color="success"),
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
app.layout = html.Div([
    html.H1([html.I(className="bi bi-twitter me-2"), "NLP Disaster Tweets"]),
    tabs
])

if __name__ == '__main__':
    app.run_server(debug=True)
