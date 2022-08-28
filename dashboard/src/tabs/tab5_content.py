# IMPORT LOCAL

# IMPORT EXTERNAL
from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# TAB 5: TWITTER BOT ANALYTICS #######################################################################################################
tab5_content = dbc.Card(
    dbc.CardBody(
        [
            html.H2("The Disaster Retweeter", className="card-text"),
            html.P(["We have built a twitter BOT whose sole purpose is to apply our classification model to share tweets that it decideds are about a disaster. ", html.A(
                ["The BOT can be accessed here."], href="https://twitter.com/elonmusk")]),
            html.P("Below are live analytics from its long-term operations."),
            dbc.Row([
                dbc.Col([
                    html.P("Time of operations: 341 days",
                           style={"color": "#35F8DF"}),
                    html.P("Number of tweets classified: 148 526",
                           style={"color": "#35F8DF"}),
                    html.P("Number of tweets retweeted: 12 162",
                           style={"color": "#35F8DF"}),
                ], width=3, xs=12, sm=12, md=12, lg=3, xl=3,),
                dbc.Col([
                    # This input is hidden, it is here to make bot-timeseries work
                    dbc.RadioItems(options=[{"label": "All groups", "value": 1}, {
                                   "label": "Exclude 'UNIDENTIFIED'", "value": 2}, ], value=2, id="bot-timeseries-input", inline=True, style={'display': 'none'}),
                    dbc.Spinner([dcc.Graph(id="bot-timeseries")], color="success",
                                spinner_style={"width": "8rem", "height": "8rem"})
                ], width=9, xs=12, sm=12, md=12, lg=9, xl=9,),
            ]),
        ]
    ),
    className="mt-3", style={"height": 600}
)
