# IMPORT LOCAL

# IMPORT EXTERNAL
from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# TAB 5: TWITTER BOT ANALYTICS #######################################################################################################
tab5_content = dbc.Card(
    dbc.CardBody(
        [
            html.H2("The Disaster Retweeter", className="card-text"),
            html.P("We have built a twitter BOT whose sole purpose is to apply our classification model to share tweets that it decideds are about a disaster."),
            html.P("Below are live analytics from its long-term operations.")
        ]
    ),
    className="mt-3", style={"height": 600}
)
