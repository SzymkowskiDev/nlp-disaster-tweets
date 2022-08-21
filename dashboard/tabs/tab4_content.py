# IMPORT LOCAL

# IMPORT EXTERNAL
from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# TAB 4: CUSTOM DATA UPLOAD #######################################################################################################
tab4_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)