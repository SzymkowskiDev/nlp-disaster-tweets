# IMPORT LOCAL

# IMPORT EXTERNAL
from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# TAB 3: BEST PERFORMING #######################################################################################################
tab3_content = dbc.Card(
    dbc.CardBody(
        [
            html.H2("BEST PERFORMING MODEL"),
            html.P("We have found that the best performing model is X, in combination with the following data preprocessing operations:")
        ], style={"height": 500}
    ),
    className="mt-3",
)
