# IMPORT LOCAL

# IMPORT EXTERNAL
from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# TAB 4: MAKE A PREDICTION #######################################################################################################
tab4_content = dbc.Card(
    dbc.CardBody(
        [
            html.H3("Classify your tweet"),
            html.P("Input the text of your tweet below, the app will use our best performing model to decide, whether to call for emergency services. "),
            # INPUTS
            dbc.Input(id="input-tweet-to-predict", type="text"),
            # OUTPUT PARAGRAPH
            dbc.Spinner([html.Div(id="output-outcome-of-prediction")],
                        color="success", type="grow", spinner_style={"width": "8rem", "height": "8rem"}),
        ]
    ),
    className="mt-3", style={"height": 600}
)
