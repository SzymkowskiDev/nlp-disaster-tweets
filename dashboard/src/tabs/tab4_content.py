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
            dbc.InputGroup(
                [
                    dbc.Button("Predict",
                               id="input-make-a-prediction", n_clicks=0),
                    dbc.Input(id="input-tweet-to-predict"),
                ]
            ),
            html.P("This is likely not about a disaster, no need for an alert.", style={
                   "color": "#49EF7B"}),
            html.P("This is likely an emergency, call 112.",
                   style={"color": "#DA525E"})
        ]
    ),
    className="mt-3", style={"height": 600}
)
