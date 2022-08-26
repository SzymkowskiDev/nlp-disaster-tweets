# IMPORT LOCAL

# IMPORT EXTERNAL
from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# TAB 6: ABOUT ############################################################################################################
tab6_content = dbc.Card(
    dbc.CardBody(
        [
            html.Div([html.H1("Why bother?", className="card-text"),
                      html.P(
                          "In the times of wide ownership and use of smartphones, many emergency services take interest in programatic monitoring of social media."),
                      html.P(
                          "This is, so as to quickly alert first responders about an emmergency taking place."),
                      html.P(
                          "So, being able to tell whether a given tweet is conveying a message on a disaster is a matter of public safety."),
                      html.P([html.A(["This is a Kaggle problem, more info here."],
                             href="https://www.kaggle.com/competitions/nlp-getting-started/overview")]),
                      html.P([html.A(["Related GitHub repo can be accessed here."],
                             href="https://github.com/SzymkowskiDev/nlp-disaster-tweets")])
                      ],
                     style={"backgroundImage": "url(assets/frist_responders.jpeg", "width": "100%", "height": 600, "padding": 40}),
        ]
    ),
    className="mt-3",
)
