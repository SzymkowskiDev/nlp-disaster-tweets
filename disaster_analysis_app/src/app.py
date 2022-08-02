from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os


df = pd.read_csv('https://raw.githubusercontent.com/Mefpef/nlp-disaster-tweets/master/data/train_split/test_new.csv')

app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server


my_title = dcc.Markdown(children='App that analyses disasters')
my_graph = dcc.Graph(figure={})
dropdown = dcc.Dropdown(options=df.columns.values[2:], clearable=False)

app.layout = dbc.Container([my_title, my_graph, dropdown])


@app.callback(
    Output(my_graph, 'figure'),
    Input(dropdown, 'value')
)
def display_graph(keyword):
    fig = px.bar(data_frame=df, x='target', y='keyword', color=keyword)

    return fig


if __name__ == '__main__':
    app.run_server(port=8053)
