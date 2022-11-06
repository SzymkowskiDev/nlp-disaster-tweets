# IMPORT LOCAL

# IMPORT EXTERNAL
from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# TAB 3: BEST PERFORMING #######################################################################################################
tab3_content = dbc.Card(
    dbc.CardBody(
        [
            html.H1(
                "BEST PERFORMING MODEL - WHAT DID WE USE AT THE COMPETITION AND WHY WAS IT A TRANSFORMER?"
            ),
            html.P(
                [
                    "The term Transformer was coined in 2017 in the groundbreaking paper ",
                    html.A(
                        "Attention is all you need",
                        href="https://arxiv.org/abs/1706.03762",
                    ),
                    " in reference to a newly proposed language model architecture. In this tab we will discuss its architecture very briefly.",
                ]
            ),
            html.P(
                [
                    "BERT stands for Bidirectional Encoder Representation from Transformers. In a transformer flow if we stack a number of encoders then we get a BERT. It is easier to make BERT understand a language. BERT also has a variety of problems such as Question-Answering , Sentiment Analysis , Text summarization ,etc.",
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.H3("Steps to use a BERT model: "),
                    html.Li(" Pretraining BERT : To understand language"),
                    html.Li("Fine tune BERT : To help us in our specific task"),
                    html.Br(),
                    html.Br(),
                    html.H3("Pretraining BERT"),
                    html.Li("To make BERT learn what is language."),
                    html.Li(
                        "It has two part Masked Langauge Modelling(MLM) and Next Sentence Prediction(NSP)."
                    ),
                    html.Li("Both of these problems are trained simultaneously."),
                    html.Br(),
                    html.Br(),
                    html.H3("Fine tuning BERT"),
                    html.Li("It is a quiet fast process."),
                    html.Li(
                        "Only the output parameters are leant from scratch and whereas the rest of the parameters are slightly fine-tuned and not that mucchanged which in turn makes the process faster."
                    ),
                ]
            ),
            html.Br(),
            html.Br(),
            html.Div(
                [
                    html.Img(
                        src="https://i.stack.imgur.com/eAKQu.png",
                        style={"height": "60%", "width": "60%"},
                    ),
                    html.P("Fig. 1 BERT Architecture"),
                ],
                style={"textAlign": "center"},
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H2("RoBERTa model - A Robustly Optimized BERT Pretraining Approach"),
            html.P(
                [
                    "BERT ",
                    html.A("(Devlin et. al.)", href="https://arxiv.org/abs/1810.04805"),
                    " is a pioneering Language Model that is pretrained for a Denoising Autoencoding objective to produce state of the art results in many NLP tasks. However, there is still room for improvement in the original BERT model w.r.t its pretraining objectives, the data on which it is trained, the duration for which it is trained, etc. These issues were identified by Facebook AI Research (FAIR), and hence, they proposed an ‘optimized’ and ‘robust’ version of BERT. This model was used for our project.",
                ]
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H2("STEP 1: TOKENIZATION"),
            html.Br(),
            html.P(
                "This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf... By default, all punctuation is removed, turning the texts into space-separated sequences of words (words maybe include the ' character). These sequences are then split into lists of tokens. They will then be indexed or vectorized."
            ),
            html.Br(),
            html.Br(),
            html.Div(
                [
                    html.Br(),
                    html.Img(
                        src="https://miro.medium.com/max/1400/1*4oivwX7zhExRCNkE5M4Ffw.png",
                        style={"height": "50%", "width": "50%"},
                    ),
                    html.P("Fig. 2 Tokenization - example"),
                ],
                style={"textAlign": "center"},
            ),
            html.Br(),
            html.H2("STEP 2: TRAINING MODEL"),
            html.P(
                [
                    "Of course we used RoBERTa pretrained model from Hugging Face. All we did was train the unlocked layers of the model on our tokenized data. Finally we were able to get a competition result of 0.84033 (F1-score).",
                ]
            ),
            html.Div(
                [
                    html.Br(),
                    html.Img(
                        src="assets\Score.png",
                        style={"height": "50%", "width": "50%"},
                    ),
                    html.P("Fig. 3 Our competition score"),
                ],
                style={"textAlign": "center"},
            ),
            html.Br(),
            html.P(
                ["If you are curious and looking for the code and documentation, you can find it at this ",
                html.A(
                    "link.",
                    href="https://www.kaggle.com/code/rafanojek/dashboard-models-f1-score-83-6",
                )]
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
        ],
        style={"text-align": "left"},
    ),
    className="mt-3",
)
