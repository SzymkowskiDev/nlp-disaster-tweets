
# NLP with Disaster Tweets (group project)
Group solution to the Kaggle problem titled "Natural Language Processing with Disaster Tweets". The problem is to classify text data from 10,000 tweets into one of two groups: representing tweets about real natural disaster (1), tweets that are not about actual disaster (0).

[Dashboard implementing our solution is available here.](https://nlp-disaster-tweets.herokuapp.com/)

* ⭐ Kaggle's top score:           0.86117
* ⭐ Our top prediction score:     0.84155

![banner](https://github.com/SzymkowskiDev/nlp-disaster-tweets/blob/master/assets/banner.PNG?raw=true)

## Contents
1. [🔗 Related Projects](#-Related-Projects)
2. [👓 Theory](#-Theory)
3. [⚙️ Setup](#-Setup)
4. [🚀 How to run](#-How-to-run)
5. [👨‍💻 Contributing](#-Contributing)
6. [🏛️ Architecture](#-Architecture)
7. [📂 Directory Structure](#-Directory-Structure)
8. [🎓 Learning Materials](#-Learning-Materials)
9. [📅 Development Schedule](#-Development-Schedule)
10. [🆕 Changelog](#-Changelog)
11. [🤖 Stack](#-Stack)
12. [📝 Examples](#-Examples)
13. [⚙ Configurations](#-Configurations)
14. [💡 Tips](#-Tips)
15. [🚧 Warnings](#-Warnings)
16. [🧰 Troubleshooting](#-Troubleshooting)
17. [📧 Contact](#-Contact)
18. [📄 License](#-License)

## 🔗 Related Projects
* Kaggle problem: ["Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started/overview)
* NLP Disaster Tweets [online dashboard](https://nlp-disaster-tweets.herokuapp.com/)
* Twitter BOT built for this project [Disaster Retweeter](https://github.com/bswck/disaster-retweeter)

## 👓 Theory
[Theory has been moved to the repo's wiki](https://github.com/SzymkowskiDev/nlp-disaster-tweets/wiki)

## ⚙️ Setup
Take these steps before section "🚀 How to run"
<li>Create a virtual environment using <code> virtualenv venv </code>
<li>Activate the virtual environment by running <code> venv/bin/activate </code>
<li>On Windows use <code> venv\Scripts\activate.bat </code>
<li>Install the dependencies using <code> pip install -r requirements.txt </code>

## 🚀 How to run

    Follow the steps in "⚙️ Setup" section that describe how to install all the dependencies
   
### How to access the web app?
The dashboard is deployed at Heroku and is live at the address [https://nlp-disaster-tweets.herokuapp.com/](https://nlp-disaster-tweets.herokuapp.com/)

### How to run the web app ("NLP Disaster Tweets") locally?
1. Clone the repo to the destination of your choice `git clone https://github.com/SzymkowskiDev/nlp-disaster-tweets.git`
2. Open Python interpreter (e.g. Anaconda Prompt) and change the directory to the root of the project `nlp-disaster-tweets`
3. In the terminal run the command `python app.py`
4. The app will launch in your web browser at the address [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

### How to run the REST API development server ("Disaster Retweeter Web") locally?
1. Clone the repo to the destination of your choice `git clone https://github.com/bswck/disaster-retweeter`
2. In your Python interpreter (e.g. Anaconda Prompt) change the directory to the root of the project `disaster-retweeter`
3. In the terminal run the command 'uvicorn retweeter_web.app.main:app`
4. The app will lanuch in your web browser at the address [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### How to run a Jupyter notebook?
In the first iteration of the project, all there is to running the project is downloading a Jupyter notebook from directory "notebooks" and launching it with Jupyter.
Jupyter is available for download as a part of Anaconda suite from https://www.anaconda.com/.

When feeding a Jupyter notebook with data, use data provided in directory "train_split" [here](https://github.com/SzymkowskiDev/nlp-disaster-tweets/tree/master/data/train_split).

## 👨‍💻 Contributing
* [SzymkowskiDev](https://github.com/SzymkowskiDev)
* [OlegTkachenkoY](https://github.com/OlegTkachenkoY)
* [PanNorek](https://github.com/PanNorek)
* [bswck](https://github.com/bswck)

## 🏛️ Architecture
Description
![architecture](https://github.com/SzymkowskiDev/nlp-disaster-tweets/blob/master/assets/architecture.png?raw=true)
   
   
## 📂 Directory Structure
    ├───assets
    ├───dashboard
    ├───data
    │   └───original
    │   │   ├───test.csv
    │   │   └───train.csv
    │   └───train_split
    │       ├───python
    │       ├───test_new.csv
    │       └───train_new.csv
    ├───models
    │   └───production
    │       ├───best_performing.py
    │       └───validation.py
    ├───notebooks
    ├───submissions
    └───reports
        ├───EDA.ipynb/.doc
        ├───Preprocessor_comparison.ipynb/.doc
        ├───Tests_of_pre_preprocessing.ipynb/.doc
        └───Validator.py

## 🎓 Learning/Reference Materials
❗ More resources are available on Team's google drive: discordnlp7@gmail.com, ask a team member for password ❗

❗ Also check [the repo's wiki](https://github.com/SzymkowskiDev/nlp-disaster-tweets/wiki) ❗

* A wonderful book on the basics of NLP ["Speech and Language Processing"](https://web.stanford.edu/~jurafsky/slp3/)
* Kaggle's introductory tutorial to NLP [NLP Getting Started Tutorial](https://www.kaggle.com/code/philculliton/nlp-getting-started-tutorial/notebook)
* How does CountVectorizer work? [towardsdatascience.com article](https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c)
* [Data Mining and Business Analytics with R - Johannes Ledolter](https://mail.sitoba.it.maranatha.edu/Temu%20Pengetahuan%201516/Buku%20Referensi/DMBAR%20-%20Data%20Mining%20and%20Business%20Analytics%20with%20R%20-%20Johannes%20Ledolter.pdf)
* [Dash tutorial](https://dash.plotly.com/installation)
* [Plotly docs](https://plotly.com/python/)
* [Markdown in Dash](https://commonmark.org/help/)
* [Dash HTML Components Gallery & code snippets](https://dash.plotly.com/dash-html-components)
* [Dash Core Components Gallery & code snippets](https://dash.plotly.com/dash-core-components)
* [Bootstrap components for Dash](https://dash-bootstrap-components.opensource.faculty.ai/)
* [Font awesome icons](https://fontawesome.com/icons)
* [Colorscales for Plotly charts](https://plotly.com/python/colorscales/)
* [How to make a choropleth map or globe with plotly.graph_objects (go)](https://plotly.com/python/map-configuration/)
* [layout.geo reference](https://plotly.com/python/reference/layout/geo/#layout-geo-center)
* [bootstrap theme explorer](https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/explorer/)
* [ML Crach Course from google developlers](https://developers.google.com/machine-learning/crash-course/classification/accuracy)


## 📅 Development Schedule
**Version 1.0.0** MILESTONE: classification model

- [X] First Solution to the Kaggle's problem as a Jupyter notebook

**Version 1.1.0** MILESTONE: model + dashboard

- [X] Deployment of a blank dashboard (and integrate Dash)
- [X] Exploratory Data Analysis tab
    - [X] Introduction
    - [X] Data Quality Issues
    - [X] Keyword
    - [X] Location (+interactive globe data viz)
    - [X] Text (Word frequency (+Wordcloud))
    - [X] Target
- [X] Customized classification tab
- [ ] Best performing
- [X] Make a prediction
- [X] Twitter BOT Analytics (blank)
- [X] About page

**Version 1.2.0** MILESTONE: twitter bot

- [ ] Twitter bot
- [ ] Dashboard live analytics


## 🆕 Project duration
03/07/2022 - 15/09/2022 (74 days)

## 🤖 Stack
* Python
* pandas
* scikit-learn
* nltk
* dash

## 📝 Examples
**Example 1. Measuring performance metrics with `generate_perf_report()`**

To generate the model performance report, use `generate_perf_report()`.
It compares predictions based on provided training data (`X`) to expected results (`y`)
and gathers certain classification metrics, like precision, accuracy etc.:

```py
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from models.production.generate_perf_report import generate_perf_report

# Load the training data, prepare the TF-IDF vectorizer just for this demo
df = pd.read_csv(r"data\original\train.csv")
tfidf_vect = TfidfVectorizer(max_features=5000)

# Prepare training data and target values
X = tfidf_vect.fit_transform(df['text'])
y = df["target"].copy()

# Generate and print the report
report = generate_perf_report(
    X, y, name="demo report", description="tfidf vectorizer and no preprocessing"
)
print(report)
```

Output:
```
Date                               2022-07-29 00:17:16
Description      tfidf vectorizer and no preprocessing
Test Size                                         0.15
Precision                                        0.875
Recall                                        0.679208
F1 Score                                      0.764771
Accuracy                                      0.815236
Roc_auc_score                                 0.801142
Name: demo report, dtype: object
```

Name, description, test size and date format in the report can be optionally specified.

**Example 2. Performing vectorization of choice with `vectorize_data()`**
 
Function `vectorize_data()` takes two parameters:
    * data - 
    * method - available options are: "tfidf"
    
    
## 📧 Contact
[![](https://img.shields.io/twitter/url?label=/kamil-szymkowski/&logo=linkedin&logoColor=%230077B5&style=social&url=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fkamil-szymkowski%2F)](https://www.linkedin.com/in/kamil-szymkowski/) [![](https://img.shields.io/twitter/url?label=@szymkowskidev&logo=medium&logoColor=%23292929&style=social&url=https%3A%2F%2Fmedium.com%2F%40szymkowskidev)](https://medium.com/@szymkowskidev) [![](https://img.shields.io/twitter/url?label=/SzymkowskiDev&logo=github&logoColor=%23292929&style=social&url=https%3A%2F%2Fgithub.com%2FSzymkowskiDev)](https://github.com/SzymkowskiDev)

[![](https://img.shields.io/twitter/url?label=/rafal-nojek/&logo=linkedin&logoColor=%230077B5&style=social&url=https%3A%2F%2Fwww.linkedin.com%2in%2rafaln97%2F)](https://www.linkedin.com/in/rafaln97/) [![](https://img.shields.io/twitter/url?label=/PanNorek&logo=github&logoColor=%23292929&style=social&url=https%3A%2F%2Fgithub.com%2FPanNorek)](https://github.com/PanNorek)
## 📄 License
[MIT License](https://choosealicense.com/licenses/mit/) ©️ 2019-2020 [Kamil Szymkowski](https://github.com/SzymkowskiDev "Get in touch!")

[![](https://img.shields.io/badge/license-MIT-green?style=plastic)](https://choosealicense.com/licenses/mit/)
