
# NLP with Disaster Tweets (group project)
Group solution to the Kaggle problem titled "Natural Language Processing with Disaster Tweets". The problem is to classify text data from 10,000 tweets into one of two groups: representing tweets about real natural disaster (1), tweets that are not about actual disaster (0).

* ⭐ Kaggle's top score:           0.86117
* ⭐ Our top prediction score:     0.84155

![banner](https://github.com/SzymkowskiDev/nlp-disaster-tweets/blob/master/assets/banner.PNG?raw=true)

## Contents
1. [🚀 How to run](#-How-to-run)
2. [👨‍💻 Contributing](#-Contributing)
3. [📂 Directory Structure](#-Directory-Structure)
4. [🔗 Related Projects](#-Related-Projects)
5. [🎓 Learning Materials](#-Learning-Materials)
6. [📅 Development Schedule](#-Development-Schedule)
7. [🆕 Changelog](#-Changelog)
8. [🤖 Stack](#-Stack)
9. [👓 Theory](#-Theory)
10. [📝 Examples](#-Examples)
11. [⚙ Configurations](#-Configurations)
12. [💡 Tips](#-Tips)
13. [🚧 Warnings](#-Warnings)
14. [🧰 Troubleshooting](#-Troubleshooting)
15. [📧 Contact](#-Contact)
16. [📄 License](#-License)

## 🚀 How to run
In the first iteration of the project, all there is to running the project is downloading a Jupyter notebook from directory "notebooks" and launching it with Jupyter.
Jupyter is available for download as a part of Anaconda suite from https://www.anaconda.com/.

When feeding a Jupyter notebook with data, use data provided in directory "train_split" [here](https://github.com/SzymkowskiDev/nlp-disaster-tweets/tree/master/data/train_split).

### Setup
<li>Create a virtual environment using <code> virtualenv venv </code>
<li>Activate the virtual environment by running <code> venv/bin/activate </code>
<li>On Windows use <code> venv\Scripts\activate.bat </code>
<li>Install the dependencies using <code> pip install -r requirements.txt </code>

## 👨‍💻 Contributing
* [SzymkowskiDev](https://github.com/SzymkowskiDev)
* [OlegTkachenkoY](https://github.com/OlegTkachenkoY)
* [laplasjan](https://github.com/laplasjan)
* [PanNorek](https://github.com/PanNorek)
* [bswck](https://github.com/bswck)
* [Mefpef](https://github.com/Mefpef)

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
    │       └───best_performing.py
    ├───notebooks
    ├───submissions
    └───reports
        ├───EDA.ipynb/.doc
        ├───Preprocessor_comparison.ipynb/.doc
        └───Tests_of_pre_preprocessing.ipynb/.doc

## 🔗 Related Projects
* Kaggle problem: ["Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started/overview)

## 🎓 Learning Materials
❗ More resources are available on Team's google drive: discordnlp7@gmail.com, ask a team member for password ❗

❗ Also check [the repo's wiki](https://github.com/SzymkowskiDev/nlp-disaster-tweets/wiki) ❗

* A wonderful book on the basics of NLP ["Speech and Language Processing"](https://web.stanford.edu/~jurafsky/slp3/)
* Kaggle's introductory tutorial to NLP [NLP Getting Started Tutorial](https://www.kaggle.com/code/philculliton/nlp-getting-started-tutorial/notebook)
* How does CountVectorizer work? [towardsdatascience.com article](https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c)
* [Data Mining and Business Analytics with R - Johannes Ledolter](https://mail.sitoba.it.maranatha.edu/Temu%20Pengetahuan%201516/Buku%20Referensi/DMBAR%20-%20Data%20Mining%20and%20Business%20Analytics%20with%20R%20-%20Johannes%20Ledolter.pdf)


## 📅 Development Schedule
**Version 1.0.0**

- [X] First Solution to the Kaggle's problem as a Jupyter notebook

**Version 1.1.0**

- [ ] Improved production model (Machine Learning)
    - [X] Selecting current best performing model
    - [ ] Exploratory Data Analysis
    - [ ] Comparison of preprocessors (vectorizers)
    - [ ] Testing influence of data pre preprocessing methods
    - [ ] Assembling a better model

**Version 1.2.0**
- [ ] Deep learning model

**Version 2.0.0**

- [X] Deployment of a blank dashboard (and integrate Dash)
- [ ] Customized classification
    - [ ] Inputs (Parameters for classification) (Blocked by Maganzo & Asia)
    - [ ] Outputs (Data Visualisation)
        - [ ] Pefromance metrics visalisation (input data)
        - [ ] Word cloud visualisation (output data)
        - [ ] Map of locations (input data)

## 🆕 Changelog
log of major changes to subsequent versions of the project/prediction model

## 🤖 Stack
* Python
* pandas
* scikit-learn
* nltk

## 👓 Theory
[Theory has been moved to the repo's wiki](https://github.com/SzymkowskiDev/nlp-disaster-tweets/wiki)

## 📝 Examples
**Example 1. Title**

Description of the example.
```javascript
CODE GOES HERE
```

## ⚙ Configurations
Sth

## 💡 Tips
💭 **Tip 1**

Description of tip 1.

## 🚧 Warnings

⚠️ **Warning 1**

Description of warning 1.

## 🧰 Troubleshooting
🚩 **Error 1**

Solution to error 1.

``` SOLUTION CODE ```

## 📧 Contact
[![](https://img.shields.io/twitter/url?label=/kamil-szymkowski/&logo=linkedin&logoColor=%230077B5&style=social&url=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fkamil-szymkowski%2F)](https://www.linkedin.com/in/kamil-szymkowski/) [![](https://img.shields.io/twitter/url?label=@szymkowskidev&logo=medium&logoColor=%23292929&style=social&url=https%3A%2F%2Fmedium.com%2F%40szymkowskidev)](https://medium.com/@szymkowskidev) [![](https://img.shields.io/twitter/url?label=/SzymkowskiDev&logo=github&logoColor=%23292929&style=social&url=https%3A%2F%2Fgithub.com%2FSzymkowskiDev)](https://github.com/SzymkowskiDev)

[![](https://img.shields.io/twitter/url?label=/joanna-michalska/&logo=linkedin&logoColor=%230077B5&style=social&url=https%3A%2F%2Fwww.linkedin.com%2Fin%2FJoanna-Michalska%2F)](https://www.linkedin.com/in/joannamichalska17/) [![](https://img.shields.io/twitter/url?label=/laplasjan&logo=github&logoColor=%23292929&style=social&url=https%3A%2F%2Fgithub.com%2Flaplasjan)](https://github.com/laplasjan)

[![](https://img.shields.io/twitter/url?label=/rafal-nojek/&logo=linkedin&logoColor=%230077B5&style=social&url=https%3A%2F%2Fwww.linkedin.com%2in%2rafaln97%2F)](https://www.linkedin.com/in/rafaln97/) [![](https://img.shields.io/twitter/url?label=/PanNorek&logo=github&logoColor=%23292929&style=social&url=https%3A%2F%2Fgithub.com%2FPanNorek)](https://github.com/PanNorek)
## 📄 License
[MIT License](https://choosealicense.com/licenses/mit/) ©️ 2019-2020 [Kamil Szymkowski](https://github.com/SzymkowskiDev "Get in touch!")

[![](https://img.shields.io/badge/license-MIT-green?style=plastic)](https://choosealicense.com/licenses/mit/)





