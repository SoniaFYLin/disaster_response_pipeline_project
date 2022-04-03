import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # disaster distribution of message from genre == 'direct', 'social' and 'news'
    df_direct = df[df['genre'] == 'direct']
    df_social = df[df['genre'] == 'social']
    df_news = df[df['genre'] == 'news']
    df_news_labels = df_news[df_news.columns[4:]]
    df_news_labels.sum(axis=0)
    df_social_labels = df_social[df_social.columns[4:]]
    df_social_labels.sum(axis=0)
    df_direct_labels = df_direct[df_direct.columns[4:]]
    df_direct_sort = df_direct_labels.sum(axis=0).sort_values(ascending=False)
    df_social_sort = df_social_labels.sum(axis=0).sort_values(ascending=False)
    df_news_sort = df_news_labels.sum(axis=0).sort_values(ascending=False)
    #my figure

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=df_social_labels.sum(axis=0).index,
                    y=df_social_labels.sum(axis=0).values,
                    name = 'Social'
                ),
                Bar(
                    x=df_news_labels.sum(axis=0).index,
                    y=df_news_labels.sum(axis=0).values,
                    name = 'News'
                ),
                Bar(
                    x=df_direct_labels.sum(axis=0).index,
                    y=df_direct_labels.sum(axis=0).values,
                    name = 'Direct'
                )
            ],

            'layout': {
                'title': 'Distribution of Disasters in different Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disasters"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_social_sort.index[:9],
                    y=df_social_sort.values[:9]
                )
            ],

            'layout': {
                'title': 'Top 10 Disasters reported from Genre=social',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disasters"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_news_sort.index[:9],
                    y=df_news_sort.values[:9]
                )
            ],

            'layout': {
                'title': 'Top 10 Disasters reported from Genre=news',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disasters"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_direct_sort.index[:9],
                    y=df_direct_sort.values[:9]
                )
            ],

            'layout': {
                'title': 'Top 10 Disasters reported from Genre=direct',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disasters"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()