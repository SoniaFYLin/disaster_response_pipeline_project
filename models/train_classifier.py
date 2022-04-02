import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import pickle

def load_data(database_filepath):
    '''
    summary: read in table from given database filepath and return X, y and category_names for labels
    param database_filepath:
    return: X, y, category_names
    '''
    print (database_filepath)
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('clean_data', engine)
    X = df['message']
    y = df[df.columns[4:]]
    category_names = list(df.columns[4:])

    return X, y, category_names


def tokenize(text):
    '''
    Summary: read in a text message and return clean tockens
    param: text: message for tokens
    return: tokens: clean tokens of the input message
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Split text into words using NLTK
    words = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words("english")]

    return tokens


def build_model():
    '''
    Summary build ML pipeline or CV model with different parameters for tunning
    return: model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Use grid search to find better parameters
    #parameters = {'clf__estimator__n_estimators': [50, 60, 70, 80]}
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    model = pipeline

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Summary: Evaluate model accuracy and report using test data
    param model: model build from pipeline
    param X_test: X features from test data
    param Y_test: labels from test data
    param category_names: categories from labels
    return: result: pa.dataframe['accuracy', 'precision', 'recall', 'f1-score']
    '''
    # predict labels for X_test
    Y_pred = model.predict(X_test)

    # summary results
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for i in range(len(category_names)):
        report = classification_report(Y_test.iloc[:, i].values, Y_pred[:, i], output_dict=True)
        precision = report.get('weighted avg').get('precision')
        precision_list.append(precision)
        recall = report.get('weighted avg').get('recall')
        recall_list.append(recall)
        f1_score = report.get('weighted avg').get('f1-score')
        f1_score_list.append(f1_score)
        accuracy = accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])
        accuracy_list.append(accuracy)
        #print('Accuracy of %25s: %.2f' % (category_names[i], accuracy))
    data = {'Accuracy': accuracy_list,
            'Precision': precision_list,
            'Recall': recall_list,
            'F1_score': f1_score_list
            }
    results = pd.DataFrame(data, index=category_names)
    print (results)

def save_model(model, model_filepath):
    '''
    Summary: export model as a pickle file
    :param model: model to save
    :param model_filepath: path of the output pickle file
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()