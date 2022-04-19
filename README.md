# Disaster Response Pipeline Project
### Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [Files](#files)
4. [Instructions](#instructions)
5. [Screenshots](#screenshots)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview <a name="overview"></a>
This project created an app that can classify the messages sent from different disasters. By efficiently classifying the disaster into correct categories and sending the requests to the apropriate disaster relief agencies, people can received the help they need in the timely manner. A multi-label classification model was built based on natural language process(NLP) techniques and ETL, Machine Learning pipelines to categorized evert message received.   

## Installation <a name="installation"></a>
This repository is running with Python3 with libraries of numpy, pandas, re, nltk, sklearn, sqlalchemy, plotly, and flask.  

## Files:<a name="files"></a>
- Data
  - disaster_categories.csv
  - disaster_message.csv
  - process_data.py: use ETL pipeline to read in above two .csv files, clean and stores in a SQLite database named DisasterResponse.db

- Models
  - train_classifier.py: load the data from database and do natural language processing and machine learning pipelines to build a machine laerning model stored in classifier.pkl   

- Apps
  - run.py: a Flask app to allow user use generated ML model to predict the disaster using input messages and prediction display visualization plots

## Instructions: <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots: <a name="screenshots"></a>
- Prediction Page
![Alt text](https://github.com/SoniaFYLin/disaster_response_pipeline_project/blob/master/prediction.jpg)

- Data Visualization 
![Alt text](https://github.com/SoniaFYLin/disaster_response_pipeline_project/blob/master/Fig1%262.jpg)
![Alt text](https://github.com/SoniaFYLin/disaster_response_pipeline_project/blob/master/Fig3%264.jpg)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

- Thanks for Udacity for providing data and template in Data Science Nanodegree program  
