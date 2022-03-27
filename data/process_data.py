import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):


    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on="id", how="outer")
    return df


def clean_data(df):

    ## create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    ## Extract category columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just number 0 or 1
    df_categories = categories
    for column in df_categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
        df_categories[column] = df_categories[column].apply(lambda x: int(x[-1]))

    ## Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df_concat = pd.concat([df, df_categories], axis=1)

    ## remove duplicates
    # check number of duplicates
    print(df_concat.duplicated().sum())

    # drop duplicates
    df_concat_drop = df_concat.drop_duplicates()

    return df_concat_drop

def save_data(df, database_filename):
    engine = create_engine('sqlite:///disaster_response.db')
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()