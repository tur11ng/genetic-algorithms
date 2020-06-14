import numpy as np
import pandas as pd


def parse_ml100k(filename):
    # Read data
    df = pd.read_csv('ml-100k/' + filename, sep='\t',
                     names=['user_id', 'movie_id', 'rating', 'unix_timestamp'], encoding='latin-1')

    # Dropping the columns that are not required
    df.drop("unix_timestamp", inplace=True, axis=1)

    # Generate matrix
    df = df.pivot_table(index=['user_id', ], columns=['movie_id'],
                        values='rating').reset_index(drop=True)
    return df


def prepare_holdout():
    df = parse_ml100k('u.data')
    df_train = parse_ml100k('ua.base')
    df_test = parse_ml100k('ua.test')

    empty_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    df_test = empty_df.merge(right=df_test, how='right')
    df_train = empty_df.merge(right=df_train, how='right')

    return df_train, df_test
