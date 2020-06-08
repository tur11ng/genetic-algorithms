import pandas as pd


def parse_ml100k():
    # Read data
    ds = pd.read_csv('ml-100k/u.data', sep='\t',
                     names=['user_id', 'movie_id', 'rating', 'unix_timestamp'], encoding='latin-1')

    # Dropping the columns that are not required
    ds.drop("unix_timestamp", inplace=True, axis=1)

    # Generate matrix
    ds = ds.pivot_table(index=['user_id', ], columns=['movie_id'],
                        values='rating').reset_index(drop=True)

    return ds.to_numpy()