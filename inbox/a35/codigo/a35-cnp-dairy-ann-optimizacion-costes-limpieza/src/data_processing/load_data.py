import pandas as pd

def load_split_data(config):
    splits_path = config['splits_path']

    X_train = pd.read_csv(splits_path + 'X_train.csv')
    X_test = pd.read_csv(splits_path + 'X_test.csv')
    X_val = pd.read_csv(splits_path + 'X_val.csv')


    y_train = pd.read_csv(splits_path + 'y_train.csv')
    y_test = pd.read_csv(splits_path + 'y_test.csv')
    y_val = pd.read_csv(splits_path + 'y_val.csv')

    return X_train, X_test, X_val, y_train, y_test, y_val