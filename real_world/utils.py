import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from src.datasets.obd_custom import OpenBanditDatasetPrivilegedInformation

FEATURES = ['age_range', 'gender', 'dow']
PROXIES = ['log_signal_0_cat', 'signal_3_binary']  # we are not using add-to-cart as proxies
TARGET = 'signal_2_binary'


def prepare_ijcai_data(action_col, k, path='data/IJCAI15/'):
    """
    choosing top 10 `cat_id` as actions results in ~4.8M observations with ~400k purchases
    choosing top 10 `item_id` as acitons results in ~137k observations with ~10k purchases
    choosing top 10 `brand_id` as actions results in ~1.2M observations with ~138k purchases

    :param path:
    :param action_col: 'cat_id', 'item_id', or 'brand_id'
    :param k: number of actions
    :return:
    """
    # load data
    user_info = pd.read_csv(path + "user_info_format1.csv")
    user_log = pd.read_csv(path + "user_log_format1.csv")

    # select k most popular actions and filter the logs
    top_k_actions = user_log.groupby(action_col)[action_col].count().sort_values().index[-k:]
    user_log_k = user_log[user_log[action_col].isin(top_k_actions)]

    # reformat data
    # Assumption: responses that happened with respect to a single action
    # and within the same day are treated as one session (just for simplicity)
    df = user_log_k.groupby(['user_id', 'time_stamp', action_col])['action_type'].value_counts()
    df = df.unstack(fill_value=0).rename(columns=int).add_prefix('signal_').reset_index()

    # encode action id's to 0...k-1
    df[action_col] = pd.factorize(df[action_col])[0]

    # prepare features
    # add day of week
    df['date'] = pd.to_datetime('2014' + df['time_stamp'].apply(str), format='%Y%m%d')
    df['dow'] = df['date'].dt.dayofweek
    # add gender and age
    df = df.join(user_info.set_index('user_id'), on='user_id')
    # 0.0 and null correspond to unknown gender
    df['age_range'] = df['age_range'].fillna(0.0)
    # 2.0 and null correspond to unknown gender
    df['gender'] = df['gender'].fillna(2.0)

    # prepare signals
    # remove rows with more than 300 clicks per session and apply logs(x+1) to # of clicks
    df = df[(df['signal_0'] < 300)]
    #df['signal_0_binary'] = (df['signal_0'] > 0) * 1
    df['log_signal_0'] = np.log(df['signal_0'] + 1)
    df['log_signal_0_cat'] = pd.qcut(df['log_signal_0'], 3, labels=range(3))
    # recast add-to-favorites as binary
    df['signal_3_binary'] = (df['signal_3'] > 0) * 1
    # recast purchases as binary
    df['signal_2_binary'] = (df['signal_2'] > 0) * 1

    final_df = df[['time_stamp', action_col, TARGET] + FEATURES + PROXIES]
    assert len(final_df[final_df.isnull().any(axis=1)]) == 0

    # add stratification on action and target in train test split
    # I could have stratified on proxies as well, but it might be restrictive for some options
    # I sort dataset by time_stamp for time-based splitting (instead of random)
    final_df = final_df.sort_values('time_stamp')
    train_df, test_df = train_test_split(final_df, test_size=0.3, random_state=42,
                                         stratify=final_df[[action_col, TARGET]])

    filename_train = f'train_{action_col}_{k}.pickle'
    with open(path + filename_train, 'wb') as handle:
        pickle.dump(train_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    filename_test = f'test_{action_col}_{k}.pickle'
    with open(path + filename_test, 'wb') as handle:
        pickle.dump(test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_ijcai_data(action_col, k, is_train, path='data/IJCAI15/'):
    """

    :param action_col:
    :param k:
    :param is_train:
    :param path:
    :return: X - numpy array of shape (n_sample, n_features)
    :return: A - numpy array of shape (n_sample, 1)
    :return: Z - numpy array of shape (n_sample, n_proxies)
    :return: y - numpy array of shape (n_sample, )
    """
    filename_train = f'train_{action_col}_{k}.pickle'
    filename_test = f'test_{action_col}_{k}.pickle'

    if not os.path.isfile(path + filename_train) or not os.path.isfile(path + filename_test):
        prepare_ijcai_data(action_col, k, path)

    if is_train:
        with open(path + filename_train, 'rb') as handle:
            df = pickle.load(handle)
    else:
        with open(path + filename_test, 'rb') as handle:
            df = pickle.load(handle)

    # X = df[FEATURES].to_numpy()
    # A = df[[action_col]].to_numpy()
    # Z = df[PROXIES].to_numpy()
    # y = df[TARGET].to_numpy()

    X = pd.get_dummies(
        df.loc[:, FEATURES], columns=FEATURES
    ).values.astype(np.float32)
    A = pd.get_dummies(
        df.loc[:, action_col], columns=[action_col]
    ).values.astype(np.float32)
    Z = pd.get_dummies(
        df.loc[:, PROXIES], columns=PROXIES
    ).values.astype(np.float32)
    y = 1 - pd.get_dummies(
        df.loc[:, TARGET], columns=[TARGET]
    ).values.astype(np.float32)

    return X, A, Z, y


def prepare_input_data(x, a, z, mode, is_train, mean_value=None):
    # during training we allow distillation approaches to use z
    # for predictions distillation approaches should not have access to z
    # `priv` mode is a cheating mode that has access to z always
    if is_train:
        if mode in ['regular', 'selfD']:
            input_data = np.hstack((x, a))
        elif mode in ['priv', 'fullD', 'mean', 'zero', 'one']:
            input_data = np.hstack((x, z, a))
        elif mode in ['genD']:
            input_data = np.hstack((z, a))
        else:
            ValueError(f"Mode {mode} is unknown.")
    else:
        if mode in ['regular', 'selfD', 'fullD', 'genD']:
            input_data = np.hstack((x, a))
        elif mode in ['priv']:
            input_data = np.hstack((x, z, a))
        elif mode in ['zero']:
            zero_input = np.zeros_like(z)
            input_data = np.hstack((x, zero_input, a))
        elif mode in ['one']:
            zero_input = np.ones_like(z)
            input_data = np.hstack((x, zero_input, a))
        elif mode in ['mean']:
            assert mean_value is not None
            mean_input = np.ones_like(z) * mean_value
            input_data = np.hstack((x, mean_input, a))
        else:
            ValueError(f"Mode {mode} is unknown.")

    return input_data
