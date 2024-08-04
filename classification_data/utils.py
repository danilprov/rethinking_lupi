import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def softmax(w, t=1.0):
    """
    softmax with max normalization
    """
    max_values, _ = torch.max(w, dim=1, keepdim=True)
    e = torch.exp((w - max_values) / t)
    return e / torch.sum(e, 1).unsqueeze(-1)

def load_health_data():
    df = pd.read_csv('data/heart_disease_health_indicators_BRFSS2015.csv')
    df = df.rename(columns={"HeartDiseaseorAttack": "target"})

    corr_matrix = df.corr()
    threshold = 0.1
    filtre = np.abs(corr_matrix["target"]) > threshold
    corr_features = corr_matrix.columns[filtre].tolist()

    y = df.target
    features = df[corr_features].drop(["target"], axis=1)
    features_train, features_test, y_train, y_test = train_test_split(features, y, test_size=0.3,
                                                                      random_state=42, stratify=y)

    x_features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'Diabetes', 'GenHlth']
    z_features = ['PhysHlth', 'DiffWalk', 'Age', 'Income']
    X_train = pd.get_dummies(
        features_train.loc[:, x_features], columns=x_features
    ).values.astype(np.float32)
    Z_train = pd.get_dummies(
        features_train.loc[:, z_features], columns=z_features
    ).values.astype(np.float32)
    Y_train = 1 - pd.get_dummies(
        y_train.loc[:], columns=['target']
    ).values.astype(np.float32)

    X_test = pd.get_dummies(
        features_test.loc[:, x_features], columns=x_features
    ).values.astype(np.float32)
    Z_test = pd.get_dummies(
        features_test.loc[:, z_features], columns=z_features
    ).values.astype(np.float32)
    Y_test = 1 - pd.get_dummies(
        y_test.loc[:], columns=['target']
    ).values.astype(np.float32)

    return X_train, X_test, Z_train, Z_test, Y_train, Y_test


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/heart_disease_health_indicators_BRFSS2015.csv')
    print(df.head())
    print(df.columns)
    print(df.shape)
    print(df['HeartDiseaseorAttack'].sum())
    df = df.rename(columns={"HeartDiseaseorAttack": "target"})

    corr_matrix = df.corr()
    threshold = 0.1
    filtre = np.abs(corr_matrix["target"]) > threshold
    corr_features = corr_matrix.columns[filtre].tolist()
    print(corr_features)

    y = df.target
    features = df[corr_features].drop(["target"], axis=1)
    print(features.nunique())
    print(sorted(df['PhysHlth'].unique()))

    features_train, features_test, target_train, target_test = train_test_split(features, y, test_size=0.3, random_state=42, stratify=y)
    print(target_train.shape)
    print(target_train[:10])
    print(target_train.unique())
    x_features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'Diabetes', 'GenHlth']
    z_features = ['PhysHlth', 'DiffWalk', 'Age', 'Income']
    X_train = pd.get_dummies(
        features_train.loc[:, x_features], columns=x_features
    ).values.astype(np.float32)
    Z_train = pd.get_dummies(
        features_train.loc[:, z_features], columns=z_features
    ).values.astype(np.float32)
    Y_train = 1 - pd.get_dummies(
        target_train.loc[:], columns=['target']
    ).values.astype(np.float32)

    X_test = pd.get_dummies(
        features_test.loc[:, x_features], columns=x_features
    ).values.astype(np.float32)
    Z_test = pd.get_dummies(
        features_test.loc[:, z_features], columns=z_features
    ).values.astype(np.float32)
    Y_test = 1 - pd.get_dummies(
        target_test.loc[:], columns=['target']
    ).values.astype(np.float32)

    print(X_train.shape, Z_train.shape, Y_train.shape)
    print(X_test.shape, Z_test.shape, Y_test.shape)

    X_train, X_test, Z_train, Z_test, Y_train, Y_test = load_health_data()

    print(X_train.shape, Z_train.shape, Y_train.shape)
    print(X_test.shape, Z_test.shape, Y_test.shape)
