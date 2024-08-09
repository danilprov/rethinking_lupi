import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

def load_nasa_data():
    df = pd.read_csv('data/neo.csv')
    df = df.drop(['id', 'name', 'est_diameter_min', 'orbiting_body', 'sentry_object'], axis=1)
    df = df.rename(columns={"hazardous": "target"})
    x_features = ['est_diameter_max', 'relative_velocity']
    z_features = ['miss_distance', 'absolute_magnitude']
    X = df.loc[:, x_features + z_features]
    y = df.loc[:, 'target']
    #features = categorize_columns(X)
    features = X
    features_train, features_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42,
                                                                      stratify=y)
    # x_features = ['est_diameter_max_quantile', 'relative_velocity_quantile']
    # z_features = ['miss_distance_quantile', 'absolute_magnitude_quantile']

    # X_train = pd.get_dummies(
    #     features_train.loc[:, x_features], columns=x_features
    # ).values.astype(np.float32)
    # Z_train = pd.get_dummies(
    #     features_train.loc[:, z_features], columns=z_features
    # ).values.astype(np.float32)
    X_train = features_train.loc[:, x_features]
    Z_train = features_train.loc[:, z_features]
    X_test = features_test.loc[:, x_features]
    Z_test = features_test.loc[:, z_features]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z_train)
    Z_test = scaler.transform(Z_test)
    Y_train = 1 - pd.get_dummies(
        y_train.loc[:], columns=['target']
    ).values.astype(np.float32)

    # X_test = pd.get_dummies(
    #     features_test.loc[:, x_features], columns=x_features
    # ).values.astype(np.float32)
    # Z_test = pd.get_dummies(
    #     features_test.loc[:, z_features], columns=z_features
    # ).values.astype(np.float32)
    Y_test = 1 - pd.get_dummies(
        y_test.loc[:], columns=['target']
    ).values.astype(np.float32)

    return X_train, X_test, Z_train, Z_test, Y_train, Y_test


def load_drink_data():
    df = pd.read_csv('data/smoking_driking_dataset_Ver01.csv')
    # Removing duplicates
    df = df.drop_duplicates(keep='first')
    df = df.rename(columns={"DRK_YN": "target"})
    x_features = ['gamma_GTP', 'HDL_chole', 'SGOT_ALT', 'SGOT_AST', 'triglyceride',
                  'LDL_chole', 'hemoglobin', 'DBP', 'waistline', 'BLDS'] # all cont features
    z_features = ['age', 'sex', 'weight', 'SMK_stat_type_cd', 'height'] # all cat features

    X = df.loc[:, x_features + z_features]
    y = df.loc[:, 'target']

    features = X
    features_train, features_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42,
                                                                      stratify=y)
    X_train = features_train.loc[:, x_features]
    Z_train = features_train.loc[:, z_features]
    X_test = features_test.loc[:, x_features]
    Z_test = features_test.loc[:, z_features]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Z_train = pd.get_dummies(
        features_train.loc[:, z_features], columns=z_features
    ).values.astype(np.float32)
    Z_test = pd.get_dummies(
        features_test.loc[:, z_features], columns=z_features
    ).values.astype(np.float32)
    Y_train = 1 - pd.get_dummies(
        y_train.loc[:], columns=['target']
    ).values.astype(np.float32)
    Y_test = 1 - pd.get_dummies(
        y_test.loc[:], columns=['target']
    ).values.astype(np.float32)

    return X_train, X_test, Z_train, Z_test, Y_train, Y_test


def load_tinyimagenet():
    # df_train = pd.read_csv('data/tinyimagenet/train_tiny_imagenet_200.csv')
    # df_test = pd.read_csv('data/tinyimagenet/train_tiny_imagenet_200.csv')
    # classes_10 = ['n02795169', 'n02769748', 'n07920052', 'n02917067', 'n01629819',
    #               'n02058221', 'n02793495', 'n04251144', 'n02814533', 'n02837789']
    # df_train = df_train[df_train.label.isin(classes_10)]
    # df_test = df_test[df_test.label.isin(classes_10)]
    df_train = pd.read_csv('data/tinyimagenet/train_tiny_imagenet_200.csv')
    df_test = pd.read_csv('data/tinyimagenet/val_tiny_imagenet_200.csv')
    df_train = df_train.rename(columns={"label": "target"})
    df_test = df_test.rename(columns={"label": "target"})
    x_features = [col for col in df_train.columns if col.startswith('_resnet')]
    z_features = ['x1', 'y1', 'x2', 'y2']

    X_train = df_train.loc[:, x_features].values
    X_test = df_test.loc[:, x_features].values
    Z_train = df_train.loc[:, z_features].values / 63
    Z_test = df_test.loc[:, z_features].values / 63

    le = LabelEncoder()
    Y_train = le.fit_transform(df_train.loc[:, 'target'])
    Y_test = le.transform(df_test.loc[:, 'target'])

    Y_train = pd.get_dummies(
        Y_train
    ).values.astype(np.float32)
    Y_test = pd.get_dummies(
        Y_test
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

    X_train, X_test, Z_train, Z_test, Y_train, Y_test = load_nasa_data()

    print(X_train.shape, Z_train.shape, Y_train.shape)
    print(X_test.shape, Z_test.shape, Y_test.shape)

    X_train, X_test, Z_train, Z_test, Y_train, Y_test = load_drink_data()

    print(X_train.shape, Z_train.shape, Y_train.shape)
    print(X_test.shape, Z_test.shape, Y_test.shape)

    X_train, X_test, Z_train, Z_test, Y_train, Y_test = load_tinyimagenet()

    print(X_train.shape, Z_train.shape, Y_train.shape)
    print(X_test.shape, Z_test.shape, Y_test.shape)
