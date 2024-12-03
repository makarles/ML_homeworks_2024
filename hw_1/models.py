import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso

import pickle


class CustomNumTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['mileage'] = X['mileage'].str.split().str[0].astype(float, errors='ignore')
        X['engine'] = X['engine'].str.split().str[0].astype(float, errors='ignore')
        X['max_power'] = X['max_power'].str.replace('bhp', 'NaN')
        X['max_power'] = X['max_power'].str.split().str[0].astype(float, errors='ignore')
        return X


class CustomCatTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X


def fit_and_save_pipeline(path='pipeline.pickle') -> None:

    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

    df_train.drop(columns=['torque', 'name'], inplace=True)
    df_test.drop(columns=['torque', 'name'], inplace=True)

    df_train = df_train[df_train.drop('selling_price', axis=1).duplicated() == False]
    df_train = df_train.reset_index(drop=True)

    assert df_train.shape[1] == 11

    X_train, y_train = df_train.drop('selling_price', axis=1), df_train['selling_price']
    X_test, y_test = df_test.drop('selling_price', axis=1), df_test['selling_price']

    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    num_preprocessor = Pipeline(steps=[
        ('custom_transformer', CustomNumTransformer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
        ('poly', PolynomialFeatures(degree=2))
    ])

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']  # name удалён
    cat_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    col_transformer = ColumnTransformer([
        ('num_preprocessor', num_preprocessor, num_cols),
        ('cat_preprocessor', cat_preprocessor, cat_cols)
    ])

    lasso_pipeline = Pipeline(steps=[
        ('preprocessor', col_transformer),
        ('classifier', Lasso(random_state=42))
    ])

    lasso_pipeline.fit(X_train, y_train)
    print(f'R2 = {lasso_pipeline.score(X_test, y_test):.2f}')

    with open(path, 'wb') as file:
        pickle.dump(lasso_pipeline, file)
        print(f'Пайплайн сохранён как "{path}"')


if __name__ == '__main__':
    from models import CustomNumTransformer, CustomCatTransformer
    fit_and_save_pipeline()
