import pandas as pd
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataImporter:
    def __init__(self):
        self.zip_file_path = 'dsb-24-german-credit.zip'
        self.test_file_path = 'german_credit_test.csv'
        self.train_file_path = 'german_credit_train.csv'
        self.submission_file_path = 'german_credit_test_submission.csv'

    def load_data(self):
        try:
            with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
                with zip_ref.open(self.test_file_path) as csv_file:
                    df_test = pd.read_csv(csv_file)
                with zip_ref.open(self.train_file_path) as csv_file:
                    df_train = pd.read_csv(csv_file)
                with zip_ref.open(self.submission_file_path) as csv_file:
                    df_submission = pd.read_csv(csv_file)
            return df_test, df_train, df_submission
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return None, None, None


class Encode:
    def __init__(self, X):
        self.X = X

    def dummying(self):
        encoding_columns = [col for col in self.X.columns if not pd.api.types.is_integer_dtype(self.X[col])]

        X_encoded = pd.get_dummies(self.X[encoding_columns])
        X_encoded = pd.concat([self.X.drop(columns=encoding_columns), X_encoded], axis=1)
        X_encoded['Dependents_Encoded'] = X_encoded['Dependents'].apply(lambda x: x == 2)

        features_selection_2 = ['Telephone_yes', 'ForeignWorker_no', 'ForeignWorker_yes', 'Sex_female',
                                'Dependents', 'CheckingStatus_less_0', 'Housing_rent', 'EmploymentDuration_unemployed',
                                'OwnsProperty_real_estate', 'OthersOnLoan_none', 'CreditHistory_no_credits',
                                'ExistingSavings_less_100']

        X_encoded = X_encoded.drop(columns=features_selection_2)
        return X_encoded


class Label:
    def __init__(self, y):
        self.y = y

    def labeling(self):
        label_encoder = LabelEncoder()
        label_encoder.fit(['No Risk', 'Risk'])
        y_labeled = label_encoder.transform(self.y)
        return y_labeled


class Split:
    def __init__(self, X_encoded, y_labeled):
        self.X_encoded = X_encoded
        self.y_labeled = y_labeled

    def splitting(self, test_size=0.1, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.X_encoded, self.y_labeled, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
