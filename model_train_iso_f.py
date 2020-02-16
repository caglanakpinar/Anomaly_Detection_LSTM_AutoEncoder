import joblib
from sklearn.ensemble import IsolationForest

import datetime
from configs import date_col, train_end_date, feature_path, features_data_path
from configs import model_iso_f_path
from data_access import get_data, decide_feature_name
from logger import get_time


class ModelTrainIsolationForest:
    """
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    data: it is the transition of raw data with generated features
    features: key of dictionary from feature.json file
    train: train data set of Isolation Forest.
    test: test data set of Isolation Forest. This is splitting according to last_day_predictor
    model_iso: isolation model aim to train
    last_day_predictor: shows how data set splitting into train and test.
                        If it is 0 that means splitting according to train_date_end.
                        IF is is 1 it is splitting according to is_last_day
    train_test_split: splits raw data into the train and test
    get_x_values: gets values of test or train with features
    model_from_to_pickle: model saving to json file
    learning_process_iso_f: fit the model
    prediction_iso_f: predicting model with trained model
    """
    def __init__(self, hyper_parameters=None, last_day_predictor=None, test_data=None):
        get_time()
        self.data = get_data(features_data_path, True)
        self.features = list(decide_feature_name(feature_path).keys())
        self.tuned_parameters = hyper_parameters
        self.train, self.test = None, None if test_data is None else test_data
        self.X, self.Y = None, None
        self.model_iso = None
        self.last_day_predictor = last_day_predictor

    def train_test_split(self):
        if self.last_day_predictor == 1:
            self.train = self.data[self.data['is_last_day'] == 0]
            self.test = self.data[self.data['is_last_day'] == 1]
        else:
            self.train = self.data[self.data[date_col] < train_end_date]
            self.test = self.data[self.data[date_col] >= train_end_date]
        print("train set :", len(self.train), " || test set :", len(self.test))

    def get_x_values(self, is_for_prediction):
        self.X = self.test[self.features].values if is_for_prediction else self.train[self.features].values

    def model_from_to_pickle(self, is_writing):
        if is_writing:
            joblib.dump(self.model_iso, open(model_iso_f_path, 'wb'))
        else:
            return joblib.load(model_iso_f_path)

    def learning_process_iso_f(self):
        print("isolation forest train process is initialized!!")
        get_time()
        self.train_test_split()
        self.get_x_values(is_for_prediction=False) # is_for_prediction sent False unsupervised
        self.model_iso = IsolationForest(n_estimators=self.tuned_parameters['isolation_forest']['num_of_trees'],
                                         max_samples='auto',
                                         contamination=self.tuned_parameters['isolation_forest']['contamination'],
                                         bootstrap=False,
                                         n_jobs=-1, random_state=42, verbose=1).fit(self.X)
        self.model_from_to_pickle(True)

    def prediction_iso_f(self, is_for_prediction):
        self.get_x_values(is_for_prediction=is_for_prediction)
        x = self.X[0:10]
        if is_for_prediction:
            self.model_iso = self.model_from_to_pickle(False)
        print("")
        self.test['label_iso'] = self.model_iso.predict(self.X)
        self.test['label_iso'] = self.test['label_iso'].apply(lambda x: 1 if x == -1 else 0)
        self.test['decision_scores'] = self.model_iso.decision_function(self.X)

