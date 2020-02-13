import joblib
from sklearn.ensemble import IsolationForest

import pickle
import datetime
from configs import date_col, train_end_date, features_data_path, feature
from configs import model_iso_f_path
from data_access import get_data
from logger import get_time


class ModelTrainIsolationForest:
    def __init__(self, hyper_parameters=None, model_deciding=None, last_day_predictor=None, test_data=None):
        get_time()
        self.data = get_data(features_data_path, True)
        self.current_date = max(self.data['Created_Time']) + datetime.timedelta(days=1)
        self.features = list(feature.keys())
        self.tuned_parameters = hyper_parameters
        self.train, self.test = None, None if test_data is None else test_data
        self.X, self.Y = None, None
        self.model_iso, self.model_a_c, self.model_a_l_c = None, None, None
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
            pickle.dump(self.model_iso, open(model_iso_f_path, 'wb'))
        else:
            # TODO: needs to be updated
            self.model_iso = joblib.load(model_iso_f_path)

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
        if is_for_prediction:
            self.model_iso = self.model_from_to_pickle(False)
        self.test['label_iso'] = self.model_iso.predict(self.test[self.features].values)
        self.test['label_iso'] = self.test['label_iso'].apply(lambda x: 1 if x == -1 else 0)
        self.test['decision_scores'] = self.model_iso.decision_function(self.test[self.features].values)

