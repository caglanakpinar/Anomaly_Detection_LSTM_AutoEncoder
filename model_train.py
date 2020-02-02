import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
import joblib
from sklearn.ensemble import IsolationForest

import pickle
import datetime
from configs import date_col, train_end_date, features_data_path, test_data_path, feature
from configs import model_keras_path, model_iso_f_path, model_abnormal_label_path
from data_access import get_data, write_to_csv
from logger import get_time


class ModelTrain:
    def __init__(self, hyper_parameters=None, model_deciding=None, last_day_predictor=None):
        get_time()
        self.data = get_data(features_data_path, True)
        self.current_date = max(self.data['Created_Time']) + datetime.timedelta(days=1)
        self.features = list(feature.keys())
        self.tuned_parameters = hyper_parameters
        self.treshold = None
        self.samples = None
        self.train, self.test = None, None
        self.X, self.Y = None, None
        self.model_iso, self.model_a_c, self.model_a_l_c = None, None, None
        self.last_day_predictor = last_day_predictor
        self.model_deciding = model_deciding

    def train_test_split(self):
        if self.X is None:
            if self.last_day_predictor == 1:
                self.train = self.data[self.data['is_last_day'] == 0]
                self.test = self.data[self.data['is_last_day'] == 1]
            else:
                self.train = self.data[self.data[date_col] < train_end_date]
                self.test = self.data[self.data[date_col] >= train_end_date]
            print("train set :", len(self.train), " || test set :", len(self.test))

    def get_x_y_values(self, is_abnormalities, is_for_prediction):
        column_for_y = 'target' if is_abnormalities else 'label_iso'
        self.X = self.test if is_for_prediction else self.train
        self.X = self.X[self.X[column_for_y].isin([0, 1])][self.features] if is_abnormalities else self.X[self.features]
        self.X = self.X[self.features].values
        if not is_for_prediction:
            self.Y = self.train[column_for_y].values.reshape(len(self.train), -1)
            if is_abnormalities:
                self.Y = self.train[self.train[column_for_y].isin([0, 1])][column_for_y].values
                self.Y = self.Y.reshape(len(self.Y), -1)

    def create_model_anomaly_classifier(self):
        visible = Input(shape=(len(self.features),))
        hidden1 = Dense(100, activation='relu')(visible)
        hidden2 = Dense(50, activation='tanh')(hidden1)
        hidden3 = Dense(10, activation='tanh')(hidden2)
        hidden4 = Dense(5, activation='relu')(hidden3)
        output = Dense(1, activation='sigmoid')(hidden4)
        self.model_a_c = Model(inputs=visible, outputs=output)
        self.model_a_c.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['accuracy'])

    def create_model_abnormal_label_classifier(self):
        visible = Input(shape=(len(self.features),))
        hidden1 = Dense(100, activation='relu')(visible)
        output = Dense(1, activation='sigmoid')(hidden1)
        self.model_a_l_c = Model(inputs=visible, outputs=output)
        self.model_a_l_c.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['accuracy'])

    def train_process(self, model_name, model, model_path):
        model.fit(self.X, self.Y,
                         epochs=int(self.tuned_parameters[model_name]['epochs']),
                         batch_size=int(self.tuned_parameters[model_name]['batch_size']), validation_split=0.2)
        self.model_from_to_json(model_path, model, True)
        return model

    def prediction_classifier(self, label, model):
        self.test['preds_' + label] = model.predict(self.X)
        self.treshold = np.median(self.test['preds_' + label])
        self.test[label] = self.test['preds_' + label].apply(lambda x: 1 if x > self.treshold else 0)

    def prediction_process_isolation_f(self, is_for_prediction):
        self.test['label_iso'] = self.model_iso.predict(self.test[self.features].values)
        self.test['label_iso'] = self.test['label_iso'].apply(lambda x: 1 if x == -1 else 0)
        self.test['decision_scores'] = self.model_iso.decision_function(self.test[self.features].values)
        if not is_for_prediction:
            self.train['label_iso'] = self.model_iso.predict(self.train[self.features].values)
            self.train['label_iso'] = self.train['label_iso'].apply(lambda x: 1 if x == -1 else 0)
            self.train['decision_scores'] = self.model_iso.decision_function(self.train[self.features].values)

    def model_from_to_json(self, path, model=None, is_writing=False):
        if is_writing:
            model_json = model.to_json()
            with open(path, "w") as json_file:
                json_file.write(model_json)
        else:
            json_file = open(path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            return model_from_json(loaded_model_json)

    def model_from_to_pickle(self, is_writing):
        if is_writing:
            pickle.dump(self.model_iso, open(model_iso_f_path, 'wb'))
        else:
            self.model_iso = joblib.load(model_iso_f_path)

    def isolation_forest_learning_process(self):
        print("isolation forest train process is initialized!!")
        get_time()
        self.train_test_split()
        self.get_x_y_values(is_abnormalities=False, is_for_prediction=True) # is_for_prediction sent False unsupervised
        self.model_iso = IsolationForest(n_estimators=self.tuned_parameters['isolation_forest']['num_of_trees'],
                                         max_samples='auto',
                                         contamination=self.tuned_parameters['isolation_forest']['contamination'],
                                         bootstrap=False,
                                         n_jobs=-1, random_state=42, verbose=1).fit(self.X)
        self.model_from_to_pickle(True)
        self.prediction_process_isolation_f(False)

    def anomaly_classifier_learning_process(self):
        print("anomaly_classifier train process is initialized!!")
        get_time()
        self.get_x_y_values(is_abnormalities=False, is_for_prediction=False)
        ## TODO: created model must be generic
        self.create_model_anomaly_classifier()
        self.model_a_c = self.train_process('anomaly_classifier', self.model_a_c, model_keras_path)

    def abnormal_label_classifier_learning_process(self):
        print("Abnormal Label Classifier is initialized!!")
        get_time()
        self.train_test_split()
        self.get_x_y_values(is_abnormalities=True, is_for_prediction=False)
        self.create_model_abnormal_label_classifier()
        self.model_a_l_c = self.train_process('abnormal_label_classifier', self.model_a_l_c, model_abnormal_label_path)

    def model_running(self):
        if self.model_deciding is None or self.model_deciding == 'all':
            self.isolation_forest_learning_process()
            self.anomaly_classifier_learning_process()
            self.abnormal_label_classifier_learning_process()
        if self.model_deciding == 'isolation_forest':
            self.isolation_forest_learning_process()
        if self.model_deciding == 'anomaly_classifier':
            self.anomaly_classifier_learning_process()
        if self.model_deciding == 'abnormal_label_classifier':
            self.abnormal_label_classifier_learning_process()

    def test_data_prediction(self):
        self.train_test_split()
        self.get_x_y_values(is_abnormalities=False, is_for_prediction=True)
        self.model_a_c = self.model_from_to_json(model_keras_path, False)
        self.model_a_l_c = self.model_from_to_json(model_keras_path, False)
        self.model_from_to_pickle(False)
        # anomaly classifier
        self.prediction_classifier('label_a_c', self.model_a_c)
        # isolation forest
        self.prediction_process_isolation_f(True)
        # abnormal label classifier
        self.prediction_classifier('label_a_l_c', self.model_a_l_c)
        # results to .csv
        self.test.to_csv(test_data_path, index=False)

    def parameter_tuning_classification(self):
        batch_size = np.arange(1000, 10000, 1000)
        hidden_layer = np.arange(1, 3, 1)
        hidden_unit = np.arange(10, 100, 10)
        epoch = np.arange(5, 15, 5)
        activations = ['']
