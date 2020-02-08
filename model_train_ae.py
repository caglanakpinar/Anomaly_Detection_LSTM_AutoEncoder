import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from configs import date_col, train_end_date, features_data_path, feature, auto_encoder_model_paths
from data_access import get_data
from logger import get_time


class ModelTrainAutoEncoder:
    def __init__(self, hyper_parameters=None, last_day_predictor=None, test_data=None):
        get_time()
        self.data = get_data(features_data_path, True)
        self.features = list(feature.keys())
        self.tuned_parameters = hyper_parameters
        self.last_day_predictor = last_day_predictor
        self.train, self.test = None, None if test_data is None else test_data
        self.X, self.y_pred, self.y = None, None, None
        self.input, self.fr_output = None, None
        self.model_ae, self.model_ae_l = None, None

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

    def auto_encoder(self):
        self.input = Input(shape=(len(self.features),))
        encoder1 = Dense(200, activation='tanh')(self.input)
        encoder2 = Dense(100, activation='tanh')(encoder1)
        encoder3 = Dense(50, activation='tanh')(encoder2)
        encoder4 = Dense(10, activation='tanh')(encoder3)
        self.fr_output = Dense(1, activation='tanh')(encoder4)
        decoder1 = Dense(10, activation='tanh')(self.fr_output)
        decoder2 = Dense(50, activation='tanh')(decoder1)
        decoder3 = Dense(100, activation='tanh')(decoder2)
        decoder4 = Dense(200, activation='tanh')(decoder3)
        decoder5 = Dense(len(self.features), activation='sigmoid')(decoder4)
        self.model_ae = Model(inputs=self.input, outputs=decoder5)
        self.model_ae.compile(loss='mse', optimizer=RMSprop(learning_rate=0.01), metrics=['mse'])

    def auto_encoder_linear(self):
        self.input = Input(shape=(len(self.features),))
        self.fr_output = Dense(1, activation='relu')(self.input)
        decoder = Dense(len(self.features), activation='sigmoid')(self.fr_output)
        self.model_ae_l = Model(inputs=self.input, outputs=decoder)
        self.model_ae_l.compile(loss='mse', optimizer=RMSprop(learning_rate=0.01), metrics=['mse'])

    def feature_reduction(self, model):
        return Model(inputs=self.input, outputs=self.fr_output)

    def model_train(self):
        print("Auto Encoder is initialized!!")
        get_time()
        self.train_test_split()
        self.get_x_values(is_for_prediction=False)
        self.auto_encoder()
        self.auto_encoder_linear()
        self.model_ae.fit(self.X, self.X,
                          epochs=int(self.tuned_parameters['feature_reduction']['epochs']),
                          batch_size=int(self.tuned_parameters['feature_reduction']['batch_size']),
                          validation_split=0.2)
        self.model_ae_l.fit(self.X, self.X,
                            epochs=int(self.tuned_parameters['feature_reduction']['epochs']),
                            batch_size=int(self.tuned_parameters['feature_reduction']['batch_size']),
                            validation_split=0.2)

        for m in [(self.model_ae, 'ae'), (self.model_ae_l, 'ae_l')]:
            self.model_from_to_json(auto_encoder_model_paths[m[1]], self.feature_reduction(m[0]), is_writing=True)

    def calculating_loss_function(self, is_for_prediction):
        if is_for_prediction:
            self.model_ae = self.model_from_to_json(auto_encoder_model_paths['ae'], is_writing=False)
            self.model_ae_l = self.model_from_to_json(auto_encoder_model_paths['ae_l'], is_writing=False)
        self.y_pred, self.y = self.model_ae.predict(self.X), self.model_ae_l.predict(self.X)
        # AE works better when we try to find out the outliers
        anomaly_calculations = list(map(lambda x: abs(x[0] - x[1])[0], zip(self.y_pred, self.y)))
        if is_for_prediction:
            self.test['anomaly_ae_values'] = pd.Series(anomaly_calculations)
        else:
            self.train['anomaly_ae_values'] = pd.Series(anomaly_calculations)
        self.test = self.test[['anomaly_ae_values'] + self.features].sort_values(by='anomaly_ae_values',
                                                                                 ascending=False)

        self.test.to_csv("prediction.csv", index=False)

