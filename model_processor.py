from os import listdir
from os.path import join, abspath, dirname
from inspect import getmembers
import pandas as pd

from configs import hyper_parameter_path, learning_model_path, pred_data_path
from data_access import model_from_to_json, callfunc
from logger import get_time


class trainModel:
    def __init__(self, args=None, is_prediction=False):
        self.train_args = args
        self.model_init_params = {}
        self.hyper_parameters = model_from_to_json(hyper_parameter_path, [], False)
        self.models = model_from_to_json(learning_model_path, [], False)
        self.files = [f for f in listdir(dirname(abspath(__file__)))]
        self.is_pred = is_prediction
        self.pred_data = []
        self.uids = []

    def define_train_args(self):
        self.params = {
                        "is_training_with_c_of_last_transactions": int(self.train_args[2]),
                        "run_model": self.train_args[3]
                    }

    def check_features_existed(self, pred_cols):
        if len(set(pred_cols) & set(list(self.pred_data.columns))) != 0:
            self.pred_data = self.pred_data.drop(list(set(pred_cols) & set(self.pred_data.columns)), axis=1)

    def get_pred_concat(self, m, data):
        self.uids = [self.models[m]['args']['uid'][num] for num in self.models[m]['args']['uid']]
        pred_cols = [col for col in data.columns if len(col.split(self.models[m]['args']['pred_field'])) ==2]
        if len(self.pred_data) != 0:
            self.check_features_existed(pred_cols)
            self.pred_data = pd.merge(self.pred_data, data[self.uids + pred_cols], on=self.uids, how='left')
        else:
            self.pred_data = data

    def process(self):
        print("*"*20, "Train Process", "*"*20) if not self.is_pred else print("*"*20, "Prediction Process", "*"*20)
        get_time()
        self.define_train_args()
        for m in self.models:
            if self.models[m]['args']['py_file'] in self.files:
                if self.params['run_model'] == 'all' or self.params['run_model'] == m:
                    print("Model :", self.models[m]['name'])
                    _file_path = join(dirname(__file__), self.models[m]['args']['py_file'])
                    model_py = callfunc(_file_path)
                    model = [o[1] for o in getmembers(model_py) if o[0] == self.models[m]['args']['calling']][0]
                    model = model(hyper_parameters=self.hyper_parameters[m],
                                  last_day_predictor=self.params['is_training_with_c_of_last_transactions'],
                                  params=self.models[m])
                    _module = self.models[m]['args']['prediction'] if self.is_pred else self.models[m]['args']['train']
                    model_process = [o[1] for o in getmembers(model) if o[0] == _module][0]
                    model_process()
                    if self.is_pred:  # if it is on prediction env. concat outputs in prediction_data
                        self.get_pred_concat(m, model.test)
            else:
                print("Pls add .py file for model :", m)
        if self.is_pred:  # import data to merged prediction data
            self.pred_data.to_csv(pred_data_path, index=False)

