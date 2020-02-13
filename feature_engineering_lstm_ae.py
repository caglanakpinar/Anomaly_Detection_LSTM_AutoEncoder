import datetime
import warnings
warnings.filterwarnings("ignore")

from configs import data_path, removing_columns, features_data_path, alpha, feature_lstm_ae
from data_access import get_data, write_to_csv, decide_feature_name
from data_manipuations import get_p_value, get_descriptive_stats
from logger import get_time

class CreateFeatures:
    def __init__(self, model_deciding=None):
        get_time()
        self.data_path = data_path if model_deciding == 'all' else features_data_path
        self.data = get_data(path=self.data_path, is_for_model=False)
        self.columns = list(self.data.columns)
        self.features = decide_feature_name(feature_lstm_ae)
        self.removing_columns = removing_columns
        self.model_deciding = model_deciding

    def deciding_computing_features(self):
        if self.model_deciding != 'all':
            feature_2 = {}
            for f in self.model_deciding.split("-"):
                feature_2[f] = self.features[f]
            self.features = feature_2
            print("features : ", self.model_deciding.split("-"))

    def check_features_existed(self, feature_col, related_cols):
        if feature_col in list(self.columns):
            print("feature columns :", feature_col, " is deleted!!")
            self.data = self.data.drop(feature_col, axis=1)
        if len(set(related_cols) & set(list(self.columns))) != 0:
            print("removing columns related to feature :", list(set(related_cols) & set(list(self.columns))))
            self.data = self.data.drop(related_cols, axis=1)

    def features_data_arrange(self):
        for f in self.features:
            self.features[f]['args']['data'] = self.data

    def removig_columns(self):
        if len(self.removing_columns) != 0:
            self.data = self.data.drop(list(set(list(self.data.columns) & set(self.removing_columns))), axis=1)

    def labeling_anormalities(self, f):
        # TODO: Each Feaure of Right Side outlies detected y T- Normal Distribution For each Transaction.
        # TODO: Intersection Of Each Transaction
        self.data[f + '_score'] = self.data[f]
        self.data = get_descriptive_stats(self.data, f + '_score', [], [])
        self.data = get_p_value(self.data, f + '_score')
        self.data[f + '_score_p_value'] = self.data[f + '_score_p_value'].apply(lambda x: 0 if x != x else x)
        self.data[f + '_h0_rejected'] = self.data[f + '_score_p_value'].apply(lambda x:
                                                                              1 if x > alpha else 0 if x < alpha else '-')

    def assign_target_variable(self):
        if 'target' not in list(self.data.columns):
            a_l_total = lambda row: sum([row[f + '_h0_rejected'] for f in list(self.features.keys())])
            self.data['target'] = self.data.apply(lambda row: 1 if a_l_total == len(self.features) else 0, axis=1)

    def assign_last_day_label(self):
        if 'is_last_day' not in list(self.data.columns):
            self.data = self.data.merge(self.data.rename(columns={'Created_Time': 'day_max'}
                                                         ).pivot_table(index='customer_id', aggfunc={'day_max': 'max'}
                                                                ).reset_index(), on='customer_id', how='left')
            self.data['is_last_day'] = self.data.apply(lambda row: 1 if row['Created_Time'] == row['day_max'] else 0, axis=1)

    def compute_features(self):
        get_time()
        self.features_data_arrange()
        for f in self.features:
            print("Feature :", f)
            self.check_features_existed(self.features[f]['args']['feature'], self.features[f]['args']['related_columns'])
            if self.features[f]['args']['num_of_transaction_removing']:
                self.data = self.features[f]['args']['noisy_data_remover'](self.data,
                                                                           self.features[f]['args'][
                                                                               'num_of_transaction_removing'],
                                                                           self.features[f]['args'][
                                                                               'num_of_days_removing'],
                                                                           )
            self.data = self.features[f]['calling'](self.data, f)
            self.labeling_anormalities(f)
            print("data sample size :", len(self.data))
        self.assign_target_variable()
        self.assign_last_day_label()
        write_to_csv(self.data, features_data_path)


