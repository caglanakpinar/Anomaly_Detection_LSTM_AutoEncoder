import pandas as pd
from math import sqrt
import numpy as np
import joblib
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from configs import date_col, train_end_date, main_data_path
from data_access import decide_feature_name, model_from_to_json
from data_access import get_data
from logger import get_time


def get_query_str(f, c):
    """
    allows you to shape query for each centriod to each feature column
    :param f: feature list
    :param c: centriod list, same size as features
    :return: query string
    """
    print([f[1] + " >= " + str(f[0]) + " and " for f in zip(c, f)])
    return "  ".join([f[1] + " >= " + str(f[0]) + " and " for f in zip(c, f) if f[0] == f[0]])[:-4]


def compute_optimum_parameter(df, paramter, outliers, sorting=False):
    # assign as a column distance value of previous cluster count.
    df[outliers + '_prev'] = df[outliers].shift(1)
    df = df.dropna()  # remove k == 1
    # calcualte distance difference each k cluster
    df['diff'] = df[outliers + '_prev'] - df[outliers]
    # calcualte k standart error of difference values. If k value increases error value changes
    standart_error = (2.58 * np.std(df['diff'])) / sqrt(len(df))  # alpha = 0.05, (t_table = 1.96)
    df['is_lower_than_std_error'] = df['diff'].apply(lambda x: 1 if x < standart_error else 0)
    # optmum k values min k whch has differenc less than standart error
    return df[df['is_lower_than_std_error'] == 1].sort_values(by=paramter, ascending=sorting).to_dict('results')[0][paramter]


class ModelTrainDBScan:
    """
    This class works for DBScan It gathers paramteres from hyper_parmaters.json.
    """
    def __init__(self, hyper_parameters=None, model_deciding=None, last_day_predictor=None, params=None):
        get_time()
        self.data = get_data(main_data_path + params['args']['data'], True)  # data that created at feature engineering
        self.features = list(decide_feature_name(main_data_path + params['args']['feature_set']).keys())
        self.params = hyper_parameters  # get hyper parameters for model: hyper_parameters.json
        self.model_params = params
        self.train, self.test = None, None
        self.X = None
        self.optimum_cluster_centroids = None
        self.centroids = None
        self.po_data = None  # Possible_outlier_transactions data
        self.model_dbscan = None
        self.m_s, self.eps = [], []
        self.o_min_sample = None
        self.o_epsilon = None
        self.o_devision = None
        self.last_day_predictor = last_day_predictor  # splitting data indicator
        self.uids = None

    def train_test_split(self):
        if self.last_day_predictor == 1:
            self.train = self.data[self.data['is_last_day'] == 0]
            self.test = self.data[self.data['is_last_day'] == 1]
        else:
            self.train = self.data[self.data[date_col] < train_end_date]
            self.test = self.data[self.data[date_col] >= train_end_date]
        print("train set :", len(self.train), " || test set :", len(self.test))

    def find_optimum_centroids_with_kmeans(self):
        self.optimum_cluster_centroids = KMeans(n_clusters=1, n_jobs=10)
        self.optimum_cluster_centroids.fit(self.data[self.features].values)
        self.centroids = self.optimum_cluster_centroids.cluster_centers_.tolist()[0]
        self.po_data = self.data.query(get_query_str(self.features, self.centroids))

    def get_outlier_count(self, e, s):
        return len([l for l in DBSCAN(eps=e, min_samples=s, n_jobs=10).fit(self.po_data[self.features].values).labels_ if l == -1])

    def optimum_min_samples(self):
        self.m_s = [int(len(self.po_data)*prc) for prc in np.arange(0.05, 0.32, 0.04).tolist() + np.arange(0.01, 0.05, 0.02).tolist()]
        self.m_s = pd.DataFrame(map(lambda s: {"outliers": self.get_outlier_count(e=0.1, s=s), "m_s": s}, sorted(self.m_s)))
        self.o_min_sample = compute_optimum_parameter(self.m_s, 'm_s', 'outliers')

    def optimum_epsilon(self):
        self.eps = np.arange(0.005, 1, 0.005).tolist()
        self.eps = pd.DataFrame(map(lambda e: {"outliers": self.get_outlier_count(e=e, s=self.o_min_sample), "eps": e},
                                    sorted(self.eps)))
        self.o_epsilon = compute_optimum_parameter(self.eps, 'eps', 'outliers', sorting=True)

    def get_x_values(self, div):
        self.po_data = self.data.query(get_query_str(self.features, np.array(self.centroids) / div))
        self.X = self.po_data[self.features].values

    def get_distance_of_outliers(self, condition, indicator):
        values = []
        for f in self.features:
            val = 0
            if len(self.po_data.query(condition)[f]) != 0:
                if indicator(self.po_data.query(condition)[f]) == indicator(self.po_data.query(condition)[f]):
                    val = indicator(self.po_data.query(condition)[f])
            values.append(val)
        return values

    def learning_process_dbscan(self):
        print("DBSCAN train process is initialized!!")
        get_time()
        print("KMeans Finding Best Centroids process is started!!")
        self.find_optimum_centroids_with_kmeans()
        print("Parameter Tuning For Epsilon and Min_Samples!!")
        self.optimum_min_samples()
        self.optimum_epsilon()
        print("number of data for DBSCAN :", len(self.po_data))
        print({'eps': self.o_epsilon, 'min_samples': self.o_min_sample, 'centroids': {c for c in self.centroids}})
        print("Optimum Centriod Divison is Initialized!!!")
        cal_divs = []
        for div in range(2, self.params['centroid_divide_range']):
            print("divide :", div)
            self.get_x_values(div)
            print(len(self.po_data) - self.o_min_sample)
            self.po_data['label_dbscan'] = DBSCAN(eps=self.o_epsilon,
                                            min_samples= len(self.po_data) - self.o_min_sample,
                                            n_jobs=-1).fit(self.X).labels_
            cal_divs.append({"cal": np.mean(np.abs(np.sum([self.get_distance_of_outliers("label_dbscan != -1", max),
                                                           np.multiply(self.get_distance_of_outliers("label_dbscan == -1", min), -1)]))), "div": div})
        print("optimum centriod distance to outliers results :")
        print(cal_divs)
        self.o_devision = list(pd.DataFrame(cal_divs).sort_values(by='cal', ascending=False)['div'])[0]
        print("optimum ", self.o_devision)
        print({'eps': self.o_epsilon, 'min_samples': self.o_min_sample,
               'centroids': {c for c in self.centroids}, "div": self.o_devision})
        model_from_to_json(main_data_path + self.model_params['args']['model_file'],
                           {'eps': self.o_epsilon,
                            'min_samples': self.o_min_sample,
                            'centroids': {c[0]: c[1] for c in zip(self.features, self.centroids)},
                            'optimum_divison': self.o_devision
                            },
                           True)
        print("DBSCAN Train Process Done!")

    def prediction_dbscan(self):
        print("DBSCAN Prediction Process Initialized!")
        self.uids = [self.model_params['args']['uid'][num] for num in self.model_params['args']['uid']]
        self.params = model_from_to_json(main_data_path + self.model_params['args']['model_file'], [], is_writing=False)
        self.optimum_cluster_centroids = [self.params['centroids'][f] / 14 for f in self.features]
        self.po_data = self.data.query(get_query_str(self.features, self.optimum_cluster_centroids))
        print("number of data for DBSCAN :", len(self.po_data))
        print("epsilon :", self.params['eps'], " || min_samples : ", self.params['min_samples'])
        self.po_data[self.model_params['args']['pred_field']] = DBSCAN(eps=self.params['eps'],
                                                                       min_samples=len(self.po_data) - int(len(self.po_data)*0.01),
                                                                       n_jobs=-1).fit(self.po_data[self.features].values).labels_
        self.po_data[self.model_params['args']['pred_field']] = self.po_data[self.model_params['args']['pred_field']].apply(lambda x: -1 if x == -1 else 0)
        self.train_test_split()
        self.test = pd.merge(self.test,
                             self.po_data[self.uids + [self.model_params['args']['pred_field']]],
                             on=self.uids, how='left')
        self.test[self.model_params['args']['pred_field']] = self.test[self.model_params['args']['pred_field']].fillna(0)
        print("DBSCAN Prediction Process Done!")
