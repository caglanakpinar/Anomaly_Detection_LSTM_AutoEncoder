import numpy as np
import pandas as pd
import datetime
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from math import sqrt
from scipy import stats
from itertools import product

import configs


def min_max_scaling(val, min_val, range_val, avg_range):
    return (val - min_val) / range_val if range_val != 0 else (val - min_val)  / avg_range


def get_day_part():
    hour_dict = {}
    for h in range(24):
        if h in range(8):
            hour_dict[h] = '[0-8)'
        if h in range(8, 12):
            hour_dict[h] = '[8-12)'
        if h in range(12, 18):
            hour_dict[h] = '[12-18)'
        if h in range(18, 24):
            hour_dict[h] = '[18-24)'
    return hour_dict


def get_day_parts_of_data(c_m_dates, start, end, day_parts, renames):
    days = []
    for c_m in c_m_dates:
        _start, _end, _count = c_m_dates[c_m]['start'], c_m_dates[c_m]['end'], c_m_dates[c_m]['total_t_count']
        c_m_days = int((end - start).total_seconds() / 60 / 60 / 24)
        while start < end:
            days += list(product(c_m, c_m_days, _count, start, day_parts))
            start + datetime.timedelta(days=1)
    return pd.DataFrame(days).rename(columns=renames)


def get_min_max_norm(df, cal_col, group_col):
    # remove column if there exists
    remove_cols = [cal_col + i for i in ['_min', '_max', '_range', '_min_max_p_value', '_min_max_p_value']]
    remove_cols = list(set(remove_cols) & set(list(df.columns)))
    if len(remove_cols) != 0:
        df = df.drop(remove_cols, axis=1)
    # calculate range and min max diff
    df[cal_col + '_max'], df[cal_col + '_min'] = df[cal_col], df[cal_col]
    min_max_vals = df.pivot_table(index=group_col, aggfunc={cal_col + '_max': 'max', cal_col + '_min': 'min'}
                                  ).reset_index()
    min_max_vals[cal_col + '_range'] = min_max_vals[cal_col + '_max'] - min_max_vals[cal_col + '_min']
    df = pd.merge(df.drop([cal_col + '_max', cal_col + '_min'], axis=1), min_max_vals, on=group_col, how='left')
    avg_range = np.median(df[df[cal_col + '_range'] == df[cal_col + '_range']][cal_col + '_range'])
    # calculate p_value accordig to min & max value
    df[cal_col + '_min_max_p_value'] = df.apply(lambda row:
                                                min_max_scaling(row[cal_col], row[cal_col + '_min'],
                                                                row[cal_col +  '_range'], avg_range), axis=1)
    return df[[cal_col + '_min_max_p_value']]


def get_descriptive_stats(df, cal_col, group_col, countable_col):
    params = None
    if len(group_col) != 0:
        df[cal_col + '_mean'], df[cal_col + '_std'] = df[cal_col], df[cal_col]
        df[cal_col + '_count'] = df[cal_col]
        params = df.pivot_table(index=group_col, aggfunc={cal_col + '_mean': 'mean', cal_col + '_std': 'std',
                                                          cal_col + '_count': 'count'}).reset_index()
        return params
    else:
        for des in [(np.mean, '_mean'), (np.std, '_std'), (len, '_count')]:
            df[cal_col + des[1]] = des[0](df[cal_col])
        return df


def recalculate_prob_around_avg(p_value, avg_p_value):
    if p_value < avg_p_value:
        return (avg_p_value - p_value) / avg_p_value if avg_p_value != 0 else p_value
    else:
        return (p_value - avg_p_value) / avg_p_value if avg_p_value != 0 else p_value


def get_p_value(df, cal_col):
    df[cal_col + '_t_value'] = df.apply(lambda row: abs((row[cal_col] - row[cal_col+'_mean']) / (row[cal_col + '_std']*np.sqrt(2/row[cal_col+'_count']))), axis=1)
    df[cal_col+'_p_value'] = df.apply(lambda row: stats.t.cdf(row[cal_col + '_t_value'], df=row[cal_col+'_count']-1), axis=1)
    df[cal_col+'_p_value'] = df[cal_col+'_p_value'].apply(lambda x: recalculate_prob_around_avg(x, 0.5))
    return df


def iqr_outlier_seperation(df_GMM, iqr_columns):
    iqr_c = []
    for i in iqr_columns:
        first_quartile = df_GMM['%s'%i].describe()['25%']
        third_quartile = df_GMM['%s'%i].describe()['75%']
        # Interquartile range
        iqr = third_quartile - first_quartile
        # Find outliers
        number_of_outliers = len(df_GMM['%s'%i][(df_GMM['%s'%i] < (first_quartile - 3 * iqr)) | \
                                                (df_GMM['%s'%i] > (third_quartile + 3 * iqr))])
        print("Outlier number in %s is equal to: "%i, number_of_outliers)
        # Drop outliers:
        if number_of_outliers > 0:
            iqr_c += list(df_GMM[((df_GMM['%s'%i] < (first_quartile - 3 * iqr)) |
                                                (df_GMM['%s'%i] > (third_quartile + 3 * iqr)))]['customer_id'].unique())
    return df_GMM.query("customer_id not in @iqr_c"), df_GMM.query("customer_id in @iqr_c")


def remove_noisy_data(data, least_transactions, least_days):
    frequency_customer = data.pivot_table(index='customer_merchant_id', aggfunc={'PaymentTransactionId': 'count'}
                                          ).reset_index().rename(columns={'PaymentTransactionId': 'frequency'})
    more_than_5_transaction_customers = list(frequency_customer[
                                                 frequency_customer['frequency'] >= least_transactions
                                                 ]['customer_merchant_id'])
    if len(more_than_5_transaction_customers) != 0:
        data = data.query("customer_merchant_id in @more_than_5_transaction_customers")
    if 'frequency' not in list(data.columns):
        data = pd.merge(data, frequency_customer[['customer_merchant_id', 'frequency']],
                        on='customer_merchant_id', how='left')
    accepted_customers = data.pivot_table(index='customer_id',
                                          aggfunc={'Created_Time': 'count'}
                                          ).reset_index().query("Created_Time >= @least_days")['customer_id']
    return data#.query("customer_id in @accepted_customers").reset_index(drop=True)


def distance_values_for_each_k_with_elbow_method(X):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k, n_jobs=-1).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_
    k_means_test = pd.DataFrame(list(mapping1.items())).rename(columns={0: 'k', 1: 'distance'})
    return compute_optimum_k(k_means_test, configs.interval_point_for_k_deciding)


def compute_optimum_k(df, alpha):
    # assign as a column distance value of previous cluster count.
    df['distance_prev'] = df['distance'].shift(1)
    df = df.dropna()  # remove k == 1
    # calculate distance difference each k cluster
    df['clusters_distance_diff'] = df['distance_prev'] - df['distance']
    # calculate k standard error of difference values. If k value increases error value changes
    standard_error = (2.58 * np.std(df['clusters_distance_diff'])) / sqrt(len(df))  # alpha = 0.05, (t_table = 1.96)
    df['is_lower_than_std_error'] = df['clusters_distance_diff'].apply(lambda x: 1 if x < standard_error else 0)
    # optimum k values min k which has difference less than standard error
    print("Here are the optimum k Values :")
    print(df[df['is_lower_than_std_error'] == 1].sort_values(by='k', ascending=True))
    print("Optimum K with Elbow :")
    print(df[df['is_lower_than_std_error'] == 1].sort_values(by='k', ascending=True).to_dict('results')[0]['k'])
    return df[df['is_lower_than_std_error'] == 1].sort_values(by='k', ascending=True).to_dict('results')[0]['k']


class MerchantClusters:
    def __init__(self, data):
        print("KMeans is initialized For Merchant Clustering!!")
        self.data = data
        self.current_date = max(data['Created_Time']) + datetime.timedelta(days=1)
        self.frequency_merchant = None
        self.monetary_merchant = None
        self.recency_merchant = None
        self.rfm = None
        self.X = None
        self.kmeans = None
        self.rfm_metrics = ['recency', 'monetary', 'frequency']

    def frequency_calculation(self):
        self.data['Created_Time_prev'] = self.data.sort_values(['Created_Time', 'merchant_id'],
                                                               ascending=True).groupby('merchant_id')['Created_Time'].shift(1)
        self.data['hour_diff'] = self.data.apply(lambda row:
                                                 (row['Created_Time'] - row['Created_Time_prev']).total_seconds() / 60 / 60, axis=1)
        self.frequency_merchant = self.data.pivot_table(index='merchant_id',
                                                        aggfunc={'hour_diff': 'median'}
                                                        ).reset_index().rename(columns={"hour_diff": 'frequency'})

    def monetary_calculation(self):
        self.monetary_merchant = self.data.pivot_table(index='merchant_id',
                                                       aggfunc={'Amount': 'median'}
                                                       ).reset_index().rename(columns={'Amount': 'monetary'})

    def recency_calculation(self):
        self.recency_merchant = self.data.pivot_table(index='merchant_id',
                                                 aggfunc={'Created_Time': 'max'}).reset_index().rename(
            columns={'Created_Time': 'max_date'})
        self.recency_merchant['recency'] = self.recency_merchant.apply(
            lambda row: (self.current_date - row['max_date']).total_seconds() / 60 / 60, axis=1)

    def check_exist_of_rfm_metrics(self):
        query_str = ""
        for metric in self.rfm_metrics:
            if metric not in list(self.rfm.columns):
                query_str += " " + metric + " == " + metric
                query_str += " and " if metric != self.rfm_metrics[-1] else " "
        self.rfm = self.rfm.query(query_str) if  query_str != "" else self.rfm

    def clustering_merchants(self):
        self.rfm = pd.merge(self.recency_merchant,
                            pd.merge(self.frequency_merchant,
                                     self.monetary_merchant, on='merchant_id', how='inner'), on='merchant_id',
                            how='inner')
        self.check_exist_of_rfm_metrics()
        # if all rfm metrics are able to calculated assigned to X variable, otherwise leave it None
        if len(self.rfm_metrics) == len(list(set(self.rfm_metrics) & set(self.rfm.columns))):
            self.X = self.rfm[self.rfm_metrics]

    def compute_k_means_clustering(self):
        if self.X is not None:  # if there is no recency, monetary, frequency values are not able to calculated
            self.kmeans = KMeans(n_clusters=distance_values_for_each_k_with_elbow_method(self.X),
                                 random_state=0, n_jobs=-1).fit(self.X)

    def assign_cluster_labels(self):
        if self.X is not None:  # if there is no recency, monetary, frequency values are not able to calculated
            self.rfm = pd.concat([self.rfm,
                                  pd.DataFrame(self.kmeans.labels_).rename(columns={0: 'merchant_label'})],
                                  axis=1)[['merchant_id', 'recency', 'frequency', 'monetary', 'merchant_label']]
        else:
            for cols in ['merchant_id', 'recency', 'frequency', 'monetary', 'merchant_label']:
                self.rfm[cols] = 0
        re_cols = list((set(list(self.data.columns)) & set(list(self.rfm.columns))) - set(['merchant_id']))
        self.data = pd.merge(self.data, self.rfm.drop(re_cols, axis=1),
                             on='merchant_id', how='left')

    def k_means_clustering(self):
        self.monetary_calculation()
        self.frequency_calculation()
        self.recency_calculation()
        self.clustering_merchants()
        self.compute_k_means_clustering()
        self.assign_cluster_labels()


def clustered_merchant_ratios_feature(data, feature):
    merchant_clustering = MerchantClusters(data)
    merchant_clustering.k_means_clustering()
    data = merchant_clustering.data
    pv_1 = data.pivot_table(index=['customer_id', 'merchant_label'],
                            aggfunc={'PaymentTransactionId': 'count'}).reset_index().rename(
        columns={'PaymentTransactionId': 'c_m_label_t_count'})
    pv_2 = data.pivot_table(index=['merchant_id', 'customer_id'],
                            aggfunc={'PaymentTransactionId': 'count'}).reset_index().rename(
        columns={'PaymentTransactionId': 'c_m_t_count'})
    merchant_labels = data.pivot_table(index=['merchant_id', 'merchant_label'],
                                       aggfunc={'PaymentTransactionId': 'count'}).reset_index()
    pv_2 = pd.merge(pv_2, merchant_labels[['merchant_id', 'merchant_label']], on='merchant_id', how='left')
    pv_1 = pd.merge(pv_2, pv_1, on=['customer_id', 'merchant_label'], how='left')
    pv_1[feature] = pv_1['c_m_t_count'] / pv_1['c_m_label_t_count']
    pv_1 = pv_1.dropna()
    data = pd.merge(data, pv_1, on=['merchant_id', 'customer_id', 'merchant_label'], how='left')
    data[feature] = data[feature].fillna(0)
    return data


def customer_transaction_day_diff_feature(data, feature):
    current_date = max(data['Created_Time']) + datetime.timedelta(days=1)
    recency = data.pivot_table(index='customer_id', aggfunc={'Created_Time': 'min', 'PaymentTransactionId': 'count'}
                               ).reset_index().rename(
        columns={'Created_Time': 'max_date', 'PaymentTransactionId': 'total_c_t_count'})
    recency['customer_freq'] = recency.apply(
        lambda row: ((current_date - row['max_date']).total_seconds() / 60 / 60) / row['total_c_t_count'], axis=1)
    data = pd.merge(data, recency, on='customer_id', how='left')
    data['prev_transactions'] = data.sort_values(['Created_Time', 'customer_id'],
                                                 ascending=True).groupby('customer_id')['Created_Time'].shift(1)
    data['day_diff'] = data.apply(
        lambda row: (row['Created_Time'] - row['prev_transactions']).total_seconds() / 60 / 60, axis=1)
    data = data[data['day_diff'] == data['day_diff']]
    data = data.reset_index(drop=True)
    data['c_freq_diff'] = data.apply(lambda row: abs(row['customer_freq'] - row['day_diff'])
                                     if row['customer_freq'] > row['day_diff'] else row['customer_freq'],
                                     axis=1)
    data = data.reset_index(drop=True)
    if configs.is_min_max_norm:
        data = pd.concat([data,
                          get_min_max_norm(data.copy(), 'c_freq_diff', 'customer_id').reset_index(drop=True)], axis=1)
    else:
        data = pd.merge(data.copy(), get_descriptive_stats(data, 'c_freq_diff', 'customer_id', 'PaymentTransactionId'),
                        on='customer_id', how='left')
        data = get_p_value(data, 'c_freq_diff')
    return data


def customer_merchant_amount_ratio(data, feature):
    m_merchant_customer = data.pivot_table(index='customer_merchant_id',
                                           aggfunc={'Amount': 'median', 'PaymentTransactionId': 'count'}
                                           ).reset_index().rename(
        columns={'Amount': 'c_m_med_amount', 'PaymentTransactionId': 't_transactions'})
    data = pd.merge(data, m_merchant_customer, on='customer_merchant_id', how='left')
    data['c_m_med_amount_change'] = data.apply(lambda row: abs(row['Amount'] - row['c_m_med_amount']), axis=1)
    data = data.reset_index(drop=True)
    if configs.is_min_max_norm:
        data = pd.concat([data,
                          get_min_max_norm(data.copy(),
                                           'c_m_med_amount_change',
                                           'customer_merchant_id'
                                           ).reset_index(drop=True)], axis=1)
    else:
        data = pd.merge(data.copy(),
                        get_descriptive_stats(data,
                                              'c_m_med_amount_change', 'customer_merchant_id', 'PaymentTransactionId'),
                        on='customer_merchant_id', how='left')
        data = get_p_value(data, 'c_m_med_amount_change')
    data[feature] = data.apply(lambda row: 0 if row['t_transactions'] < 5 else row[feature], axis=1)
    return data


def last_month_of_total_transactions(data, feature):
    data_2 = data[
        ['PaymentTransactionId', 'Created_Time', 'Created_Time_str', 'day_diff', 'customer_id', 'Amount']].sort_values(
        ['Created_Time', 'customer_id'], ascending=True)
    data_2['has_last_month_of_transactions'] = data_2['day_diff'].apply(lambda x: 0 if x > 30 else 1)
    data_3 = data_2[data_2['has_last_month_of_transactions'] == 1]
    data_5 = data.pivot_table(index='customer_id',
                              aggfunc={'Amount': lambda x: list(x), 'Created_Time': lambda x: list(x)}).to_dict('index')
    customer_dict = {}
    for c in list(data_3['customer_id'].unique()):
        customer_dict[c] = list(zip(data_5[c]['Amount'], data_5[c]['Created_Time']))
    data = pd.merge(data, data_2[['PaymentTransactionId', 'has_last_month_of_transactions']], on='PaymentTransactionId',
                    how='left')
    data['last_30_days'] = data['Created_Time'].apply(lambda x: x - datetime.timedelta(days=30))
    data = data.reset_index(drop=True)
    data_for_last_month = data[
        ['has_last_month_of_transactions', 'customer_id', 'Amount', 'last_30_days', 'Created_Time']].to_dict('results')
    last_month_amounts = []
    for row in data_for_last_month:
        if row['has_last_month_of_transactions'] == 1:
            last_month_amounts.append(sum(
                map(lambda x: x[0] if x[1] > row['last_30_days'] and x[1] <= row['Created_Time'] else 0,
                    customer_dict[row['customer_id']])))
        else:
            last_month_amounts.append(row['Amount'])
    data = pd.concat([data.reset_index(drop=True), pd.DataFrame(last_month_amounts).rename(columns={0: 'last_month_totals'})], axis=1)
    data.reset_index(drop=True)
    if configs.is_min_max_norm:
        data = pd.concat([data,
                          get_min_max_norm(data.copy(), 'last_month_totals', 'customer_id').reset_index(drop=True)], axis=1)
    else:
        data = pd.merge(data.copy(),
                        get_descriptive_stats(data, 'last_month_totals', 'customer_id', 'PaymentTransactionId'),
                        on='customer_id', how='left')
        data = get_p_value(data, 'last_month_totals')
    return data


def gmm_cluster_p_value(data, features):
    customer_clustering_pobs = pd.read_csv("customer_GMM_label_score_without_iqr.csv").rename(columns={'score': 'gmm_min_max_p_value'})
    df = pd.merge(data, customer_clustering_pobs, on='customer_id', how='left')
    df['total_danger_value'] = df.apply(lambda row: sum([row[f] for f in features]), axis=1)


def gmm_customer_scoring(data, feature):
    data_prev = data.copy()
    current_date = max(data['Created_Time']) + datetime.timedelta(days=1)
    recency_merchant = data.pivot_table(index='customer_id',
                                        aggfunc={'Created_Time': 'max'}
                                        ).reset_index().rename(columns={'Created_Time': 'max_date'})
    recency_merchant['recency'] = recency_merchant.apply(lambda row:
                                                         (current_date - row['max_date']).total_seconds() / 60 / 60,
                                                         axis=1)
    # frequency
    data['Created_Time_prev'] = data.sort_values(['Created_Time', 'customer_id'], ascending=True).groupby('customer_id')['Created_Time'].shift(1)
    data['frequency'] = data.apply(
        lambda row: (row['Created_Time'] - row['Created_Time_prev']).total_seconds() / 60 / 60, axis=1)
    frequency_merchant = data.pivot_table(index='customer_id', aggfunc={'frequency': 'mean'}).reset_index()
    # monetary Median
    agg_dict = {}
    for col in ['median']:
        data['monetary_'+ col] = data['Amount']
        agg_dict['monetary_'+ col] = col

    monetary_merchant = data.pivot_table(index='customer_id', aggfunc=agg_dict).reset_index()
    rfm = pd.merge(pd.merge(recency_merchant, frequency_merchant,
                            on='customer_id', how='inner'), monetary_merchant,
                   on='customer_id', how='inner')
    # Re Order dataframe
    df_GMM = rfm[['recency', 'frequency', 'monetary_median', 'customer_id']]
    df_GMM, df_GMM_iqr = iqr_outlier_seperation(df_GMM, ['recency', 'frequency', 'monetary_median'])
    del rfm
    # Make it array
    X = df_GMM[['recency', 'frequency', 'monetary_median']].values
    # GaussianMixture Model
    gmm = GaussianMixture(n_components=4, covariance_type='spherical').fit(X)
    labels = gmm.predict(X)
    score = pd.DataFrame(gmm.predict_proba(X)).rename(columns={i: str(i) + '_label' for i in range(4)})
    df_GMM['labels'] = labels
    df_GMM = pd.concat([df_GMM, score], axis=1)

    df_GMM['gmm'] = df_GMM.apply(lambda row:
                                 row[str(int(row['labels'])) + '_label'] if row['labels'] == row['labels'] else 1, axis=1)
    df_GMM = df_GMM.reset_index(drop=True)
    df_GMM = pd.concat([df_GMM,
                        get_min_max_norm(df_GMM.copy(),  'gmm', 'labels').reset_index(drop=True)], axis=1)
    df_GMM[feature] = 1 - df_GMM[feature]
    df_GMM_iqr[feature] = 1
    df_GMM = pd.concat(([df_GMM[[feature, 'customer_id']].reset_index(drop=True),
                         df_GMM_iqr[[feature, 'customer_id']].reset_index(drop=True)]))
    data = pd.merge(data_prev, df_GMM[[feature, 'customer_id']], on='customer_id', how='left')
    return data


def get_last_day_comparisions(data, feature):
    data['amount_mean'], data['amount_max'], data['amount_min'] = data['Amount'], data['Amount'], data['Amount']
    data['amount_total'] = data['Amount']
    data['transaction_count'] = data['PaymentTransactionId']
    data['day'] = data['Created_Time'].apply(lambda x: datetime.datetime.strptime(str(x)[0:10], '%Y-%m-%d'))
    data_pv = data.pivot_table(index=['customer_id', 'day'], aggfunc={'amount_mean': 'mean',
                                                                      'amount_max': 'max',
                                                                      'amount_min': 'min',
                                                                      'amount_total': 'sum',
                                                                      'transaction_count': 'count',
                                                                      }).reset_index()
    data_pv = data_pv.sort_values(by=['customer_id', 'day'])
    data['day'] = data['Created_Time'].apply(lambda x: datetime.datetime.strptime(str(x)[0:10], '%Y-%m-%d'))
    return pd.merge(data.drop('transaction_count', axis=1),
                    data_pv[['customer_id', 'transaction_count']], on='customer_id', how='left')


def get_customer_label_encoder(data, feature):
    data['day'] = data['day'].apply(lambda x: datetime.datetime.strptime(str(x)[0:10], '%Y-%m-%d'))
    customers = data.pivot_table(index=['customer_id', 'day'],
                                 aggfunc={'transaction_count': 'mean'}).reset_index().sort_values(by='transaction_count',
                                                                                                  ascending=True,
                                                                                                 ).reset_index(drop=True).reset_index()[['index', 'customer_id']]
    data = pd.merge(data, customers, on='customer_id', how='left').rename(columns={'index': 'customers'})


def customer_merchant_peak_drop(data, feature):
    # max
    the_most_peak = data.pivot_table(index='customer_merchant_id',
                                     aggfunc={'Amount': 'max'}).reset_index().rename(columns={'Amount': 'amount_max'})
    transaction_ids = pd.merge(data,
                               the_most_peak,
                               on='customer_merchant_id',
                               how='left').query("amount_max == amount_max")[
        ['PaymentTransactionId', 'customer_merchant_id']]
    data['Created_Time_prev'] = data.sort_values(['Amount', 'customer_merchant_id'],
                                                 ascending=False).groupby('customer_merchant_id')['Created_Time'].shift(1)
    the_most_peak_2 = data[data['Created_Time_prev'] == data['Created_Time_prev']].pivot_table(
        index='customer_merchant_id',
        aggfunc={'Amount': 'max'}).reset_index().rename(columns={'Amount': 'amount_max_2nd'})
    the_most_peak = pd.merge(the_most_peak, the_most_peak_2, on='customer_merchant_id', how='inner')
    the_most_peak = pd.merge(the_most_peak, transaction_ids, on='customer_merchant_id', how='inner')
    the_most_peak['the_most_max_diff'] = the_most_peak['amount_max'] - the_most_peak['amount_max_2nd']
    the_most_peak = the_most_peak.query("the_most_max_diff != 0")
    # min
    the_least_peak = data.pivot_table(index='customer_merchant_id',
                                      aggfunc={'Amount': 'min'}
                                      ).reset_index().rename(columns={'Amount': 'amount_min'})
    transaction_ids = pd.merge(data,
                               the_least_peak,
                               on='customer_merchant_id',
                               how='left').query("amount_min == amount_min")[
        ['PaymentTransactionId', 'customer_merchant_id']]
    data['Created_Time_prev'] = data.sort_values(
        ['Amount', 'customer_merchant_id'], ascending=True
    ).groupby('customer_merchant_id')['Created_Time'].shift(1)
    the_least_peak_2 = data[data['Created_Time_prev'] == data['Created_Time_prev']].pivot_table(
        index='customer_merchant_id',
        aggfunc={'Amount': 'min'}).reset_index().rename(columns={'Amount': 'amount_min_2nd'})
    # score calculation for both min and max values
    the_least_peak = pd.merge(the_least_peak, the_least_peak_2, on='customer_merchant_id', how='inner')
    the_least_peak = pd.merge(the_least_peak, transaction_ids, on='customer_merchant_id', how='inner')
    the_least_peak['the_least_min_diff'] = the_least_peak['amount_min_2nd'] - the_least_peak['amount_min']
    the_least_peak = the_least_peak.query("the_least_min_diff != 0")
    metrics = [(the_most_peak, 'the_most_max_diff'), (the_least_peak, 'the_least_min_diff')]
    for metric in metrics:
        min_value = min(list(metric[0][metric[0][metric[1]] == metric[0][metric[1]]][metric[1]]))
        max_value = max(list(metric[0][metric[0][metric[1]] == metric[0][metric[1]]][metric[1]]))
        metric[0][metric[1] + '_min_max_p_value'] = metric[0].apply(
            lambda row: min_max_scaling(row[metric[1]], min_value, max_value - min_value, None), axis=1)
        data = pd.merge(data, metric[0], on='PaymentTransactionId', how='left')
        data[metric[1] + '_min_max_p_value'] = data[metric[1] + '_min_max_p_value'].fillna(0)
    col_1, col_2 = metrics[0][1] + '_min_max_p_value', metrics[1][1] + '_min_max_p_value'
    cond = lambda x: x[col_1] if x[col_1] == x[col_1] else x[col_2] if x[col_2] == x[col_2] else 0
    data[feature] = data.apply(cond, axis=1)
    return data


def get_customer_merchant_hourly_sequential_data(data, feature):
    group_cols = ['customer_merchant_id', 'Created_Time', 'day_part']
    day_parts = get_day_part()
    data['day_part'] = data['hour'].apply(lambda x: day_parts[x])
    data_pv = data.pivot_table(index=group_cols,
                               aggfunc={'Amount': 'median'}).reset_index()
    data_pv = data_pv.sort_values(by=group_cols, ascending=True)
    data_pv['start'], data_pv['end'] = data_pv['Created_Time'], data_pv['Created_Time']
    c_m_dates = data_pv.pivot_table(index='customer_merchant_id',
                                    aggfunc={'start': 'min', 'end': 'max', 'total_t_count': 'count'}).to_dict('index')
    dates = get_day_parts_of_data(c_m_dates=c_m_dates,
                                  start=min(data['Created_Time']),
                                  end=max(data['Created_Time']),
                                  day_parts=list(day_parts.keys()),
                                  renames={0: 'customer_merchant_id', 1: group_cols[0], 2: 'total_days',
                                           3: group_cols[1], 4: group_cols[2]}
                                  )
    dates = pd.merge(dates, data_pv, on=group_cols, how='left')
    dates['Amount'] = dates['Amount'].fillna(0)
    return dates




