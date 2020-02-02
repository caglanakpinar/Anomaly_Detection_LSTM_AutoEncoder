import datetime

from data_manipuations import clustered_merchant_ratios_feature, customer_transaction_day_diff_feature
from data_manipuations import customer_merchant_amount_ratio, last_month_of_total_transactions, gmm_cluster_p_value
from data_manipuations import remove_noisy_data, gmm_customer_scoring, get_last_day_comparisions


is_local_run = True
is_min_max_norm = False
sample_args = ['main.py', 'feature_engineering', 'all']
data_path = "transaction_data_all_sample.csv"
features_data_path = "features_sample.csv"
test_data_path = 'test_data.csv'
train_data_path = 'train_data.csv'
model_keras_path = 'model_keras.json'
model_iso_f_path = 'iso_forest.sav'
model_abnormal_label_path = 'model_keras_abnormal_label.json'
model_autoencoder_path = 'iso_forest.sav'
hyper_parameter_path = 'hyper_parameters.json'
runs_at_sample_data = False
sample_size = 80000
removing_columns = ['TransactionType', 'TerminalId', 'Created_Time_str', 'source_API', 'source_Common Payment',
                     'source_Donation UI', 'source_User Interface', 'merchant_terminal_id']
at_least_t_count_per_user = 8
at_least_t_count_per_user_gmm = 8
at_least_day_count_for_user = 8
interval_point_for_k_deciding = 0.05

feature = {
    #'transaction_count': {'args': {
    #                   'data': None,
    #                   'noisy_data_remover': remove_noisy_data,
    #                   'num_of_transaction_removing': at_least_t_count_per_user,
    #                   'num_of_days_removing': at_least_day_count_for_user,
    #                   'feature': 'transaction_count',
    #                   'related_columns': [],
    #               },
    #              'calling': get_last_day_comparisions,
    #              'name': 'Last Day Of Customer Transactions'
    #},
    'c_m_ratios': {'args': {
                        'data': None,
                        'noisy_data_remover': remove_noisy_data,
                        'num_of_transaction_removing': at_least_t_count_per_user,
                        'num_of_days_removing': at_least_day_count_for_user,
                        'feature': 'c_m_ratios',
                        'related_columns': []
                    },
                   'calling': clustered_merchant_ratios_feature,
                   'name': 'C. - M. Transaction Ratio Scores'
    },
    'c_freq_diff_p_value': {'args': {
                                        'data': None,
                                        'noisy_data_remover': remove_noisy_data,
                                        'num_of_transaction_removing': at_least_t_count_per_user,
                                        'num_of_days_removing': at_least_day_count_for_user,
                                        'feature': 'c_freq_diff',
                                        'related_columns': [],
                                        },
                                    'calling': customer_transaction_day_diff_feature,
                                    'name': 'C. Difference Of Each Transaction Score'
    },
    'c_m_med_amount_change_p_value': {'args': {
                                                       'data': None,
                                                       'noisy_data_remover': remove_noisy_data,
                                                       'num_of_transaction_removing': at_least_t_count_per_user,
                                                       'num_of_days_removing': at_least_day_count_for_user,
                                                       'feature': 'c_m_med_amount_change', 'related_columns': []
                                                       },
                                              'name': 'C. M. Amount Change On Each Transaction Score',
                                              'calling': customer_merchant_amount_ratio
                                              },
    #'last_month_totals_min_max_p_value': {'args': {
    #                                        'data': None,
    #                                        'noisy_data_remover': remove_noisy_data,
    #                                        'num_of_transaction_removing': at_least_t_count_per_user,
    #                                        'num_of_days_removing': at_least_day_count_for_user,
    #                                        'feature': 'last_month_of_total_transactions', 'related_columns': []},
    #                                     'calling': last_month_of_total_transactions
    #                                    },
    #'gmm_min_max_p_value': {'args': {
    #                            'data': None,
    #                            'noisy_data_remover': remove_noisy_data,
    #                            'num_of_transaction_removing': at_least_t_count_per_user_gmm,
    #                            'num_of_days_removing': at_least_day_count_for_user,
    #                            'feature': 'gmm_min_max_p_value', 'related_columns': []},
    #                            'calling': gmm_customer_scoring
    #                        }
}

anomaly_ratio = 0.001
class_ratio = 0.005
train_end_date = datetime.datetime.strptime('2019-11-01', '%Y-%m-%d')
model_path = 'model.json'
model_iso_f_model = "iso_forest.sav"

date_col = 'Created_Time'
train_end_date = datetime.datetime.strptime('2019-11-01', '%Y-%m-%d')

outputs = ['label', 'label_iso', 'intersection_of_models']
indicator_column_name = 'total_danger_value'

sample_sizes = list(zip(['%20', '%30', '%40', '%50', '%75'], [.2, .3, .4, .5, 0.75]))
models_output = {'label': 'Anomaly Classification', 'label_iso': 'Isolation Forest',
                 'intersection_of_models': 'Intersection Of Models'}
related_columns = ['PaymentTransactionId', 'Created_Time', 'customer_id', 'merchant_id', 'label', 'label_iso', 'Amount']
k_means_cluster_colors = ['cyan', 'red', 'red'] # ['darkgray', 'red']
related_cols = ['customer_merchant_id', 'merchant_id', 'customer_id', 'Amount', 'label_iso', 'label_a_c',
                'intersection_of_models',
                'Created_Time', 'c_m_label_t_count', 'c_m_t_count', 'c_freq_diff', 'RequestInsertTime']


features_cols_2 = {
    'last_month_totals': 'C. Of Last Months Totals',
    'c_freq_diff': 'C. Difference Of Each Transaction Score',
    'c_m_ratios': 'C. - M. Transaction Ratio Scores',
    'c_m_med_amount_change': 'C. M. Amount Change On Each Transaction Score',
    'gmm': 'C. GMM Clustering Score'
}
for f in feature:
    print(features_cols_2[feature[f]['args']['feature']])
    features_cols_2[f] = features_cols_2[feature[f]['args']['feature']]
print(features_cols_2)
alpha = 0.05
train_test_split_ratio  = 0.3

start_date = datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')
end_date = datetime.datetime.now()

merchant_ids = ['merchant_' + str(m) for m in range(3000)]
sec_diff_evening = list(range(1, 20))
sec_diff_morning = list(range(1, 200))
sec_diff_night = list(range(120, 800))
card_ids = ['card_' + str(m) for m in range(3000000)]
customer_ids = ['customer_' + str(m) for m in range(150000)]

amount_range = {'segment_1': {'value': list(range(1, 100)), 'ratio': 0.3},
                'segment_2': {'value': list(range(80, 25000)), 'ratio': 0.5},
                'segment_3': {'value': list(range(25000, 100000)), 'ratio': 0.15},
                'segment_4': {'value': list(range(100000, 1000000)), 'ratio': 0.15}
                }