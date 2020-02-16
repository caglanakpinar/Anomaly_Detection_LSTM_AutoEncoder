import datetime



is_local_run = True
is_min_max_norm = True
run_from_sample_data = False
sample_args = ['main.py', 'dashboard', '0']
data_path = "transaction_data_all_sample_.csv"
features_data_path = "features_.csv"
test_data_path = 'test_data.csv'
train_data_path = 'train_data.csv'
auto_encoder_model_paths = {'ae': 'auto_encoder.json', 'ae_l': 'auto_encoder_linear.json'}
model_iso_f_path = 'iso_forest.sav'
hyper_parameter_path = 'hyper_parameters.json'
feature_lstm_ae = 'features_lstm_ae.json'
feature_path = 'features.json'
sample_size = 80000
removing_columns = ['TransactionType', 'TerminalId', 'Created_Time_str', 'source_API', 'source_Common Payment',
                     'source_Donation UI', 'source_User Interface', 'merchant_terminal_id']
at_least_t_count_per_user = 8
at_least_t_count_per_user_gmm = 8
at_least_day_count_for_user = 8
interval_point_for_k_deciding = 0.05

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
models_output = {'anomaly_ae_values': 'AutoEncoder Anomaly Score',
                 'label_iso': 'Isolation Forest Labels',
                 }
related_columns = ['PaymentTransactionId', 'Created_Time', 'customer_id', 'merchant_id', 'label', 'label_iso', 'Amount']
k_means_cluster_colors = ['cyan', 'red', 'red'] # ['darkgray', 'red']


related_cols = ['customer_merchant_id', 'merchant_id', 'customer_id', 'Amount',
                'Created_Time', 'c_m_label_t_count', 'c_m_t_count', 'c_freq_diff', 'RequestInsertTime']


features_cols_2 = {
    'last_month_totals': 'C. Of Last Months Totals',
    'c_freq_diff': 'C. Difference Of Each Transaction Score',
    'c_m_ratios': 'C. - M. Transaction Ratio Scores',
    'c_m_med_amount_change': 'C. M. Amount Change On Each Transaction Score',
    'gmm': 'C. GMM Clustering Score'
}


alpha = 0.05
train_test_split_ratio = 0.3

start_date = datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')
end_date = datetime.datetime.now()

merchant_ids = ['merchant_' + str(m) for m in range(40)]
sec_diff_evening = {0: list(range(1, 20))}
sec_diff_morning = {1: list(range(1, 100))}
sec_diff_night = {'': list(range(80, 300))}
customer_ids = ['customer_' + str(m) for m in range(1500)]

amount_range = {'segment_1': {'value': list(range(1, 100)), 'ratio': 0.2},
                'segment_2': {'value': list(range(80, 1000)), 'ratio': 0.35},
                'segment_3': {'value': list(range(1000, 25000)), 'ratio': 0.25},
                'segment_4': {'value': list(range(25000, 100000)), 'ratio': 0.15},
                'segment_5': {'value': list(range(100000, 1000000)), 'ratio': 0.5}
                }

diff_range = {'segment_1': {'value': list(range(1, 324000)), 'ratio': 0.5},
              'segment_2': {'value': list(range(324000, 648000)), 'ratio': 0.3},
              'segment_3': {'value': list(range(648000, 1296000)), 'ratio': 0.5},
              'segment_4': {'value': list(range(1296000, 2592000)), 'ratio': 0.15},
              'segment_5': {'value': list(range(2592000, 5184000)), 'ratio': 0.15}
            }
host, port = '127.0.0.1', 8050