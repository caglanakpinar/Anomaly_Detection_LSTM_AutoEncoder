import random
from datetime import timedelta
import pandas as pd
import numpy as np

from configs import sec_diff_morning, sec_diff_evening, sec_diff_night
from configs import data_path


def generate_random_data_v2(start_date, end_date, amount_range, merchant_ids, diff_range, customer_ids):
    data = []
    count = 0
    merchant_ids = merchant_segments(amount_range, merchant_ids)
    anomalies = np.concatenate([np.ones(2000), np.zeros(10000)], axis=0).tolist()
    segments = list(amount_range.keys())
    for c in customer_ids:
        _segs = random.sample(segments, 2)
        _merchants = [m for m in merchant_ids if m[1] in _segs]
        not_related_merchants = [m for m in merchant_ids if m[1] not in _segs]

        for m in _merchants:
            _start_date = start_date + timedelta(seconds=random.sample(list(range(1000)), 1)[0])
            while _start_date < end_date:
                _merchant_id = m[0]
                _amount = random.sample(amount_range[m[1]]['value'], 1)[0]
                _sec = random.sample(diff_range[m[1]]['value'], 1)[0]

                _transaction = 'transaction_' + str(count)
                d = {
                    'PaymentTransactionId': _transaction,
                    'RequestInsertTime': _start_date + timedelta(seconds=_sec),
                    'MerchantId': m[0],
                    'customer_id': c,
                    'Amount': _amount
                }
                if random.sample(anomalies, 1)[0] == 1:
                    _not_related = random.sample(not_related_merchants, 1)[0]
                    d['MerchantId'] = _not_related[0]
                    d['RequestInsertTime'] = d['RequestInsertTime'] - timedelta(seconds=_sec) + timedelta(seconds=1)
                    _start_date += timedelta(seconds=1)
                    if random.sample([0,1], 1)[0] == 1:
                        d['Amount'] *= 1000
                    else:
                        d['Amount'] = 0.1
                else:
                    _start_date += timedelta(seconds=_sec)
                data.append(d)
                count += 1
    pd.DataFrame(data).to_csv(data_path, index=False)


def merchant_segments(amount_range, merchant_ids):
    merchant_ids_v2 = []
    rest_merchant_ids = merchant_ids.copy()
    for r in amount_range:
        if r != list(amount_range.keys())[-1]:
            sample_merchant_ids = random.sample(rest_merchant_ids, int(amount_range[r]['ratio'] * len(merchant_ids)))
            rest_merchant_ids = list(set(rest_merchant_ids) - set(sample_merchant_ids))
        else:
            sample_merchant_ids = rest_merchant_ids
        merchant_ids_v2 += zip(sample_merchant_ids, [r] * len(sample_merchant_ids))
    return merchant_ids_v2


def generate_random_data(start_date, end_date, amount_range, merchant_ids, customer_ids):
    count = 0
    data = []
    merchant_ids = merchant_segments(amount_range, merchant_ids)
    while start_date < end_date:
        _merchant = random.sample(merchant_ids, 1)[0]
        _merchant_id = _merchant[0]
        _customer = random.sample(customer_ids, 1)[0]
        _amount = random.sample(amount_range[_merchant[1]]['value'], 1)[0]
        if start_date.hour in list(range(8,16)):
            _secs = random.sample(sec_diff_morning, 1)[0]
        if start_date.hour in list(range(16,24)):
            _secs = random.sample(sec_diff_evening, 1)[0]
        if start_date.hour in list(range(0,8)):
            _secs = random.sample(sec_diff_night, 1)[0]
        _transaction = 'transaction_' + str(count)
        data.append({
                     'PaymentTransactionId': _transaction,
                     'RequestInsertTime': start_date + timedelta(seconds=_secs),
                     'MerchantId': _merchant[0],
                     'customer_id': _customer,
                     'Amount': _amount
         })
        count += 1
        start_date += timedelta(seconds=_secs)
    pd.DataFrame(data).to_csv(data_path, index=False)









