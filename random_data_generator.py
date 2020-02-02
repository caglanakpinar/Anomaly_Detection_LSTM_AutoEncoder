import random
import datetime
import pandas as pd

from configs import card_ids, customer_ids
from configs import sec_diff_morning, sec_diff_evening, sec_diff_night
from configs import data_path


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


def generate_random_data(start_date, end_date, amount_range, merchant_ids):
    count = 0
    data = []
    merchant_ids = merchant_segments(amount_range, merchant_ids)
    while start_date < end_date:
        _merchant = random.sample(merchant_ids, 1)[0]
        _merchant_id = _merchant[0]
        _card = random.sample(card_ids, 1)[0]
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
                     'RequestInsertTime': start_date + datetime.timedelta(seconds=_secs),
                     'MerchantId': _merchant[0],
                     'customer_id': _customer,
                     'card_id': _card,
                     'Amount': _amount
         })
        count += 1
        start_date += datetime.timedelta(seconds=_secs)
    pd.DataFrame(data).to_csv(data_path, index=False)









