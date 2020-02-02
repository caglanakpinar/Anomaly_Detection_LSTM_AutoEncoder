import random
import datetime
import pandas as pd

from configs import merchant_ids, card_ids, customer_ids, sec_diff
from configs import data_path


def generate_random_data(start_date, end_date):
    count = 0
    data = []
    while start_date < end_date:
        _merchant = random.sample(merchant_ids, 1)
        _card = random.sample(card_ids, 1)
        _customer = random.sample(customer_ids, 1)
        _secs = random.sample(sec_diff, 1)
        _transaction = 'transaction_' + str(count)

        data.append({
                     'PaymentTransactionId': _transaction,
                     'RequestInsertTime': start_date + datetime.timedelta(seconds=_secs),
                     'MerchantId': _merchant,
                     'customer_id': _customer,
                     'card_id': _card
         })
        count += 1
        start_date += datetime.timedelta(seconds=_secs)
    pd.DataFrame(data).to_csv(data_path, index=False)









