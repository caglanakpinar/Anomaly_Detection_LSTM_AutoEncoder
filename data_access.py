import pandas as pd
import datetime
import random
import json
import os
from os import listdir

from configs import runs_at_sample_data, sample_size, start_date, end_date, amount_range, merchant_ids
from random_data_generator import generate_random_data


def get_data(path, is_for_model):
    if path not in listdir(os.path.abspath("")):
        print("Random Data is generating!!!")
        generate_random_data(start_date, end_date, amount_range, merchant_ids)
    df = pd.read_csv(path)
    if 'day' not in list(df.columns):
        df['Created_Time'] = df['RequestInsertTime'].apply(lambda x: datetime.datetime.strptime(str(x)[0:13], '%Y-%m-%d %H'))
    else:
        df['Created_Time'] = df['day'].apply(lambda x: datetime.datetime.strptime(str(x)[0:13], '%Y-%m-%d'))
    if runs_at_sample_data:
        print("SAMPLE DATA IS RUNNING !!!!!!!")
        random_index = random.sample(list(range(len(df))), sample_size)
        print(len(random_index), len(df))
        data = df.ix[random_index].reset_index()
        print("sample data :", sample_size)
    else:
        data = df
        print("data with row: ", len(df))

    if not is_for_model:
        data = data.rename(columns={'MerchantId': 'merchant_id'})
        data['customer_merchant_id'] = data.apply(lambda row: row['customer_id'] + '_' + str(row['merchant_id']), axis=1)
        data['Created_Time'] = data['RequestInsertTime'].apply(
            lambda x: datetime.datetime.strptime(str(x)[0:13], '%Y-%m-%d %H'))
        data['hour'] = data['Created_Time'].apply(lambda x: x.hour)
        data['Created_Time_str'] = data['Created_Time'].apply(lambda x: str(x)[0:10])
    print("Data Access Done For Model !!!" if is_for_model else "Data Access Done For Feature Engineering !!!")
    return data


def write_to_csv(data, path):
    data.to_csv(path, index=False)


def model_from_to_json(file_path, data, is_writing):
    if is_writing:
        with open(file_path, 'w') as file:
            json.dump(data, file)
    else:
        try:
            with open(file_path, "r") as file:
                data = json.loads(file.read())
        except:
            data = None
        return data