from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

from main_airflow import main
from data_access import decide_feature_name, model_from_to_json
from configs import feature_path, learning_model_path



default_args = {
    "owner": 'airflow',
    "start_date": days_ago(1)
}

# running event base
dag = DAG(dag_id='anomaly_detection_framework_fe_train_predict_dashboard',
          description="Simple test6 DAG",
          default_args=default_args,
          schedule_interval='@once')

# dag init bash 1st task
bash_1 = BashOperator(task_id='dag_initialize', bash_command='echo start', dag=dag)

# feature engineering ends
bash_2 = BashOperator(task_id='fe_ends', bash_command='echo start', dag=dag)

# train ends
bash_3 = BashOperator(task_id='train_ends', bash_command='echo start', dag=dag)

task1 = PythonOperator(task_id='fe',
                       op_kwargs={'args': ['main.py', 'feature_engineering', '0', 'all']},
                       python_callable=main,
                       dag=dag
                       )

task2 = PythonOperator(task_id='train',
                       op_kwargs={'args': ['main.py', 'train_process', '0', 'all']},
                       python_callable=main,
                       dag=dag
                       )

task3 = PythonOperator(task_id='prediction',
                       op_kwargs={'args': ['main.py', 'prediction', '0', 'all']},
                       python_callable=main,
                       dag=dag
                       )

task4 = PythonOperator(task_id='dashboard',
                       op_kwargs={'args': ['main.py', 'dashboard', '0']},
                       python_callable=main,
                       dag=dag
                       )

# Feature Engineering
count = 0
for f in decide_feature_name(feature_path):
    _task_1 = PythonOperator(task_id='fe_' + str(count),
                             op_kwargs={'args': ['main.py', 'feature_engineering', '0', f]},
                             python_callable=main,
                             dag=dag
                             )
    bash_1 >> _task_1
    _task_1.set_downstream(bash_2)
    count += 1

# Train Process
count = 0
for m in model_from_to_json(learning_model_path, [], False):
    _task_2 = PythonOperator(task_id='fe_' + str(count),
                             op_kwargs={'args': ['main.py', 'train_process', '0', m]},
                             python_callable=main,
                             dag=dag
                             )
    bash_2 >> _task_2
    _task_2.set_downstream(bash_3)
    count += 1


