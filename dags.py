from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

from main_airflow import main



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
task_1 = PythonOperator(task_id='feature_engineering',
                        op_kwargs={'args': ['main.py', 'feature_engineering', 'all']},
                        python_callable=main,
                        dag=dag
                        )
bash_1 >> task_1 >> bash_2

# Train Process
count = 0
for m in ['iso_f', 'e_iso_f', 'ae']:
    _task_2 = PythonOperator(task_id='fe_' + str(count),
                             op_kwargs={'args': ['main.py', 'train_process', '0', m]},
                             python_callable=main,
                             dag=dag
                             )
    bash_2 >> _task_2
    _task_2.set_downstream(bash_3)
    count += 1

bash_3 >> task3 >> task4
