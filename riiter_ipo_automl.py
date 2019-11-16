# Ritter IPO Analysis
#   AutoML
# https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py
# pip install --upgrade azureml-sdk
# pip install --upgrade azureml-train-automl
# pip install xgboost

import numpy as np
import pandas as pd
import logging
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment
#from sklearn.model_selection import train_test_split

ws = Workspace.from_config()

df = pd.read_csv('IPO2609FeatureEngineering.csv')

#y_df = df.pop("underpriced")
#x_df = df
#x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

automl_classifier = AutoMLConfig(
    task='classification',
    debug_log='automl_debug.log',
    primary_metric='AUC_weighted',
    featurization= 'auto',
    experiment_timeout_minutes=30,
    training_data=df,
    label_column_name='underpriced',
    n_cross_validations=5
    )

experiment = Experiment(ws, "ritter-experiment")
local_run = experiment.submit(automl_classifier, show_output=True)