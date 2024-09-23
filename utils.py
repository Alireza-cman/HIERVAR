
import random as rd
import numpy as np
import pandas as pd 

from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from os import walk
import matplotlib.pyplot as plt
import seaborn as sns


from csv import writer



def load_dataset(dataset_name,verbose= True):
    directory ='./Dataset/'
    archive_files = []
    for (dirpath, dirnames, filenames) in walk(directory):
        archive_files.extend(dirnames)
    files_name = list(filter(lambda x: '_' not in x, archive_files))   
    for dataset in files_name: 
        if dataset not in [dataset_name]:
            continue
        if verbose == True: 
            print(dataset, ": ",end='\t')
        TRAIN = directory + dataset +'/' + dataset +'_TRAIN.tsv' 
        TEST = directory + dataset +'/' + dataset +'_TEST.tsv' 

        X_train_pandas = pd.read_csv(TRAIN, sep='\t',header=None)
        X_test_pandas = pd.read_csv(TEST, sep='\t',header=None)
        #
        X_train = X_train_pandas.drop([0],axis=1).fillna(0).to_numpy().astype(np.float64)
        X_test = X_test_pandas.drop([0],axis=1).fillna(0).to_numpy().astype(np.float64)
    #         ########=============
        y_train = X_train_pandas[0].tolist()
        y_test = X_test_pandas[0].tolist()
        if verbose == True: 
            print(X_train.shape , X_test.shape)
            print('#labels: ',np.unique(y_train))
    return X_train, y_train , X_test, y_test

def load_dataset_multivariate(dataset_name,verbose= True):
    directory ='./Dataset/Multivariate_ts/Dataset/'
    archive_files = []
    for (dirpath, dirnames, filenames) in walk(directory):
        archive_files.extend(dirnames)
    files_name = list(filter(lambda x: '_' not in x, archive_files))   
    for dataset in files_name: 
        if dataset not in [dataset_name]:
            continue
        if verbose == True: 
            print(dataset, ": ",end='\t')
        TRAIN = directory + dataset +'/' + dataset +'_TRAIN.ts' 
        TEST = directory + dataset +'/' + dataset +'_TEST.ts' 

        # dataset_name = 'SelfRegulationSCP2'
        X_train, y_train =load_from_tsfile_to_dataframe(TRAIN)
        X_test, y_test =load_from_tsfile_to_dataframe(TEST)
        X_train = from_nested_to_3d_numpy(X_train).transpose(0,2,1)
        X_test = from_nested_to_3d_numpy(X_test).transpose(0,2,1)
        if verbose == True: 
            print(X_train.shape , X_test.shape)
            print('#labels: ',np.unique(y_train))
    return X_train, y_train , X_test, y_test

