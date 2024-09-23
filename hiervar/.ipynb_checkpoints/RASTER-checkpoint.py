import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import hiervar.miniROCKET as mr
import hiervar.minirocket_multivariate as mrm
import hiervar.raster_multivariate as rsm
import hiervar.classifier as CLF
from hiervar.grsr_module import improved_multi_curve_feature_pruner_exp # experimental version
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV

from sklearn.decomposition import PCA
import hiervar.grsr_module 
    
     
#########
from numba import njit, jit

def Scatter_score(x_train, y_train):
    n_sample,n_feature = x_train.shape[0], x_train.shape[1]
    y_train = np.array(y_train)
    result = []
    for i in range(n_feature):
        ss = misc.scatter_score(x_train[:,i],y_train)
        result.append(ss)
    return np.array(result)
    


def elbow(x,y):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x)
    # Train the classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(x_train_scaled, y)
    # n_targets, n_features = classifier.coef_.shape
    aggregated_coef = np.mean(classifier.coef_,axis=0)
    absoluted_coef = np.abs(aggregated_coef)
    return absoluted_coef

def erocket(x,y):
    sclr = StandardScaler()
    sclr.fit(x)
    X_training_transform_scaled = sclr.transform(x)
    #X_test_transform_scaled = sclr.transform(X_test_transform)
    clf = RidgeClassifierCV(np.logspace(-3,3,10))
    clf.fit(X_training_transform_scaled, y)
    w_ridgecv = clf.coef_
    u_tilde, first_point_non_neg_weight, knees_minus, knees_plus = improved_multi_curve_feature_pruner_exp(w_ridgecv)
    return u_tilde




# @jit(cache=True)
def select_best(x,y, k = 100 , method = 'MI'):
    n_sample , n_feature = x.shape[0], x.shape[1]
    # if method == 'MI':
    #     sorted_index = np.argsort(mutual_info_classif(x,y))[::-1]
    if method == 'SS':
        sorted_index = np.argsort(Scatter_score(x,y))[::-1]
    elif( method == 'random'):
        indexes = np.arange(n_feature)
        sorted_index = np.random.choice(indexes, k, replace=False)
    elif( method == 'elbow'):
        sorted_index = np.argsort(elbow(x,y))[::-1]
    elif( method == 'erocket'):
        best = erocket(x,y)
        return x[:,best],best

    best_index = sorted_index[0:k]
    return x[:,best_index],best_index


def MiniROCKET(x_train,y_train, x_test, y_test,function_type='ter',n_features = 10000,shuffle_quant=False,parameter = None,ali=False):
    # parameter = mr.fit(x_train,num_features =n_features)
    if type(parameter) == type(None):
        if shuffle_quant: 
            parameter = mr.fit_shuffled_quantiles(x_train,num_features =n_features)
        else: 
            parameter = mr.fit(x_train,num_features =n_features)

    if ali == True: 
        parameter = mr.ali_fit(x_train,num_features =n_features)

    dilations, num_features_per_dilation,_, biases = parameter
    parameter = dilations, num_features_per_dilation, biases

    x_train_trans_org = mr.transform(x_train, parameter,function_type)
    x_test_trans_org = mr.transform(x_test, parameter,function_type)
    return x_train_trans_org, x_test_trans_org, parameter




def RASTER(x_train,y_train, x_test, y_test, sizes = 4 ,n_features = 10000, shuffle_quant = False, parameter=None, fixed= False):
    if type(parameter) == type(None):
        if shuffle_quant: 
            parameter = mr.fit_shuffled_quantiles(x_train,num_features =n_features,sizes= sizes)
        else: 
            parameter = mr.fit(x_train,num_features =n_features,sizes= sizes)


    dilation , num_feature_dilation , my_size, biases = parameter
    
    if fixed==True:
        my_size = np.ones(len(my_size))*sizes
        my_size = my_size.astype(np.int64)
        parameter = (dilation , num_feature_dilation , my_size, biases)
    

    x_train_trans_org = mr.transform_refined(x_train, parameter,'ter')
    x_test_trans_org = mr.transform_refined(x_test, parameter,'ter')
    return x_train_trans_org, x_test_trans_org,parameter



def MiniROCKET_MV(x_train,y_train, x_test, y_test,n_features = 10000,shuffle_quant=False,parameter = None):
    # parameter = mr.fit(x_train,num_features =n_features)
    if len(x_train.shape) < 3:
        raise TypeError("it is not multi variate")
        
    # if type(parameter) == type(None):
    #     dilations, num_features_per_dilation,_, biases = parameter
    #     parameter = dilations, num_features_per_dilation, biases
    parameter = mrm.fit(x_train, num_features = n_features)
    x_train_trans_org = mrm.transform(x_train, parameter)
    x_test_trans_org = mrm.transform(x_test, parameter)
    return x_train_trans_org, x_test_trans_org, parameter

def RASTER_MV(x_train,y_train, x_test, y_test,n_features = 10000,shuffle_quant=False,parameter = None):
    # parameter = mr.fit(x_train,num_features =n_features)
    if len(x_train.shape) < 3:
        raise TypeError("it is not multi variate")
        
    # if type(parameter) == type(None):
    #     dilations, num_features_per_dilation,_, biases = parameter
    #     parameter = dilations, num_features_per_dilation, biases
    parameter = rsm.fit(x_train, num_features = n_features)
    x_train_trans_org = rsm.transform(x_train, parameter)
    x_test_trans_org = rsm.transform(x_test, parameter)
    return x_train_trans_org, x_test_trans_org, parameter
 
