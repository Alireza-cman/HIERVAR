import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif as sklearn_f_classif
import hiervar.RASTER as RASTER
import matplotlib.pyplot as plt
import hiervar.miniROCKET as mr




def anova_erocket_pruner(x_train, y_train,threshold= 1.0,verbose = True, divider= 10):
    """
        INPUT: 
            x_train, 
            y_train, 
            threshold: default on 1, but if you set it on None, it will use divider to use the erocket mean anova score divide by divider. 
            verbose, 
            divider
        OUTPUT: 
            result , 
            erocket_index , 
            anova_raster[erocket_index].mean()
    """
    _, erocket_index = RASTER.select_best(x_train,y_train, k = 200 , method='erocket')
    _, anova_raster = anova_feature_selection(x_train,y_train)
    result = []
    
    if type(threshold) == type(None):
        threshold = anova_raster[erocket_index].mean()/divider
    # print(len(erocket_index))
    
    for i in erocket_index:
        # print(i,anova_raster[i])
        if anova_raster[i] > threshold:
            result.append(i)
    if verbose:
        print(len(erocket_index) , '-->', len(result), 'where the mean of selected erocket anova is: ', anova_raster[erocket_index].mean())
        print(threshold)
    if len(result) < 2: 
        result = erocket_index
        print("ANOVA faced with all zero score")
    return result , erocket_index , anova_raster[erocket_index].mean()

def anova_feature_selection(x_train,y_train, K = 200):
    """
     INPUT: 
        x_train, 
        y_train, 
        K
    Output: 
        selected_feature, 
        Anova: Score of all features

    """
    scaler = StandardScaler(with_mean=True) 
    scaler.fit(x_train) 
    x_train_trans_raster = scaler.transform(x_train)
    # x_test_trans_raster  = scaler.transform(x_test_trans_raster)
    anova, pvalue = sklearn_f_classif(x_train, y_train)
    anova = np.nan_to_num(anova,nan=0.0, posinf=0, neginf=0)
    pvalue = np.nan_to_num(pvalue,nan=0.0, posinf=0, neginf=0)
    anova[anova<0 ]= 0
    pvalue[pvalue<0 ]= 0
    selected_features = np.argsort(anova)[::-1][:K]
    return selected_features, anova





# Function to calculate the dilation and kernel index based on a global index
def find_dilation_and_kernel(global_index,parameters):
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)
    
    total_kernels = 84
    if len(parameters) == 4:
        dilations , num_features_per_dilation, sizes , biases = parameters
    else :
        dilations , num_features_per_dilation , biases = parameters
        sizes = np.ones_like(biases).astype(np.int32)
    cumulative_features = np.cumsum(num_features_per_dilation * total_kernels)

    # Find the dilation index where the global index would belong
    dilation_index = np.searchsorted(cumulative_features, global_index, side='right')
    
    if dilation_index == 0:
        previous_cumulative = 0
    else:
        previous_cumulative = cumulative_features[dilation_index - 1]
    
    # Find the local index within the dilation group
    local_index = global_index - previous_cumulative
    cumulative_features[dilation_index]//num_features_per_dilation[dilation_index]
    # Find which kernel it belongs to and the feature index within that kernel
    kernel_index = local_index // num_features_per_dilation[dilation_index]
    feature_within_kernel = local_index % num_features_per_dilation[dilation_index]
    return {
        'dilation': dilations[dilation_index],
        'kernel_index': kernel_index,
        'feature_within_kernel': feature_within_kernel,
        'size' : sizes[global_index],
        'biases': biases[global_index:global_index+sizes[global_index]]
        
    }


