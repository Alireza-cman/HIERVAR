# General random sampling and resampling module
import numpy as np
from kneed import KneeLocator
from sklearn.linear_model import RidgeClassifierCV,RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def improved_multi_curve_feature_pruner_exp(W):
    '''
    inpute: weight matrix W for C or C-1 hyperplanes
            W is C X F, where F is the number of features
            and C is the number of classes.
            This is an experimental version with more outputs
    output: 
    unique feature indices maintain_ftr_indices_unique
    
    This implementation is based on the E-ROCKET paper
    '''
    Sensitivity = 2 # Sensitivity of kneedle detection
    poly_deg = 3 # Polynomial degree of spline smoother.
    n_targ = W.shape[0]
    num_features = W.shape[1]
    argsorted_W = np.zeros((n_targ, num_features), dtype = int)
    split_points = np.zeros(n_targ, dtype = int)
    for i in range(n_targ):
        Wi = W[i,:] 
        argsorted_W[i,:] = np.argsort(Wi)
        sorted_curve = Wi[argsorted_W[i,:]]
        posit_points = np.where(sorted_curve >=0)[0] # Positive points in the sorted curve
        split_points[i] = posit_points[0]
        # positive_ftr_indices = argsorted_W_0[split_point:] # positive feature indices
        # negative_ftr_indices = argsorted_W_0[:split_point] # negative feature indices

    knees_plus = np.zeros((n_targ,))

    for i in range(n_targ):
        posit_len = len(W[i,:]) - split_points[i]
        k1 = KneeLocator(range(posit_len), 
                        W[i, argsorted_W[i, split_points[i]:].astype(int)], 
                        curve="convex", direction="increasing",
                        S = Sensitivity,
                        polynomial_degree = poly_deg)
        knee_loc = k1.knee
        knees_plus[i] = knee_loc
                
    # find knee on negative side
    knees_minus = np.zeros((n_targ,))
    for i in range(n_targ):
        negat_len = split_points[i]
        k2 = KneeLocator(range(negat_len), 
                        W[i,argsorted_W[i, : negat_len].astype(int)], 
                        curve="concave", direction="increasing",
                        S = Sensitivity,
                        polynomial_degree = poly_deg)
        knee_loc2 = k2.knee
        knees_minus[i] = knee_loc2
                
    maintain_ftr_indices = []
            
    for i in range(n_targ):
        maintain_ftr_indices = maintain_ftr_indices +\
        argsorted_W[i, split_points[i] + knees_plus[i].astype(int):].tolist() +\
        argsorted_W[i,: knees_minus[i].astype(int)].tolist()
                
    maintain_ftr_indices_unique = np.unique(maintain_ftr_indices) # union of indices
    return maintain_ftr_indices_unique, split_points, knees_minus, knees_plus


def improved_multi_curve_feature_pruner(W):
    '''
    inpute: weight matrix W for C or C-1 hyperplanes
            W is C X F, where F is the number of features
            and C is the number of classes.
    output: 
    unique feature indices maintain_ftr_indices_unique
    
    This implementation is based on the E-ROCKET paper
    '''
    Sensitivity = 2 # Sensitivity of kneedle detection
    poly_deg = 3 # Polynomial degree of spline smoother.
    n_targ = W.shape[0]
    num_features = W.shape[1]
    argsorted_W = np.zeros((n_targ, num_features), dtype = int)
    split_points = np.zeros(n_targ, dtype = int)
    for i in range(n_targ):
        Wi = W[i,:] 
        argsorted_W[i,:] = np.argsort(Wi)
        sorted_curve = Wi[argsorted_W[i,:]]
        posit_points = np.where(sorted_curve >=0)[0] # Positive points in the sorted curve
        split_points[i] = posit_points[0]
        # positive_ftr_indices = argsorted_W_0[split_point:] # positive feature indices
        # negative_ftr_indices = argsorted_W_0[:split_point] # negative feature indices

    knees_plus = np.zeros((n_targ,))

    for i in range(n_targ):
        posit_len = len(W[i,:]) - split_points[i]
        k1 = KneeLocator(range(posit_len), 
                        W[i, argsorted_W[i, split_points[i]:].astype(int)], 
                        curve="convex", direction="increasing",
                        S = Sensitivity,
                        polynomial_degree = poly_deg)
        knee_loc = k1.knee
        knees_plus[i] = knee_loc
                
    # find knee on negative side
    knees_minus = np.zeros((n_targ,))
    for i in range(n_targ):
        negat_len = split_points[i]
        k2 = KneeLocator(range(negat_len), 
                        W[i,argsorted_W[i, : negat_len].astype(int)], 
                        curve="concave", direction="increasing",
                        S = Sensitivity,
                        polynomial_degree = poly_deg)
        knee_loc2 = k2.knee
        knees_minus[i] = knee_loc2
                
    maintain_ftr_indices = []
            
    for i in range(n_targ):
        maintain_ftr_indices = maintain_ftr_indices +\
        argsorted_W[i, split_points[i] + knees_plus[i].astype(int):].tolist() +\
        argsorted_W[i,: knees_minus[i].astype(int)].tolist()
                
    maintain_ftr_indices_unique = np.unique(maintain_ftr_indices) # union of indices
    return maintain_ftr_indices_unique

def multi_curve_abs_feature_pruner(W):
    '''
    inpute: weight matrix W for C or C-1 hyperplanes
            W is C X F, where F is the number of features
            and C is the number of classes.
    output: 
    unique feature indices maintain_ftr_indices_unique
    obtained from sorting based on the absolute value of the coefficients.
    '''
    n_targ = W.shape[0]
    num_features = W.shape[1]
    argsorted_W = np.zeros((n_targ, num_features))
    for i in range(n_targ):
        Wi = np.abs(W[i,:]) 
        argsorted_W[i,:] = np.argsort(Wi)

    knees_plus = np.zeros((n_targ,))
    posit_len = num_features

    for i in range(n_targ):
        k1 = KneeLocator(range(posit_len), 
                        W[i,argsorted_W[i,:].astype(int)], 
                        curve="convex", direction="increasing")
        knee_loc = k1.knee
        knees_plus[i] = knee_loc
                
    
                
    maintain_ftr_indices = []
            
    for i in range(n_targ):
        maintain_ftr_indices = maintain_ftr_indices +\
        argsorted_W[i, knees_plus[i].astype(int):].tolist()
                
    maintain_ftr_indices_unique = np.unique(maintain_ftr_indices) # union of indices
    return maintain_ftr_indices_unique

def multi_curve_feature_pruner(W):
    '''
    inpute: weight matrix W for C or C-1 hyperplanes
            W is C X F, where F is the number of features
            and C is the number of classes.
    output: 
    unique feature indices maintain_ftr_indices_unique
    '''
    Sensitivity = 2 # Sensitivity of kneedle detection
    poly_deg = 3 # Polynomial degree of spline smoother.
    n_targ = W.shape[0]
    num_features = W.shape[1]
    argsorted_W = np.zeros((n_targ, num_features))
    for i in range(n_targ):
        Wi = W[i,:] 
        argsorted_W[i,:] = np.argsort(Wi)

    knees_plus = np.zeros((n_targ,))
    posit_len = num_features - num_features // 2

    for i in range(n_targ):
        k1 = KneeLocator(range(posit_len), 
                        W[i,argsorted_W[i,num_features // 2:].astype(int)], 
                        curve="convex", direction="increasing",
                        S = Sensitivity,
                        polynomial_degree = poly_deg)
        knee_loc = k1.knee
        knees_plus[i] = knee_loc
                
    # find knee on negative side
    knees_minus = np.zeros((n_targ,))
    negat_len = num_features // 2
    for i in range(n_targ):
        k2 = KneeLocator(range(negat_len), 
                        W[i,argsorted_W[i, : num_features // 2].astype(int)], 
                        curve="concave", direction="increasing",
                        S = Sensitivity,
                        polynomial_degree = poly_deg)
        knee_loc2 = k2.knee
        knees_minus[i] = knee_loc2
                
    maintain_ftr_indices = []
            
    for i in range(n_targ):
        maintain_ftr_indices = maintain_ftr_indices +\
        argsorted_W[i, num_features // 2 + knees_plus[i].astype(int):].tolist() +\
        argsorted_W[i,: knees_minus[i].astype(int)].tolist()
                
    maintain_ftr_indices_unique = np.unique(maintain_ftr_indices) # union of indices
    return maintain_ftr_indices_unique
    
def single_curve_feature_pruner(W):
    '''
    inpute: weight vector W for 1 hyperplane
            W is  F X 1, where F is the number of features
    output: 
    unique feature indices maintain_ftr_indices_unique
    '''
    num_features = W.shape[0]
    argsorted_W = np.argsort(W)

    posit_len = num_features - num_features // 2

    k1 = KneeLocator(range(posit_len), 
                    W[argsorted_W[num_features // 2:].astype(int)], 
                    curve="convex", direction="increasing")
    knee_loc = k1.knee
    knees_plus = knee_loc
                
    # find knee on negative side
    negat_len = num_features // 2
    
    k2 = KneeLocator(range(negat_len), 
                    W[argsorted_W[: num_features // 2].astype(int)], 
                    curve="concave", direction="increasing")
    knee_loc2 = k2.knee
    knees_minus = knee_loc2
                
    maintain_ftr_indices = []
            
    maintain_ftr_indices = maintain_ftr_indices +\
    argsorted_W[num_features // 2 + knees_plus.astype(int):].tolist() +\
    argsorted_W[: knees_minus.astype(int)].tolist()
                
    maintain_ftr_indices_unique = np.unique(maintain_ftr_indices)
    return maintain_ftr_indices_unique

def grsr_ftr_selctr_xoutlr(X_training_transform_H, Y_training):
    scaler = StandardScaler(with_mean=True) 
    scaler.fit(X_training_transform_H) 
    X_training_transform_H_scaled = scaler.transform(X_training_transform_H)
    reg_H_trn = RidgeCV(alphas = np.logspace(-3, 3, 10))
    
    reg_H_trn.fit(X_training_transform_H_scaled, Y_training)
   
    # save regularization constant
    #alpha_reg_trn = reg_H_trn.alpha_
    #print('Alpha regularization in training set: ', alpha_reg_trn)
    W_trn = reg_H_trn.coef_ # weight matrix for regressor
    
    print('W_trn.shape',W_trn.shape)
    w_trn_max = W_trn.max()
    w_trn_min = W_trn.min()
    print('maximum w traning: ', w_trn_max)
    print('minimum w traning: ', w_trn_min)

    maintain_ftr_indices_unique_trn = single_curve_feature_pruner(W_trn)
        
        
    maintain_ftr_indices_unique = maintain_ftr_indices_unique_trn
    print('Number of maintained features : ',
              maintain_ftr_indices_unique.shape[0])
    
    return maintain_ftr_indices_unique


def grsr_ftr_selctr_olsr(X_training_transform_H, Y_training):
    '''
    Feature selection with ordinary least square method.
    '''
    scaler = StandardScaler(with_mean=True) 
    scaler.fit(X_training_transform_H) 
    X_training_transform_H_scaled = scaler.transform(X_training_transform_H)
    reg_H_trn = LinearRegression().fit(X_training_transform_H_scaled, 
                                       Y_training)
    reg_H_trn.fit(X_training_transform_H_scaled, Y_training)
   
    # save regularization constant
    #alpha_reg_trn = reg_H_trn.alpha_
    #print('Alpha regularization in training set: ', alpha_reg_trn)
    W_trn = reg_H_trn.coef_ # weight matrix for regressor
    
    print('W_trn.shape',W_trn.shape)
    w_trn_max = W_trn.max()
    w_trn_min = W_trn.min()
    print('maximum w traning: ', w_trn_max)
    print('minimum w traning: ', w_trn_min)

    maintain_ftr_indices_unique_trn = single_curve_feature_pruner(W_trn)
        
        
    maintain_ftr_indices_unique = maintain_ftr_indices_unique_trn
    print('Number of maintained features : ',
              maintain_ftr_indices_unique.shape[0])
    
    return maintain_ftr_indices_unique

def gen_kernel_H_samples_w(maintain_ftr_indices_unique, kernels_H):
    '''
    Only kernels' weights are considered.
    For grev7w9 features
    '''
    weights, _, _, _, _,_ = kernels_H
    ftr_indices = maintain_ftr_indices_unique.astype(int)
    num_samples = len(ftr_indices)
    Sample_kernels = np.zeros((num_samples, 9))
    for count, idx in enumerate(ftr_indices):
        #print('idx: ',idx)
        Sample_kernels[count, :] = weights[9*idx : 9*idx + 9]
    return Sample_kernels

def gen_kernel_samples_w(maintain_ftr_indices_unique, kernels_H):
    '''
    Only kernels' weights are considered.
    '''
    weights, lengths, biases, dilations, paddings = kernels_H
    ftr_indices = maintain_ftr_indices_unique.astype(int)
    num_samples = len(ftr_indices)
    Sample_kernels = np.zeros((num_samples, 9))
    for count, idx in enumerate(ftr_indices):
        #print('idx: ',idx)
        Sample_kernels[count, :] = weights[9*idx : 9*idx + 9]
    return Sample_kernels
    
def gen_kernel_samples(maintain_ftr_indices_unique, kernels_H):
    weights, lengths, biases, dilations, paddings = kernels_H
    ftr_indices = maintain_ftr_indices_unique.astype(int)
    num_samples = len(ftr_indices)
    Sample_kernels = np.zeros((num_samples, 11))
    for count, idx in enumerate(ftr_indices):
        #print('idx: ',idx)
        Sample_kernels[count, 0:9] = weights[9*idx : 9*idx + 9]
        Sample_kernels[count, 9] = biases[idx]
        Sample_kernels[count, 10] = dilations[idx]
    return Sample_kernels

def gen_full_kernel(Resampled_kernels, input_length):
    simple_kernels = Resampled_kernels[0]
    num_kernels = len(simple_kernels)
    candidate_lengths = np.array((9,), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(num_kernels * 9, dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)
    for i in range(num_kernels):
        weights[i * 9 : i * 9 + 9] = simple_kernels[i, 0:9]
        biases[i] = simple_kernels[i, 9]
        _length = lengths[i]
        max_dilate = 2 ** np.log2((input_length - 1) / (_length - 1))
        dilation = np.int32(simple_kernels[i, 10] if simple_kernels[i, 10] < max_dilate else max_dilate - 1)
        dilations[i] = dilation
        
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
    return  weights, lengths, biases, dilations, paddings

def gen_full_kernel_H_w(Resampled_kernels, input_length):
    '''
    For KDE method, only kernels weights were sampled
    This is designed for gre features
    '''
    
    num_kernels = Resampled_kernels.shape[0]
    candidate_lengths = np.array((9,), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(num_kernels * 9, dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)
    H = []
    for i in range(num_kernels):
        
        weights[i * 9 : i * 9 + 9] = Resampled_kernels[i, 0:9]
        biases[i] =  np.random.uniform(-1, 1)
        _length = lengths[i]
        max_dilate = 2 ** np.log2((input_length - 1) / (_length - 1))
        dilation = np.int32(2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1))))
        dilations[i] = dilation
        
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
        
        len_h = (input_length + 2 * padding) - ((_length - 1) * dilation)
        h = np.random.choice(np.array((3,5)), len_h)
        H.append(h)
        
    return  weights, lengths, biases, dilations, paddings, H

def gen_full_kernel_w(Resampled_kernels, input_length):
    '''
    For KDE method, only kernels weights were sampled
    '''
    
    num_kernels = Resampled_kernels.shape[0]
    candidate_lengths = np.array((9,), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(num_kernels * 9, dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)
    for i in range(num_kernels):
        
        weights[i * 9 : i * 9 + 9] = Resampled_kernels[i, 0:9]
        biases[i] =  np.random.uniform(-1, 1)
        _length = lengths[i]
        max_dilate = 2 ** np.log2((input_length - 1) / (_length - 1))
        dilation = np.int32(2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1))))
        dilations[i] = dilation
        
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
    return  weights, lengths, biases, dilations, paddings

def gen_full_kernel2(Resampled_kernels, input_length):
    '''
    For KDE method
    '''
    
    num_kernels = Resampled_kernels.shape[0]
    candidate_lengths = np.array((9,), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(num_kernels * 9, dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)
    for i in range(num_kernels):
        
        weights[i * 9 : i * 9 + 9] = Resampled_kernels[i, 0:9]
        biases[i] = Resampled_kernels[i, 9]
        _length = lengths[i]
        max_dilate = 2 ** np.log2((input_length - 1) / (_length - 1))
        dilation = np.int32(Resampled_kernels[i, 10] if Resampled_kernels[i, 10] < max_dilate else max_dilate - 1)
        dilations[i] = dilation
        
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
    return  weights, lengths, biases, dilations, paddings
    
def gen_full_kernel_H(Resampled_kernels, input_length):
    '''
    For KDE method and GRE features
    '''
    
    num_kernels = Resampled_kernels.shape[0]
    candidate_lengths = np.array((9,), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(num_kernels * 9, dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)
    H = []
    for i in range(num_kernels):
        
        weights[i * 9 : i * 9 + 9] = Resampled_kernels[i, 0:9]
        biases[i] = Resampled_kernels[i, 9]
        _length = lengths[i]
        max_dilate = 2 ** np.log2((input_length - 1) / (_length - 1))
        dilation = np.int32(Resampled_kernels[i, 10] if Resampled_kernels[i, 10] < max_dilate else max_dilate - 1)
        dilations[i] = dilation
        
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
        len_h = (input_length + 2 * padding) - ((_length - 1) * dilation)
        h = np.random.choice(np.array((3,5)), len_h)
        H.append(h)

    return  weights, lengths, biases, dilations, paddings, H

def EEE_rank(InputData):
    Len  = len(InputData)
    Hhat=np.zeros((Len,1))
    for i in range(0,Len):
        xhat = InputData[i:]
        Hhat[i] = Entropy(xhat)
    DifHhat = [x - Hhat[i - 1] for i, x in enumerate(Hhat)][1:]
    return np.argmin(DifHhat)


def Entropy(InputData):
    eps = 2 * 10 ** -16
    d=1 #input data dimension
    N= len(InputData)
    if N==1:
        Sigma = 1.06*InputData[0] + eps
    else:
        #Sigma = 1.06*(np.var(InputData))/(N**5) + eps
        Sigma = np.std(InputData) *(4/N/(2*d+1))**(1/(d+4)) + eps #silverman
    Hhat = np.zeros((N,1))
    for j in range(N):
        xj = InputData[j]
        Kernel = np.zeros((N,1))
        for i in range(N):
            xi = InputData[i]
            Kernel[i] = 1/(np.sqrt(2*np.pi)*Sigma)*np.exp(-((xj-xi)**2)/(2*Sigma**2))
        Hhat[j]=np.log10((1/N)*np.sum(Kernel))
    return -(1/N)*np.sum(Hhat)