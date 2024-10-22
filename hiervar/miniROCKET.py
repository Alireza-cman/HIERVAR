# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification

# https://arxiv.org/abs/2012.08791

from numba import njit, prange, vectorize
from numba.types import unicode_type
import numpy as np

@njit("float32[:](float32[:,:],int32[:],int32[:],float32[:])", fastmath = True, parallel = False, cache = True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles):

    num_examples, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
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

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype = np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(num_examples)]

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[feature_index_start:feature_index_end]) 
            2.2,6.1,6.7
            
            
           
            feature_index_start = feature_index_end

    return biases

def _fit_dilations(input_length, num_features, max_dilations_per_kernel):

    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
    np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32) # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation

# def ali_fit_dilations(input_length):

#     max_exponent = np.log2((input_length - 1) / (9 - 1))
#     dilations, num_features_per_dilation = \
#     np.unique(np.logspace(0, max_exponent, 5, base = 2).astype(np.int32), return_counts = True)
#     dilations = 1

#     return dilations

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype = np.float32)

def fit(X, num_features = 10_000, max_dilations_per_kernel = 32,sizes= 4):

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)
    
    random_size = np.random.randint(1,sizes+1,len(biases))

    return dilations, num_features_per_dilation, random_size ,biases 


def fit_shuffled_quantiles(X, num_features = 10_000, max_dilations_per_kernel = 32,sizes= 4):

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)
    
    np.random.shuffle(quantiles)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)
    
    random_size = np.random.randint(1,sizes+1,len(biases))

    return dilations, num_features_per_dilation, random_size ,biases 

def ali_fit(X, num_features = 10_000, max_dilations_per_kernel = 32,sizes= 4):

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)
    multiplier = (num_features//num_kernels)//len(dilations)
    num_features_per_dilation = np.ones(len(dilations),dtype=np.int32)*multiplier

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)
    
    random_size = np.random.randint(1,sizes+1,len(biases))

    return dilations, num_features_per_dilation, random_size ,biases 

def ProbabilisticDilation_fit(X, num_features = 10_000, max_dilations_per_kernel = 32,sizes= 4,index = 0):

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_dilation = np.ones(len(dilations),dtype=np.int32)

    dilations = np.array([dilations[index]]).astype(np.int32)
    num_features_per_dilation = np.array([num_features_per_dilation[index]]).astype(np.int32)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)
    
    random_size = np.random.randint(1,sizes+1,len(biases))

    return dilations, num_features_per_dilation, random_size ,biases 


def impared_fit(X, parameter,sizes= 4):
    _, input_length = X.shape
    num_kernels = 84
    dilations, num_features_per_dilation = parameter
    num_features_per_kernel = np.sum(num_features_per_dilation)
    quantiles = _quantiles(num_kernels * num_features_per_kernel)
    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)
    random_size = np.random.randint(1,sizes+1,len(biases))
    return dilations, num_features_per_dilation, random_size ,biases 


# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
# @vectorize("float32(float32,float32)", nopython = True, cache = True)
@vectorize(nopython = True, cache = True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0

# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
# @vectorize("float32(float32,float32)", nopython = True, cache = True)
@vectorize(nopython = True, cache = True)
def _ReLU(a, b):
    if a > b:
        return a
    else:
        return 0
@vectorize(nopython = True, cache = True)
def _Sigmoid(a,b):
    return 1 / (1 + np.exp(-(a-b)))

@vectorize(nopython = True, cache = True)
def _Cos(a,b):
    if a > b:
        return np.cos(a)
    else:
        return 0

@vectorize(nopython = True, cache = True)
def _Pass(a,b):
    return a

@vectorize(nopython = True, cache = True)
def _Linear(a,b):
    return a - b

@vectorize(nopython = True, cache = True)
def _LeakyRelu(a,b):
    if a > b:
        return a
    else:
        return 0.3*a


    

@njit()
def create_rter(chunk , length = 5120):
    curve = np.zeros(length)
    # chunk = np.zeros(3)
    # print('alireza', curve)
    chunk_size = length//len(chunk)
    for c in range(len(chunk)):
        value = chunk[c]
        # print(c,value,chunk_size)
        if c == len(chunk) - 1 :
            curve[c*chunk_size:] = value 
        curve[c*chunk_size: (c+1)*chunk_size] = value
        # plt.plot(curve.squeeze())
    # curve = curve.squeeze()
    return curve
    
@njit()
def create_curve(chunk , length = 5120):
    # curve = np.zeros(length)
    chunk_size = length//len(chunk)
    t = np.linspace(0, length, length)
    for c in range(len(chunk)):
        period = length/(c+1)
        if c == 0: 
            y = chunk[c] * np.sin(2 * np.pi * t / period)
        else:
            y += chunk[c] * np.sin(2 * np.pi * t / period)
# y = A_A * np.sin(2 * np.pi * t / A_T) + B_A * np.sin(2 * np.pi * t / B_T) 

    return y


@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],float32[:])), unicode_type)", fastmath = True, parallel = True, cache = True)
def transform(X, parameters,function_name='ter'):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
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

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        if function_name == 'ter':
                            features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'relu':
                            features[example_index, feature_index_start + feature_count] = _ReLU(C, biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'cos':
                            features[example_index, feature_index_start + feature_count] = _Cos(C, biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'sigmoid':
                            features[example_index, feature_index_start + feature_count] = _Sigmoid(C, biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'pass':
                            features[example_index, feature_index_start + feature_count] = _Pass(C, biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'linear':
                            features[example_index, feature_index_start + feature_count] = _Linear(C, biases[feature_index_start + feature_count]).mean()   
                        elif function_name == 'leakyrelu':
                            features[example_index, feature_index_start + feature_count] = _LeakyRelu(C, biases[feature_index_start + feature_count]).mean()  
                        elif function_name == 'max':                        
                            features[example_index, feature_index_start + feature_count] = np.max(C)  
                        else:
                            features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()


                            
                else:
                    for feature_count in range(num_features_this_dilation):
                        if function_name == 'ter':
                             features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'relu':
                            features[example_index, feature_index_start + feature_count] = _ReLU(C[padding:-padding], biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'cos':
                            features[example_index, feature_index_start + feature_count] = _Cos(C[padding:-padding], biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'sigmoid':
                            features[example_index, feature_index_start + feature_count] = _Sigmoid(C[padding:-padding], biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'pass':
                            features[example_index, feature_index_start + feature_count] = _Pass(C[padding:-padding], biases[feature_index_start + feature_count]).mean()
                        elif function_name == 'linear':
                            features[example_index, feature_index_start + feature_count] = _Linear(C[padding:-padding], biases[feature_index_start + feature_count]).mean() 
                        elif function_name == 'leakyrelu':
                            features[example_index, feature_index_start + feature_count] = _LeakyRelu(C[padding:-padding], biases[feature_index_start + feature_count]).mean() 
                        else:
                            features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()
                        

                feature_index_start = feature_index_end

    return features 
    



@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],int64[:],float32[:])), unicode_type) ", fastmath = True, parallel = True, cache = True)
def transform_rter(X, parameters,function_name = 'relu'):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, random_size ,biases = parameters
    # random_size = np.random.randint(1,4,500)
    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    if function_name == 'relu':
        function = _ReLU
    elif function_name == 'ter':
        function = _PPV
    else: 
        function = _PPV
        
        
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

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    
    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):
        i = 0 
        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):
            i+= 1
            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):
                
                feature_index_end = feature_index_start + num_features_this_dilation
                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        ##
                        start = feature_index_start + feature_count
                        c_size = random_size[start]
                        end = start + c_size
                        beh = biases[start:end]
                        curve = create_rter(chunk = beh,length=len(C))
                        if function_name == 'relu':
                            features[example_index, feature_index_start + feature_count] = _ReLU(C,curve).mean()                            
                        elif function_name == 'ter':
                            features[example_index, feature_index_start + feature_count] = _PPV(C,curve).mean()
                            
                        elif function_name == 'cos':
                            features[example_index, feature_index_start + feature_count] = _Cos(C,curve).mean()
                        elif function_name == 'sigmoid':
                            features[example_index, feature_index_start + feature_count] = _Sigmoid(C,curve).mean()
                        elif function_name == 'pass':
                            features[example_index, feature_index_start + feature_count] = _Pass(C,curve).mean()
                        elif function_name == 'linear':
                            features[example_index, feature_index_start + feature_count] = _Linear(C,curve).mean()
                        elif function_name == 'leakyrelu':
                            features[example_index, feature_index_start + feature_count] = _LeakyRelu(C,curve).mean()
                        else:
                            features[example_index, feature_index_start + feature_count] = _PPV(C,curve).mean()
                        
                else:
                    for feature_count in range(num_features_this_dilation):
                        start = feature_index_start + feature_count
                        c_size = random_size[start]
                        end = start + c_size
                        beh = biases[start:end]
                        curve = create_rter(beh,length=len(C))
                        if function_name == 'relu':
                            features[example_index, feature_index_start + feature_count] = _ReLU(C[padding:-padding], curve[padding:-padding]).mean()
                        elif function_name == 'ter':
                            features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], curve[padding:-padding]).mean()
                        elif function_name == 'cos':
                            features[example_index, feature_index_start + feature_count] = _Cos(C[padding:-padding], curve[padding:-padding]).mean()
                        elif function_name == 'sigmoid':
                            features[example_index, feature_index_start + feature_count] = _Sigmoid(C[padding:-padding], curve[padding:-padding]).mean()
                        elif function_name == 'pass':
                            features[example_index, feature_index_start + feature_count] = _Pass(C[padding:-padding], curve[padding:-padding]).mean()
                        elif function_name == 'linear':
                            features[example_index, feature_index_start + feature_count] = _Linear(C[padding:-padding], curve[padding:-padding]).mean()
                        elif function_name == 'leakyrelu':
                            features[example_index, feature_index_start + feature_count] = _LeakyRelu(C[padding:-padding], curve[padding:-padding]).mean()
                        else: 
                            features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], curve[padding:-padding]).mean()

                feature_index_start = feature_index_end

    return features





from numba.types import unicode_type
@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],int64[:],float32[:])), unicode_type) ")
def transform_tar(X, parameters,function_name = 'ter'):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, random_size ,biases = parameters
    # print(dilations,num_features_per_dilation)
    if function_name == 'relu':
        function = _ReLU
    elif function_name == 'ter':
        function = _PPV
    else: 
        function = _PPV
        
        
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

    num_kernels = len(indices)
    num_dilations = len(dilations)


    # num_features = num_kernels * np.sum(num_features_per_dilation) * np.sum(random_size)
    
    num_features = np.sum(random_size)
    features = np.zeros((num_examples, num_features), dtype = np.float32)
    
    
    for example_index in prange(num_examples):
        i = 0 
        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0
        counter = 0 
        for dilation_index in range(num_dilations):
            i+= 1
            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):
                
                feature_index_end = feature_index_start + 1
                # _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                

                ##
                start = feature_index_start 
                c_size = random_size[start]
                end = start + c_size
                beh = biases[start:end]
                segments = len(beh)
                M = len(C)//segments
                # M = len(C)
                # segments = 1
                for portion in range(segments):
                    
                    C_tmp = C[portion*M: (portion+1)*M]
                    
                    # if portion == len(beh)-1 : 
                    #     C_tmp = C[portion*M: ]
                    C_segment = len(C_tmp)
                    beh_tmp = beh[portion]
                    beh_tmp = np.array([beh_tmp])
                    
                    curve = create_rter(chunk = beh_tmp,length=len(C_tmp))
                    if _padding0 == 0 :
                        
                        if function_name == 'relu':
                            features[example_index,counter] = _ReLU(C_tmp,curve).mean()                            
                        elif function_name == 'ter':
                            features[example_index,counter] = _PPV(C_tmp,curve).mean()
                        counter = counter + 1 
                    else: 
                        if function_name == 'relu':
                            features[example_index,counter] = _ReLU(C_tmp[padding:-padding],curve[padding:-padding]).mean()                            
                        elif function_name == 'ter':
                            features[example_index,counter] = _PPV(C_tmp[padding:-padding],curve[padding:-padding]).mean()
                        counter = counter + 1 

                    #print(feature_index_start,counter)



                feature_index_start = feature_index_end

    return features


@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],float32[:])), unicode_type)", fastmath = True, parallel = True, cache = True)
def transform_np(X, parameters,function_name='ter'):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
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

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                
                for feature_count in range(num_features_this_dilation):
                    if function_name == 'ter':
                        features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                    elif function_name == 'relu':
                        features[example_index, feature_index_start + feature_count] = _ReLU(C, biases[feature_index_start + feature_count]).mean()
                    elif function_name == 'cos':
                        features[example_index, feature_index_start + feature_count] = _Cos(C, biases[feature_index_start + feature_count]).mean()
                    elif function_name == 'sigmoid':
                        features[example_index, feature_index_start + feature_count] = _Sigmoid(C, biases[feature_index_start + feature_count]).mean()
                    elif function_name == 'pass':
                        features[example_index, feature_index_start + feature_count] = _Pass(C, biases[feature_index_start + feature_count]).mean()
                    elif function_name == 'linear':
                        features[example_index, feature_index_start + feature_count] = _Linear(C, biases[feature_index_start + feature_count]).mean()   
                    elif function_name == 'leakyrelu':
                        features[example_index, feature_index_start + feature_count] = _LeakyRelu(C, biases[feature_index_start + feature_count]).mean()  
                    else:
                        features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()


                            
                

                feature_index_start = feature_index_end

    return features 



@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],int64[:],float32[:])), unicode_type)", fastmath = True, parallel = True, cache = True)
def transform_refined(X, parameters,function_name='ter'):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, random_size ,biases = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
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

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
                
               
                
                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        c_size = random_size[feature_index_start + feature_count]
                
                        start_feature = feature_index_start + feature_count
                        end_feature = start_feature + c_size

                        beh = biases[start_feature:end_feature]
                        curve = create_rter(chunk = beh,length=len(C))
                        
                        if function_name == 'ter':
                            features[example_index, feature_index_start + feature_count] = _PPV(C, curve).mean()
                        elif function_name == 'relu':
                            features[example_index, feature_index_start + feature_count] = _ReLU(C, curve).mean()
                        else:
                            features[example_index, feature_index_start + feature_count] = _PPV(C, curve).mean()


                            
                else:
                    for feature_count in range(num_features_this_dilation):
                        c_size = random_size[feature_index_start + feature_count]
                
                        start_feature = feature_index_start + feature_count
                        end_feature = start_feature + c_size

                        beh = biases[start_feature:end_feature]
                        curve = create_rter(chunk = beh,length=len(C))
                        
                        if function_name == 'ter':
                             features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], curve[padding:-padding]).mean()
                        elif function_name == 'relu':
                            features[example_index, feature_index_start + feature_count] = _ReLU(C[padding:-padding], curve[padding:-padding]).mean()
                      
                        else:
                            features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], curve[padding:-padding]).mean()
                        

                feature_index_start = feature_index_end

    return features 
    

