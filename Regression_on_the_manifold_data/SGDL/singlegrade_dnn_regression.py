# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:06:10 2023

@author: rfang002
"""


import numpy as np
import math
import copy
from tqdm import tqdm
from ManifoldsData import fft
import time



#-------------------initialize_parameters for grade 1: {W_i, b_i}_{i=1}^{n1}, {M_i}_{i=0}^{n1}--------------------------------
def initialize_parameters_deep(layers_dims, init_method):
    """
    initialize parameter for deep nerual network
    

    Parameters
    ----------
    layers_dims :        list
                         containing the dimensions of each layer in network
                         [input, hiddlen_layer1, ..., hiddlen_layern1, output]
    init_method :        string
                         method of initiation
                         "he": he initalize 

 
    
    Returns
    -------
    parameters :         dictionary 
                         containing neural network parameters
                         Wl : weight matrix of shape (layer_dims[l], layer_dims[l-1])
                         bl : bias vector of shape (layer_dims[l], 1)


                       
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)            # number of layers in the network  
    
        
        
        
    #{W_i, b_i}_{i=1}^{n1}
    for l in range(1, L):
    
        if init_method == "he":
            print('------------------we use he initialize--------------')
            ## He initialize : Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))   
        elif init_method == "xavier":
            print('------------------we use xavier initialize-------------------')
            #xavier initialize:Understanding the difficulty of training deep feedforward neural networks
            parameters["W" + str(l)] = np.random.uniform(-1, 1, (layers_dims[l], layers_dims[l - 1])) * np.sqrt(1 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.random.uniform(-1, 1, (layers_dims[l], 1)) * np.sqrt(1 / layers_dims[l - 1])

        
    return parameters
#---------------------------------------------------------------------------------------------------------------------------




#------------------------------------------linear_activation_forward---------------------------------------------------------
def linear_activation_forward(N_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer


    Parameters
    ----------
    N_prev :           numpy array of shape (dimension of previous, number of example) 
                       activations from previous layer (or input data)
    W :                numpy array of shape (size of current layer, size of previous layer)
                       weights matrix  
    b :                numpy array of shape (size of the current layer, 1)
                       bias vector 
    activation :       string,  "sin" or "relu" or "identity"
                       the activation to be used in this layer

    Returns
    -------
    N :                numpy array of shape (size of the current layer, 1)
                       the output of the activation function, also called the post-activation value 
    cache :            dictionary containing "linear_cache" and "activation_cache";
                       stored for computing the backward pass efficiently



    """
    Z = np.dot(W, N_prev) + b

    if activation == "relu":
        N = np.maximum(0,Z)
    elif activation == "sin":
        N = np.sin(Z)
    elif activation == "identity":
        N = Z   

    cache = ((N_prev, W, b), Z)

    return N, cache
#----------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------singlegrade model_forward---------------------------------------------------------
def singlegrade_model_forward(X, layers_dims, parameters, activation, sinORrelu):
    """
    Implement forward propagation
    
    Parameters
    ----------
    X :                numpy array of shape (input size, number of examples)
                       dataset
    layers_dims :      list 
                       containing the dimensions of each layer in network
    parameters :       dictionary
                       output of initialize_parameters_deep()
    activation :       string,  "sin" or "relu"
                       the activation to be used in this layer
    sinORrelu :        integer
                       the first sinORrelu layer use sin function as 
                       activation function
                                                                
    Returns
    -------
    N :                numpy array
                       singlegrade dnn predict  
    caches :           list
                       containing [((N_0, W_1, b_1)， Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1})]
    
    """

    caches = []
    N = X
    L = len(layers_dims)                     # number of layers in the neural network, W, b
    
    
    # Implement forward propagation
    for l in range(1, L):
        N_prev = N 
        
        if l<sinORrelu:
            N, cache = linear_activation_forward(N_prev, parameters["W" + str(l)], 
                                                 parameters["b" + str(l)], "sin")  
        elif l == L-1:
            N, cache = linear_activation_forward(N_prev, parameters["W" + str(l)], 
                                                 parameters["b" + str(l)], "identity")            
        else:
            N, cache = linear_activation_forward(N_prev, parameters["W" + str(l)], 
                                                 parameters["b" + str(l)], activation)               

        caches.append(cache)
        
    
                


    return N, caches
#---------------------------------------------------------------------------------------------------------------------------




#------------------------------------------------------compute cost----------------------------------------------------------
def squareloss_cost_L2reg(N, Y, layers_dims, parameters, lambd_W):
    """
    Implement the square loss cost function
                        cost = 1/(2m) sum_{i=1}^{m}(phi(x_i) - y_i)^2 + lambd_W/2 sum_{i=1}^{n_1} square(W_i) 

    Parameters
    ----------
    N :                numpy array
                       singlegrade dnn predict  
    Y :                numpy array
                       true "label" vector 
    layers_dims :      list 
                       containing the dimensions of each layer in network
    parameters :       dictionary 
                       containing parameters of the model
    lambd_W :          scalar
                       regularization hyperparameter for weight W

                       

    Returns
    -------
    cost :             number 
                       square loss cost function
          
    
    """
    
    
    # the number of train data 
    m = Y.shape[1]


    #compute square loss
    squareloss_cost = np.sum(np.square(Y - N)) / (2*m) 
    
    
    # compute L2 regularization term
    L = len(layers_dims)
    
    L2reg_cost = 0
    for l in range(1, L):
        L2reg_cost = L2reg_cost + lambd_W/2 *  np.sum(np.square(parameters["W"+str(l)])) 
    
   
    cost = squareloss_cost  + L2reg_cost        # total loss
        
    
    return cost
#---------------------------------------------------------------------------------------------------------------------------




#-----------------------------------------------------compute gradient------------------------------------------------------           
#------------------------------------------linear_activation_backward--------------------------------------------------------
#---------------------compute gradient for {W_i, b_i}_{i=0}^{n1}-------------------------------------------------------------
def singlegrade_linear_activation_backward_L2reg(dN, cache, lambd_W, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer. compute gradient for W, b
    
    Parameters
    ----------
    dN :            numpy array with shape (dimension of current layer, number of example) 
                    post-activation gradient for current layer l 
    caches :        list
                    containing [((N_0, W_1, b_1)， Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1}), ((N_{n1}))]
    lambd_W :       scalar
                    regularization hyperparameter
    activation :    string,  "sin" or "relu"
                    the activation to be used in this layer
    
    Returns
    -------
    dN_prev :       numpy array with shape (dimension of previous layer, number of example)
                    gradient of the cost with respect to the activation (of the previous layer l-1), same shape as N_prev
    dW :            numpy array with shape (dimension of current layer, dimension of previous layer)
                    gradient of the cost with respect to W (current layer l), same shape as W
    db :            numpy array with shape (dimension of current layer, 1)
                    gradient of the cost with respect to b (current layer l), same shape as b
        
        
    """
    linear_cache, activation_cache = cache
    
    
    # calculate dZ
    if activation == "relu":
        dN[activation_cache<0] = 0
        dZ = dN  
    elif activation == "sin":
        dZ = np.multiply(dN, np.cos(activation_cache))   
    elif activation == "identity":
        dZ = dN
                
        
    #calculate dW, db, dN_prev based on dZ     
    N_prev, W, b = linear_cache

    dW = np.dot(dZ, N_prev.T) + lambd_W*W             
    db = np.sum(dZ, axis=1, keepdims=True)
    dN_prev = np.dot(W.T, dZ)
    
    return dN_prev, dW, db
#---------------------------------------------------------------------------------------------------------------------------




#---------------------------------------compute gradient for W, b-----------------------------------------------------------
def singlegrade_backward_L2reg(Y, N, layers_dims, parameters, caches, lambd_W, activation, sinORrelu):
    """
    Implement the backward propagation
    
    Parameters
    ----------
    Y :              numpy array with shape (the dimension of output, number of data)
                     the "label" of data                 
    N :              numpy array with shape (the dimension of output, number of data)
                     predict of the singlegrade dnn                                           
    layers_dims :    list 
                     containing the dimensions of each layer in network
    parameters :     dictionary 
                     containing parameters of the model
    caches :         list 
                     containing [((N_0, W_1, b_1), Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1}), ((N_{n1}))]
    lambd_W :        scalar
                     regularization hyperparameter for weight W
    activation :     string,  "sin" or "relu"
                     the activation to be used in this layer
    sinORrelu :      integer
                     the first sinORrelu layer use sin function as 
                     activation function

    
    Returns
    -------
    grads :          dictionary with the gradients
                     grads["dN" + str(l)] = ... 
                     grads["dW" + str(l)] = ...
                     grads["db" + str(l)] = ... 
                    
                    
    """
    # the number of train data 
    m = Y.shape[1]    
    
    L = len(layers_dims) # the number of layers
    grads = {}
    
    
    # Initializing the backpropagation
    grads["dN" + str(L-1)] =  1/m * (N - Y)
    
    
    for l in reversed(range(1, L)):
        # backward
        
        if l == L-1:
            grads["dN" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = singlegrade_linear_activation_backward_L2reg(grads["dN" + str(l)], caches[l-1], 
                                                                                                                             lambd_W, "identity")
        elif l < sinORrelu:
            grads["dN" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = singlegrade_linear_activation_backward_L2reg(grads["dN" + str(l)], caches[l-1], 
                                                                                                                             lambd_W, "sin")
        else:
            grads["dN" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = singlegrade_linear_activation_backward_L2reg(grads["dN" + str(l)], caches[l-1], 
                                                                                                                             lambd_W, activation)           
            
    return grads
#---------------------------------------------------------------------------------------------------------------------------





#------------------------------------------------------optimization--------------------------------------------------------
#-----------------------------------------------------random mini batches--------------------------------------------------
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Parameter
    ---------
    X :                  numpy array with shape (input size, number of examples)      
                         input data 
    Y :                  numpy array with shape (output size, number of examples)              
                         true "label" vector 
    mini_batch_size :    integer 
                         size of the mini-batches
    
    Returns
    -------
    mini_batches :       list 
                         synchronous (mini_batch_X, mini_batch_Y)
                        
                        
    """    
    
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    c, m = Y.shape                  # m, number of training examples, c, number of components
                               
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((c,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
#---------------------------------------------------------------------------------------------------------------------------





#------------------------------------------------------adam method---------------------------------------------------------
#-----------------------------------------------------initialize_adam------------------------------------------------------
def initialize_adam(parameters, layers_dims) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL", "dM0", "dM1", ..., "dML" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Parameters
    ----------
    parameters :         dictionary containing parameters
                         parameters["W" + str(l)] = Wl
                         parameters["b" + str(l)] = bl
                         parameters["M" + str(l)] = Ml
    layers_dims :        list 
                         containing the dimensions of each layer in network
    
    Returns
    -------
    v :                  dictionary containing the exponentially weighted average of the gradient
                         v["dW" + str(l)] = ...
                         v["db" + str(l)] = ...
    s :                  dictionary containing the exponentially weighted average of the squared gradient.
                         s["dW" + str(l)] = ...
                         s["db" + str(l)] = ...

    
   """
    
    L = len(layers_dims)   # number of layers in the neural networks
    v = {}
    s = {}
    
    
    for l in range(1, L):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    
    return v, s
#------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------update_parameters_with_adam-------------------------------------------------
def update_parameters_with_adam(parameters, grads, v, s, layers_dims, t, learning_rate=0.001,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam
    
    Parameters
    ----------
    parameters :         dictionary containing parameters
                         parameters["W" + str(l)] = Wl
                         parameters["b" + str(l)] = bl
    grads :              dictionary containing your gradients for each parameters:
                         grads['dW' + str(l)] = dWl
                         grads['db' + str(l)] = dbl
    v :                  dictionary containing the exponentially weighted average of the gradient
                         v["dW" + str(l)] = ...
                         v["db" + str(l)] = ...
    s :                  dictionary containing the exponentially weighted average of the squared gradient.
                         s["dW" + str(l)] = ...
                         s["db" + str(l)] = ...
    layers_dims :        list 
                         containing the dimensions of each layer in network
    learning_rate :      scalar
                         learning rate
    beta1 :              scalar
                         exponential decay hyperparameter for the first moment estimates 
    beta2 :              scalar
                         exponential decay hyperparameter for the second moment estimates 
    epsilon :            scalar
                         hyperparameter preventing division by zero in Adam updates

    Returns
    -------
    parameters :         dictionary containing updated parameters
                         parameters["W" + str(l)] = Wl
                         parameters["b" + str(l)] = bl
    v :                  dictionary containing the updated exponentially weighted average of the gradient
                         v["dW" + str(l)] = ...
                         v["db" + str(l)] = ...
    s :                  dictionary containing the updated exponentially weighted average of the squared gradient.
                         s["dW" + str(l)] = ...
                         s["db" + str(l)] = ...
                         
    """
    
    L = len(layers_dims)                     # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # perform Adam update on all parameters
    
    for l in range(1, L):
        # moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]


        # compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))
        

        # moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads["dW" + str(l)], 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads["db" + str(l)], 2)       
        
        
        # compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))   
        
        
        # update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s
#-----------------------------------------------------------------------------------------------------------------------------




def spectral_norm_weight(parameters, layers_dims):
    """
    

    Parameters
    ----------
    parameters :         dictionary containing parameters
                         parameters["W" + str(l)] = Wl
                         parameters["b" + str(l)] = bl       
                         
    layers_dims :        list 
                         containing the dimensions of each layer in network

    Returns
    -------
    spectral_norm_W :    list
                         spectral norm for weight matrix for each layer

    """
    
    
    L = len(layers_dims)            # number of layers in the network  
    
        
    spectral_norm_W = []  
        
    #{W_i, b_i}_{i=1}^{n1}
    for l in range(1, L):
        W = parameters["W"+str(l)]
        spectral_norm_W.append(np.linalg.norm(W, ord = 2))   #2-norm (largest sing. value)
        
    return spectral_norm_W


def fro_norm_weight_gradient(grads, layers_dims):
    """
    

    Parameters
    ----------
    grads :              dictionary containing gradient
                         grads["dW" + str(l)]
                         grads["db" + str(l)]      
                         
    layers_dims :        list 
                         containing the dimensions of each layer in network

    Returns
    -------
    spectral_norm_dW :    list
                          fro norm of the gradient of weight matrix for each layer

    """
    
    
    L = len(layers_dims)            # number of layers in the network  
    
        
    spectral_norm_dW = []  
        
    #{W_i, b_i}_{i=1}^{n1}
    for l in range(1, L):
        dW = grads["dW"+str(l)]
        spectral_norm_dW.append(np.linalg.norm(dW, ord = 'fro'))   #2-norm (largest sing. value)
        
    return spectral_norm_dW
        
    


#-------------------------------------------------put all moudel together---------------------------------------------------
def singlegrade_dnn_model_grade(data, nn_parameter, opt_parameter):
    """
    Implements a singlegrade deep neural network
    
    Parameters
    ----------
    data :                  dictionary 
                            information of orginal data  (train_X, train_Y, test_X, test_Y) 
    nn_parameter :          dictionary
                            containing the information for nueral network                        
                               layers_dims-----list, containing the dimensions of each layer in network
                               lambd_W---------scalar, L2 regularization parameter for W
                               activation------string, the activation function used in this grade                            
                               sinORrelu-------integer, the first sinORrelu layer use sin function as activation function                                                           
    opt_parameter :         dictionary, information of optimization 
                            containing (optimizer,  learning_rate, mini_batch_size, beta1, beta2, epsilon, error, epochs)                            
                                error ----------------- scalar. if relative error of the cost function less than error then stop      
                                learning_rate --------- scalar, learning rate
                                mini_batch_size ------- integer, size of a mini batch
                                beta1 ----------------- scalar, exponential decay hyperparameter for the first moment estimates 
                                beta2 ----------------- scalar, exponential decay hyperparameter for the second moment estimates
                                epsilon --------------- scalar, hyperparameter preventing division by zero in Adam updates 
                                epochs ---------------- integer, number of epochs


    
    Returns
    -------
    parameters :           dictionary containing optimized parameters
                           parameters["W" + str(l)] = Wl
                           parameters["b" + str(l)] = bl
    costs :                list
                           record the history of cost
    ite :                  integer 
                           the number of iteration used in optimization
    train_rses :           list
                           record train rse
    test_rses :            record test rse

    """
    
    # neural network parameter
    init_method = nn_parameter["init_method"]
    layers_dims = nn_parameter["layers_dims"]
    lambd_W = nn_parameter["lambd_W"]
    activation = nn_parameter["activation"]
    sinORrelu = nn_parameter["sinORrelu"]
    
    
    
    # optimization parameter
    mini_batch_size = opt_parameter["mini_batch_size"] 
    beta1 = opt_parameter["beta1"] 
    beta2 = opt_parameter["beta2"]
    epsilon = opt_parameter["epsilon"] 
    error = opt_parameter["error"]  
    epochs = opt_parameter["epochs"]
    max_learning_rate = opt_parameter["max_learning_rate"]
    min_learning_rate = opt_parameter["min_learning_rate"]
    
    
    
    
    history  = {}
    history["data"] = data
    
    
    history["train_predict_record"] = []
    history["validation_predict_record"] = []
    history["train_costs"] = []
    history["validation_costs"] = []
    history["train_rses"] = []
    history["validation_rses"] = []
    history["spectral_norm_W"] = []
    history["fro_norm_dW"] = []
    history["REC_FRQ_iter"] = []
    history["time"] = []
    history["parameters"] = []
    

    gamma = 1/epochs *  np.log( max_learning_rate /  min_learning_rate  )    
    
    
    t = 0                              # initializing the counter required for Adam update
    seed = 1
    
    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims, init_method)

    
    # Initialize
    v, s = initialize_adam(parameters, layers_dims)

    s_time = time.time()
    

    
    # Optimization loop
    for i in range(epochs+1):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        if opt_parameter['SGD'] == True:
            seed = seed + 1
            minibatches = random_mini_batches(data["train_X"], data["train_Y"], mini_batch_size, seed) 
        else:
            minibatches = [(data["train_X"], data["train_Y"])]


        if i%opt_parameter["REC_FRQ"]==0:
            
            train_predict, _ = singlegrade_model_forward(data["train_X"], layers_dims, parameters, activation, sinORrelu)
            validation_predict, _ = singlegrade_model_forward(data["validation_X"], layers_dims, parameters, activation, sinORrelu)
            e_time = time.time()
            history["time"].append(e_time - s_time)
            history["train_rses"].append(rse(data["train_Y"], train_predict))
            history["validation_rses"].append(rse(data["validation_Y"], validation_predict))
            history["train_costs"].append(squareloss_cost_L2reg(train_predict, data["train_Y"], layers_dims, parameters, lambd_W))
            history["validation_costs"].append(squareloss_cost_L2reg(validation_predict, data["validation_Y"], layers_dims, parameters, lambd_W))
            history["train_predict_record"].append(train_predict)
            history["validation_predict_record"].append(validation_predict)
            history["spectral_norm_W"].append(spectral_norm_weight(parameters, layers_dims))
            history["parameters"].append(parameters)
            history["REC_FRQ_iter"].append(i)


        
        for minibatch in minibatches:    
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation
            N, caches = singlegrade_model_forward(minibatch_X, layers_dims, parameters, activation, sinORrelu)
 
            # Backpropagation
            grads = singlegrade_backward_L2reg(minibatch_Y, N , layers_dims, parameters, caches, lambd_W, activation, sinORrelu)
            t = t + 1 # Adam counter
            learning_rate = max_learning_rate / np.exp( gamma * i )    
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, layers_dims, t, learning_rate, 
                                                           beta1, beta2,  epsilon)


            

        

    return history
#-----------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------calculate rse----------------------------------------------------------
def rse(y_true, y_predict):
    """
    calculate the rse of the results:
        rse = sum_{i=1}^{N}(y_true - y_predict)^2 / sum_{i=1}^{N} (y_true)^2
    

    Parameters
    ----------
    y_true :           numpy array
                       true label
    y_predict :        numpy predict     
                       predict label          

    Returns
    -------
    rse:               scale
                       rse between true label and predict label
                       

    """
    rse = np.sum( np.power(y_true - y_predict, 2) ) / np.sum( np.power(y_true, 2) )
    
    
    return rse
#-----------------------------------------------------------------------------------------------------------------------------





