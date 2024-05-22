
import numpy as np
import math
import copy
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




#-------------------------------------------multigrade model_forward---------------------------------------------------------
def multigrade_model_forward(X, layers_dims, parameters, activation):
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
    
    Returns
    -------
    N :                numpy array
                       multigrade dnn predict  
    caches :           list
                       containing [((N_0, W_1, b_1)， Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1})]
    
    """

    caches = []
    N = X
    L = len(layers_dims)                     # number of layers in the neural network, W, b
    
    
    # Implement forward propagation
    for l in range(1, L):
        N_prev = N 
        
        if l == L-1:
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
                       multigrade dnn predict  
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
'''---------------------compute gradient for {W_i, b_i}_{i=0}^{n1}--------------------------------------------'''

def multigrade_linear_activation_backward_L2reg(dN, cache, lambd_W, activation):
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
def multigrade_backward_L2reg(Y, N, layers_dims, parameters, caches, lambd_W, activation):
    """
    Implement the backward propagation
    
    Parameters
    ----------
    Y :              numpy array with shape (the dimension of output, number of data)
                     the "label" of data                 
    N :              numpy array with shape (the dimension of output, number of data)
                     predict of the multigrade dnn                                           
    layers_dims :    list 
                     containing the dimensions of each layer in network
    parameters :     dictionary 
                     containing parameters of the model
    caches :         list 
                     containing [((N_0, W_1, b_1), Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1}), ((N_{n1}))]
    lambd_W :        scalar
                     regularization hyperparameter for weight W
    activation :    string,  "sin" or "relu"
                    the activation to be used in this layer        

    
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
            grads["dN" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = multigrade_linear_activation_backward_L2reg(grads["dN" + str(l)], caches[l-1], 
                                                                                                                             lambd_W, "identity")
        else:
            grads["dN" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = multigrade_linear_activation_backward_L2reg(grads["dN" + str(l)], caches[l-1], 
                                                                                                                             lambd_W, activation)           
            
    return grads
#---------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------optimization---------------------------------------------------------
"""-----------------------------------------------------random mini batches----------------------------------------------"""
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





"""--------------------------------------------------adam method-------------------------------------------------------"""
#-----------------------------------------------------initialize_adam-----------------------------------------------
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
#---------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------put all moudel together---------------------------------------------------
def multigrade_dnn_model_grade_1(trained_variable, layers_dims, lambd_W, opt_parameter, max_learning_rate, min_learning_rate, epochs, activation):
    """
    Implements a multigrade deep neural network
    
    Parameters
    ----------
    data :                  dictionary 
                            information of orginal data  (train_X, train_Y, test_X, test_Y) 
    layers_dims :           list 
                            containing the dimensions of each layer in network
    lambd_W :               scalar
                            L2 regularization parameter for W              
    opt_parameter :         dictionary, information of optimization 
                            containing (optimizer,  learning_rate, mini_batch_size, beta1, beta2, epsilon, error, epochs)                            
                                error ----------------- scalar. if relative error of the cost function less than error then stop      
                                mini_batch_size ------- integer, size of a mini batch
                                beta1 ----------------- scalar, exponential decay hyperparameter for the first moment estimates 
                                beta2 ----------------- scalar, exponential decay hyperparameter for the second moment estimates
                                epsilon --------------- scalar, hyperparameter preventing division by zero in Adam updates 
                                epochs ---------------- integer, number of epochs
    max_learning_rate :     scalar
                            max learning rate
    min_learning_rate :     scalar
                            minimum learning rate
    epochs :                integer
                            the number of steps
    activation :            string
                            the activation function used in this grade
    
    Returns
    -------
    
    
    
    learning rate at step k :  max_learning_rate / e^{gamma * k}
                               where gamma = 1/epochs ln( max_learning_rate /  min_learning_rate  )
                       
                          
    """

    data = trained_variable["data"]
    
    # optimization parameter
    init_method = opt_parameter["init_method"]
    mini_batch_size = opt_parameter["mini_batch_size"] 
    beta1 = opt_parameter["beta1"] 
    beta2 = opt_parameter["beta2"]
    epsilon = opt_parameter["epsilon"] 
    error = opt_parameter["error"]  

                        
    train_rses = []                   
    validation_rses = []    
    train_costs = []
    validation_costs = []
    train_predict_record = []
    validation_predict_record = []  
    train_time_record = []
    spectral_norm_W = []
    REC_FRQ_iter = []
    
    
    
    gamma = 1/epochs*  np.log( max_learning_rate /  min_learning_rate  )
    
    
    t = 0                                                               # initializing the counter required for Adam update
    seed = 1
    
    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims, init_method)

    
    # Initialize
    v, s = initialize_adam(parameters, layers_dims)
    
    s_time = time.time()
    
    # Optimization loop
    for i in range(epochs+1):


        if i%opt_parameter["REC_FRQ"]==0:
            train_predict, _ = multigrade_model_forward(data["train_X"], layers_dims, parameters, activation)
            validation_predict, _ = multigrade_model_forward(data["validation_X"], layers_dims, parameters, activation) 
            e_time = time.time()
            train_predict_record.append(train_predict)
            validation_predict_record.append(validation_predict)
            train_time_record.append(e_time - s_time)
            train_costs.append(squareloss_cost_L2reg(train_predict, data["train_Y"], layers_dims, parameters, lambd_W))
            validation_costs.append(squareloss_cost_L2reg(validation_predict, data["validation_Y"], layers_dims, parameters, lambd_W))
            train_rses.append(rse(data["train_Y"], train_predict))
            validation_rses.append(rse(data["validation_Y"], validation_predict))
            REC_FRQ_iter.append(i)

        
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        if opt_parameter['SGD']:
            seed = seed + 1
            minibatches = random_mini_batches(data["train_X"], data["train_Y"], mini_batch_size, seed) 
        else:
            minibatches = [(data["train_X"], data["train_Y"])]


        for minibatch in minibatches:    
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation
            N, caches = multigrade_model_forward(minibatch_X, layers_dims, parameters, activation)


            # Backpropagation
            grads = multigrade_backward_L2reg(minibatch_Y, N , layers_dims, parameters, caches, lambd_W, activation)
        
 
            t = t + 1 # Adam counter
            learning_rate = max_learning_rate / np.exp( gamma * i )           
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, layers_dims, t, learning_rate, 
                                                           beta1, beta2,  epsilon)
            
        
    
        
    #current grade information, will be used in next grade    
    prev_info = {}    
    train_predict, train_caches = multigrade_model_forward(data["train_X"], layers_dims, parameters, activation)
    validation_predict, validation_caches = multigrade_model_forward(data["validation_X"], layers_dims, parameters, activation)   
    prev_info["train_predict"] = train_predict
    prev_info["validation_predict"] = validation_predict
    prev_info["train_N_prev_X"] = train_caches[-1][0][0]  
    prev_info["validation_N_prev_X"] = validation_caches[-1][0][0]
    
    
    
    trained_variable["multrain_Y"]  = data["train_Y"]  - train_predict    
    trained_variable["mulvalidation_Y"]  = data["validation_Y"]  - validation_predict  
    trained_variable["train_costs"].append(train_costs)
    trained_variable["validation_costs"].append(validation_costs)
    trained_variable["mul_parameters"].append(parameters)
    trained_variable["train_predict"].append(train_predict)
    trained_variable["validation_predict"].append(validation_predict)
    trained_variable['train_rse'].append(train_rses)
    trained_variable['validation_rse'].append(validation_rses)
    trained_variable["REC_FRQ_iter"].append(REC_FRQ_iter)
    trained_variable["prev_info"] = prev_info    
    trained_variable["train_predict_record"].append(train_predict_record)
    trained_variable["validation_predict_record"].append(validation_predict_record)     
    trained_variable['train_time'].append(train_time_record)
    

    return trained_variable
#-----------------------------------------------------------------------------------------------------------------------------






#-------------------------------------------------put all moudel together---------------------------------------------------
def multigrade_dnn_model_grade_ell(trained_variable, layers_dims, lambd_W, opt_parameter,  max_learning_rate, min_learning_rate, epochs, activation):
    """
    Implements a multiscale deep neural network
    
    Parameters
    ----------
    data :                  dictionary
                            store train_X, train_Y, test_X, test_Y
    multrain_Y :            numpy array
                            true label for current grade
    layers_dims :           list 
                            containing the dimensions of each layer in network
    lambd_W :               scalar
                            L2 regularization parameter for W              
    opt_parameter :         dictionary, information of optimization 
                            containing (optimizer,  learning_rate, mini_batch_size, beta1, beta2, epsilon, error, epochs)                            
                                error ----------------- scalar. if relative error of the cost function less than error then stop      
                                learning_rate --------- scalar, learning rate
                                mini_batch_size ------- integer, size of a mini batch
                                beta1 ----------------- scalar, exponential decay hyperparameter for the first moment estimates 
                                beta2 ----------------- scalar, exponential decay hyperparameter for the second moment estimates
                                epsilon --------------- scalar, hyperparameter preventing division by zero in Adam updates 
                                epochs ---------------- integer, number of epochs
    max_learning_rate :     scalar
                            max learning rate
    min_learning_rate :     scalar
                            minimum learning rate
    activation :            string
                           the activation function used in this grade
    prev_info :            dictionary, information of trained information. 
                                train_predict --------- numpy array, trained predict for the previous grade
                                test_predict ---------- numpy array, tested predict for the previous grade
                                train_N_prev_grade ---- list, each element contains the output of each layer previous grade in train data
                                test_N_prev_grade ----- list, each element contains the output of each layer previous grade in test data
    
    Returns
    -------
    
    
    learning rate at step k :  max_learning_rate / e^{gamma * k}
                               where gamma = 1/epochs ln( max_learning_rate /  min_learning_rate  )
                          
    """
    data = trained_variable["data"]
    multrain_Y = trained_variable["multrain_Y"] 
    mulvalidation_Y = trained_variable["mulvalidation_Y"] 
    
    # optimization parameter
    init_method = opt_parameter["init_method"]
    mini_batch_size = opt_parameter["mini_batch_size"] 
    beta1 = opt_parameter["beta1"] 
    beta2 = opt_parameter["beta2"]
    epsilon = opt_parameter["epsilon"] 
    error = opt_parameter["error"]    
    
    
    
    # pretrained information
    prev_info = trained_variable["prev_info"]
    train_predict = prev_info["train_predict"]
    validation_predict = prev_info["validation_predict"]
    train_N_prev_X = prev_info["train_N_prev_X"]
    validation_N_prev_X = prev_info["validation_N_prev_X"]
    
    
    
    gamma = 1/epochs*  np.log( max_learning_rate /  min_learning_rate  )

                     
    train_rses = []              
    validation_rses = [] 
    train_costs = []
    validation_costs = []
    train_predict_record = []
    validation_predict_record = []  
    train_time_record = []
    spectral_norm_W = []
    REC_FRQ_iter = []
    
    
    
    t = 0                              # initializing the counter required for Adam update
    seed = 1
    
    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims, init_method)
    
    # Initialize
    v, s = initialize_adam(parameters, layers_dims)
    

    s_time = time.time()
    # Optimization loop
    for i in range(epochs+1):


        if i%opt_parameter["REC_FRQ"]==0:
            train_N, _ = multigrade_model_forward(train_N_prev_X, layers_dims, parameters, activation)
            validation_N, _ = multigrade_model_forward(validation_N_prev_X, layers_dims, parameters, activation)  
            e_time = time.time()
            train_predict_record.append(train_N)
            validation_predict_record.append(validation_N)
            train_time_record.append(e_time - s_time)
            # Cost function
            train_costs.append(squareloss_cost_L2reg(train_N, multrain_Y, layers_dims, parameters, lambd_W))
            validation_costs.append(squareloss_cost_L2reg(validation_N, mulvalidation_Y, layers_dims, parameters, lambd_W))
            train_rses.append(rse(data["train_Y"], train_predict + train_N ))
            validation_rses.append(rse(data["validation_Y"], validation_predict + validation_N ))
            REC_FRQ_iter.append(i)


        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        if opt_parameter['SGD'] == True:
            seed = seed + 1
            minibatches = random_mini_batches(train_N_prev_X, multrain_Y, mini_batch_size, seed)
        else:
            minibatches = [(train_N_prev_X, multrain_Y)]


        
        for minibatch in minibatches:    
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation
            N, caches = multigrade_model_forward(minibatch_X, layers_dims, parameters, activation)

            # Backpropagation
            grads = multigrade_backward_L2reg(minibatch_Y, N , layers_dims, parameters, caches, lambd_W, activation)
        

            t = t + 1 # Adam counter
            learning_rate = max_learning_rate / np.exp( gamma * i)  
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, layers_dims, t, learning_rate, 
                                                           beta1, beta2,  epsilon)

        
        
        
    #current grade information, will be used in next grade    
    prev_info = {}    
    
    train_N, train_caches = multigrade_model_forward(train_N_prev_X, layers_dims, parameters, activation)
    validation_N, validation_caches = multigrade_model_forward(validation_N_prev_X, layers_dims, parameters, activation) 
    prev_info["train_predict"] = train_predict + train_N
    prev_info["validation_predict"] = validation_predict + validation_N
    prev_info["train_N_prev_X"] = train_caches[-1][0][0]  
    prev_info["validation_N_prev_X"] = validation_caches[-1][0][0]
    

    trained_variable["multrain_Y"]  = data["train_Y"] - prev_info["train_predict"]        
    trained_variable["mulvalidation_Y"]  = data["validation_Y"] - prev_info["validation_predict"]  
    trained_variable["train_costs"].append(train_costs)
    trained_variable["validation_costs"].append(validation_costs)
    trained_variable["mul_parameters"].append(parameters)
    trained_variable["train_predict"].append(train_predict + train_N)
    trained_variable["validation_predict"].append(validation_predict + validation_N)
    trained_variable['train_rse'].append(train_rses)
    trained_variable['validation_rse'].append(validation_rses)
    trained_variable["REC_FRQ_iter"].append(REC_FRQ_iter)
    trained_variable["prev_info"] = prev_info
    trained_variable['train_time'].append(train_time_record)
    trained_variable["train_predict_record"].append(train_predict_record)
    trained_variable["validation_predict_record"].append(validation_predict_record)     

    return trained_variable
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





