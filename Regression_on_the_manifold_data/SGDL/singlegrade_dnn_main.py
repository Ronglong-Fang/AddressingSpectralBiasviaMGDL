import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os.path
import singlegrade_dnn_regression as dnn
from argparse import Namespace
from ManifoldsData import Manifolds_regression_increase_amplituide, Manifolds_regression_vary_amplituide



def single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, Amptype, L, SGD):

                     
    nn_parameter = {}   
    nn_parameter["layers_dims"] = layers_dims
    nn_parameter["lambd_W"] = 0
    nn_parameter["sinORrelu"] = 0
    nn_parameter["activation"] = "relu"
    nn_parameter["init_method"] = "xavier"
    
    
    
    #------------------------optimization parameter----------------------------------
    opt_parameter = {}
    opt_parameter["optimizer"] = "adam"
    opt_parameter["beta1"] = 0.9
    opt_parameter["beta2"] = 0.999
    opt_parameter["epsilon"] = 1e-8
    opt_parameter["error"] = 1e-7
    opt_parameter["max_learning_rate"] = max_learning_rate
    opt_parameter["min_learning_rate"] = min_learning_rate
    opt_parameter["epochs"] = epochs
    opt_parameter["REC_FRQ"] = 100
    opt_parameter["SGD"] = SGD
    opt_parameter["mini_batch_size"] = mini_batch_size
    

    if Amptype == "vary":
        opt = Namespace()
        opt.ntrain = 12000
        opt.nvalidation = 4000
        opt.ntest = 4000
        opt.kappa = np.linspace(10, 400, 40)
        random_state = np.random.RandomState(seed=0)
        opt.phi = random_state.uniform(low=0, high=2*np.pi, size=40)
        opt.L = L
        data = Manifolds_regression_vary_amplituide(opt)
        
    elif Amptype == "increase":
        opt = Namespace()
        opt.ntrain = 12000
        opt.nvalidation = 4000
        opt.ntest = 4000
        opt.kappa = np.linspace(10, 400, 40)
        opt.alpha = np.linspace(0.025, 1, 40)
        random_state = np.random.RandomState(seed=0)
        opt.phi = random_state.uniform(low=0, high=2*np.pi, size=40)
        opt.L = L
        data = Manifolds_regression_increase_amplituide(opt)
        
        
    history = dnn.singlegrade_dnn_model_grade(data, nn_parameter, opt_parameter)

    save_path = 'results/'
    filename = "Amp%s_xavier_single_epochs%d_MAXlearningrate%.2e_MINlearningrate%.2e_validation%.4e_train%.4e_minibatchzie%s_L%d.pickle"%(Amptype, epochs, opt_parameter["max_learning_rate"],
                                                                                                                                          opt_parameter["min_learning_rate"],history['validation_rses'][-1],
                                                                                                                                          history['train_rses'][-1], opt_parameter["mini_batch_size"], L)
    fullfilename = os.path.join(save_path, filename)    
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([history, nn_parameter, opt_parameter],f)






    











