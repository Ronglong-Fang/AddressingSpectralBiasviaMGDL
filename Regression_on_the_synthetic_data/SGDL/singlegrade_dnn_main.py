# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:06:10 2023

@author: rfang002
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os.path
import singlegrade_dnn_regression as dnn
from argparse import Namespace
from SpectralBiasData import Spectral_bias_constantORincrease_amplituide, Spectral_bias_vary_amplituide



def single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, Amptype, SGD):

                     
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
    

    if Amptype == "constant":
        # constant
        opt = Namespace()
        opt.ntrain = 6000
        opt.nvalidation = 2000
        opt.ntest = 2000
        opt.kappa = np.linspace(10, 200, 20)
        opt.alpha = np.linspace(1, 1, 20)
        random_state = np.random.RandomState(seed=0)
        opt.phi = random_state.uniform(low=0, high=2*np.pi, size=40)
        data = Spectral_bias_constantORincrease_amplituide(opt)       

        
        
    elif Amptype == "decrease":
        # increase
        opt = Namespace()
        opt.ntrain = 6000
        opt.nvalidation = 2000
        opt.ntest = 2000
        opt.kappa = np.linspace(10, 200, 20)
        opt.alpha = np.linspace(1, 0.05, 20)
        random_state = np.random.RandomState(seed=0)
        opt.phi = random_state.uniform(low=0, high=2*np.pi, size=40)
        data = Spectral_bias_constantORincrease_amplituide(opt)
        
        
        
    elif Amptype == "increase":
        # increase
        opt = Namespace()
        opt.ntrain = 6000
        opt.nvalidation = 2000
        opt.ntest = 2000
        opt.kappa = np.linspace(10, 200, 20)
        opt.alpha = np.linspace(0.05, 1, 20)
        random_state = np.random.RandomState(seed=0)
        opt.phi = random_state.uniform(low=0, high=2*np.pi, size=40)
        data = Spectral_bias_constantORincrease_amplituide(opt)

        
    elif Amptype == "vary":
        opt = Namespace()
        opt.ntrain = 6000
        opt.nvalidation = 2000
        opt.ntest = 2000
        opt.kappa = np.linspace(10, 200, 20)
        random_state = np.random.RandomState(seed=0)
        opt.phi = random_state.uniform(low=0, high=2*np.pi, size=40)
        data = Spectral_bias_vary_amplituide(opt)
        
        
    history = dnn.singlegrade_dnn_model_grade(data, nn_parameter, opt_parameter)

    save_path = 'results/'.format(Amptype)
    filename = "Amp%s_xavier_single_epochs%d_MAXlearningrate%.2e_MINlearningrate%.2e_validation%.4e_train%.4e_minibatchzie%s.pickle"%(Amptype, epochs, opt_parameter["max_learning_rate"],
                                                                                                                                      opt_parameter["min_learning_rate"],history['validation_rses'][-1],
                                                                                                                                      history['train_rses'][-1], opt_parameter["mini_batch_size"])
    fullfilename = os.path.join(save_path, filename)        
    with open(fullfilename, 'wb') as f:
        pickle.dump([history, nn_parameter, opt_parameter],f)






    











