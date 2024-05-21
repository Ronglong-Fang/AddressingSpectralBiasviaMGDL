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
from MNISTLabelNoiseData import get_mnist



def single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, freq, amp):

                     
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
    opt_parameter["REC_FRQ"] = 1
    opt_parameter["SGD"] = SGD
    opt_parameter["mini_batch_size"] = mini_batch_size
    

    # generate dataset
    opt = Namespace()
    opt.NORM_K = freq            # <--- Norm of the frequency vector
    opt.NOISE = True             # <--- If true, add noise to the train label
    opt.AMP_Z = amp              # <--- Amplitude of noise
    data  = get_mnist(opt)
    
    history = dnn.singlegrade_dnn_model_grade(data, nn_parameter, opt_parameter)
    history["opt"] = opt

    save_path = 'results'
    filename = "SGDL_epochs%d_MAXLrate%.2e_MINLrate%.2e_vLoss%.4e_cleanvloss%.4e_tLoss%.4e_batchzie%s_amp%.1e_freq%.1e.pickle"%(opt_parameter['epochs'], opt_parameter["max_learning_rate"],
                                                                                                                                opt_parameter["min_learning_rate"],
                                                                                                                                history['validation_costs'][-1],
                                                                                                                                history['cleanvalidation_costs'][-1],
                                                                                                                                history['train_costs'][-1],
                                                                                                                                opt_parameter["mini_batch_size"],
                                                                                                                                opt.AMP_Z, opt.NORM_K)
    fullfilename = os.path.join(save_path, filename)    
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([history, nn_parameter, opt_parameter],f)






    











