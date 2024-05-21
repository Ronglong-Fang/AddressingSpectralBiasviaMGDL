# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:06:10 2023

@author: rfang002
"""

#import matplotlib.pyplot as plt
import time
import pickle
import os.path
#from mmdnn_gc_utils import gradient_check_n_test_case
import multigrade_dnn_model as m_dnn
from ManifoldsData import Manifolds_regression_increase_amplituide, Manifolds_regression_vary_amplituide
from argparse import Namespace
import numpy as np


#------------------------------------nn_parameter--------------------------------

def multi_grade_dnn(max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, L, Amptype, SGD, mini_batch):
                               
    #---------------------------neural network parameter----------------------------
    nn_parameter = {}
    nn_parameter["mul_layers_dims"] = mul_layers_dims               # neural network strucure
    nn_parameter["lambd_W"] = [0, 0, 0, 0]                          # the L2 regularization  for the weight matrix. 
                                                                    # In the paper, we did not apply L2 regularization, 
                                                                    # thereby set to 0 for this parameter

    #------------------------optimization parameter----------------------------------
    opt_parameter = {}
     #---------------default paramwter for Adam algoirthm----------------------------------
    opt_parameter["optimizer"] = "adam"                             # use Adam optimizer  
    opt_parameter["beta1"] = 0.9
    opt_parameter["beta2"] = 0.999
    opt_parameter["epsilon"] = 1e-8
    opt_parameter["error"] = 1e-7
    #--------------------------------------------------------------------------------------
    opt_parameter["SGD"] = SGD
    opt_parameter["mini_batch_size"] = mini_batch                  # mini batch size
    opt_parameter["REC_FRQ"] = 100                                 # record loss evry 100 steps
    opt_parameter["init_method"] = "xavier"                        # use xavier initialization
    opt_parameter["max_learning_rate"] = max_learning_rate         # maximum learning rate
    opt_parameter["min_learning_rate"] = min_learning_rate         # minimum learning rate
    opt_parameter["epochs"] = mul_epochs                           # the training number of epoch in each grade
    opt_parameter['activation'] = 'relu'                           # use relu activation function
    #---------------------------------------------------------------------------------
    
    
    
    
    #----------------------------------record train history---------------------------
    trained_variable = {}                                     #store the different true lable and output for each grade     
    trained_variable["train_time"] = []                       #store train time for each grade
    trained_variable["mul_parameters"] = []                   #store parameter for each grade                  
    trained_variable["train_rse"] = []
    trained_variable["validation_rse"] = [] 
    trained_variable["train_predict"] = []
    trained_variable["validation_predict"] = []
    trained_variable["train_costs"] = []
    trained_variable["validation_costs"] = []  
    trained_variable["train_predict_record"] = []
    trained_variable["validation_predict_record"] = []
    trained_variable["REC_FRQ_iter"] = []
        


    # create data
    #------------
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
        
    



    trained_variable["data"] = data
    trained_variable = m_dnn.multigrade_dnn_model(nn_parameter, opt_parameter, trained_variable)


    save_path = 'results/'
 
    filename = "Amp%s_xavier_epoch%s_minibatch%s_MAXlearningrate%.4e_MINlearningrate%.4e_validation%.4e_train%.4e_L%d.pickle"%(Amptype, opt_parameter["epochs"], opt_parameter["mini_batch_size"], opt_parameter["max_learning_rate"],
                                                                                                                        opt_parameter["min_learning_rate"],trained_variable['validation_rse'][-1][-1],
                                                                                                                        trained_variable['train_rse'][-1][-1], L)
    fullfilename = os.path.join(save_path, filename)    
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([trained_variable, nn_parameter, opt_parameter],f)





    












