
#import matplotlib.pyplot as plt
import time
import pickle
import os.path
import multigrade_dnn_model as m_dnn
from MNISTLabelNoiseData import get_mnist
from argparse import Namespace
import numpy as np


#------------------------------------nn_parameter--------------------------------

def multi_grade_dnn(max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, freq, amp):
                               
    #---------------------------neural network parameter----------------------------
    nn_parameter = {}
    nn_parameter["mul_layers_dims"] = mul_layers_dims         # neural network strucure
    nn_parameter["lambd_W"] = [0, 0, 0]                       # the L2 regularization  for the weight matrix. 
                                                              # In the paper, we did not apply L2 regularization, 
                                                              # thereby set to 0 for this parameter


    #------------------------optimization parameter----------------------------------
    opt_parameter = {}
     #---------------default paramwter for Adam algoirthm----------------------------------
    opt_parameter["optimizer"] = "adam"                        # use Adam optimizer  
    opt_parameter["beta1"] = 0.9
    opt_parameter["beta2"] = 0.999
    opt_parameter["epsilon"] = 1e-8
    opt_parameter["error"] = 1e-7
    #-------------------------------------------------------------------------------------
    opt_parameter["SGD"] = SGD
    opt_parameter["mini_batch_size"] = mini_batch             # mini batch size
    opt_parameter["REC_FRQ"] = 1                              # record loss evry steps
    opt_parameter["init_method"] = "xavier"                   # use xavier initialization
    opt_parameter["max_learning_rate"] = max_learning_rate    # maximum learning rate
    opt_parameter["min_learning_rate"] = min_learning_rate    # minimum learning rate
    opt_parameter["epochs"] = mul_epochs                      # the training number of epoch in each grade
    opt_parameter['activation'] = 'relu'                           # use relu activation function
    #---------------------------------------------------------------------------------
    
    
    
    
    #----------------------------------record train history---------------------------
    trained_variable = {}                                     #store the different true lable and output for each grade     
    trained_variable["train_time"] = []                       #store train time for each grade
    trained_variable["mul_parameters"] = []                   #store parameter for each grade                  
    trained_variable["train_rse"] = []
    trained_variable["validation_rse"] = [] 
    trained_variable["cleantrain_rse"] = []
    trained_variable["cleanvalidation_rse"] = []     
    trained_variable["train_predict"] = []
    trained_variable["validation_predict"] = []
    trained_variable["train_costs"] = []
    trained_variable["validation_costs"] = []
    trained_variable["cleantrain_costs"] = []
    trained_variable["cleanvalidation_costs"] = []    
    trained_variable["REC_FRQ_iter"] = []
        

    # generate dataset
    opt = Namespace()
    opt.NORM_K = freq            # <--- Norm of the frequency vector
    opt.NOISE = True             # <--- If true, add noise to the train label
    opt.AMP_Z = amp              # <--- Amplitude of noise
    data  = get_mnist(opt)
    trained_variable["opt"] = opt
    trained_variable = m_dnn.multigrade_dnn_model(nn_parameter, opt_parameter, trained_variable, data)


    save_path = 'results/'
 
    filename = "MGDL_batchzie%s_epochs%s_MAXLrate%.2e_MINLrate%.2e_vLoss%.4e_cleanvloss%.4e_tLoss%.4e_amp%.1e_freq%.1e.pickle"%(opt_parameter["mini_batch_size"],opt_parameter['epochs'],
                                                                                                                                opt_parameter["max_learning_rate"],
                                                                                                                                opt_parameter["min_learning_rate"],
                                                                                                                                trained_variable['validation_costs'][-1][-1],
                                                                                                                                trained_variable['cleanvalidation_costs'][-1][-1],
                                                                                                                                trained_variable['train_costs'][-1][-1], 
                                                                                                                                opt.AMP_Z, opt.NORM_K)
    fullfilename = os.path.join(save_path, filename)    
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([trained_variable, nn_parameter, opt_parameter],f)





    












