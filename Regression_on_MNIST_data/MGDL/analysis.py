# -*- coding: utf-8 -*-
"""
Created on Wed 12/31 2023

@author: rfang002
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import shutil
import os
import itertools
import sys
from multigrade_dnn_model import multigrade_dnn_model_predict_noisedata, multigrade_dnn_model_predict_cleandata
sys.path.append('/home/rfang002/envs/default-tensorflow-cpu-2.6.0/lib/python3.10/site-packages')
import seaborn as sns
sns.set()




def results_analysis(fullfilename, figure):

    with open(fullfilename, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)
        
    
    print("################################################################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)
    


    num_iter = trained_variable["REC_FRQ_iter"]

    min_indexs = []
    min_train_rses = []
    min_validation_rses = []

    train_rse = []
    validation_rse = []
    train_time = []

    
    total_time =  0

    for i in range(0, len(nn_parameter["mul_layers_dims"])):       
        total_time = total_time + trained_variable['train_time'][i][-1]
        train_rse.append(trained_variable['train_rse'][i][-1])
        validation_rse.append(trained_variable['validation_rse'][i][-1])
        train_time.append(trained_variable['train_time'][i][-1])
        
        
    noisetest_rse = multigrade_dnn_model_predict_noisedata(nn_parameter, opt_parameter, trained_variable)
    cleantest_rse = multigrade_dnn_model_predict_cleandata(nn_parameter, opt_parameter, trained_variable)
        

    print('the train rse for each grade is {}'.format(train_rse))
    print('the validation rse for each grade is {}'.format(validation_rse))
    print('the noisetest rse for each grade is {}'.format(noisetest_rse))
    print('the cleantest rse for each grade is {}'.format(cleantest_rse))
    
    print('the train times for each grade is {}'.format(train_time))
    print('the total train times is {}'.format(total_time))
    

    return 



# compare loss for SGDL and MGDL
def results_analysis_bestval(singlegradepicklefile, multigradepicklefile, tmaxnum, tminnum, vmaxnum, vminnum):
    num = len(multigradepicklefile)

    fig, axs = plt.subplots(1, 2)  # Create subplots for MDNN and SDNN

    for i in range(num):
        singlefile = singlegradepicklefile[i]
        multifile = multigradepicklefile[i]

        with open(singlefile, 'rb') as f:
            history, _, SGDL_opt_parameter = pickle.load(f)
        with open(multifile, 'rb') as f:
            trained_variable, MGDL_nn_parameter, MGDL_opt_parameter = pickle.load(f)

        opt = history["opt"]
        SGDL_epochs = SGDL_opt_parameter['epochs']
        
        SGDL_time  = history['time'][-1]
        MGDL_time =  0
        MGDL_epochs = 0
        for i in range(0, len(MGDL_nn_parameter["mul_layers_dims"])):       
            MGDL_time = MGDL_time + trained_variable['train_time'][i][-1]
            MGDL_epochs = MGDL_epochs + MGDL_opt_parameter['epochs'][i]                   
        
        SGDL_xaxis = np.arange(0, SGDL_epochs) * (SGDL_time / SGDL_epochs)
        MGDL_xaxis = np.arange(0, MGDL_epochs)* (MGDL_time / MGDL_epochs)       
        
        single_validation_costs = history["validation_costs"]
        mul_validation_costs = list(itertools.chain.from_iterable(trained_variable['validation_costs']))

        # if opt.NORM_K == 1.0 or opt.NORM_K ==5 or opt.NORM_K == 10 or opt.NORM_K == 50:
        axs[0].plot(SGDL_xaxis, single_validation_costs[:SGDL_epochs], label='amp = {}'.format(int(opt.AMP_Z)))
        axs[1].plot(MGDL_xaxis, mul_validation_costs[:MGDL_epochs], label='freq = {}'.format(int(opt.AMP_Z)))
  

    axs[0].set_xlabel("Training time [s]")
    axs[0].set_ylabel("Validation loss")
    axs[0].set_ylim(vminnum, vmaxnum)
    axs[0].legend()
    axs[0].set_title('SGDL')

    axs[1].set_xlabel("Training time [s]")
    axs[1].set_ylabel("Validation loss")
    axs[1].set_ylim(vminnum, vmaxnum)
    axs[1].legend()
    axs[1].set_title('MGDL')

    plt.tight_layout()
    plt.show()
    
    
    fig, axs = plt.subplots(1, 2)  # Create subplots for MDNN and SDNN

    for i in range(num):
        singlefile = singlegradepicklefile[i]
        multifile = multigradepicklefile[i]

        with open(singlefile, 'rb') as f:
            history, _, SGDL_opt_parameter = pickle.load(f)
        with open(multifile, 'rb') as f:
            trained_variable, MGDL_nn_parameter, MGDL_opt_parameter = pickle.load(f)

        opt = history["opt"]

        SGDL_epochs = SGDL_opt_parameter['epochs']
        
        SGDL_time  = history['time'][-1]
        MGDL_time =  0
        MGDL_epochs = 0
        for i in range(0, len(MGDL_nn_parameter["mul_layers_dims"])):       
            MGDL_time = MGDL_time + trained_variable['train_time'][i][-1]
            MGDL_epochs = MGDL_epochs + MGDL_opt_parameter['epochs'][i]           
        
        SGDL_xaxis = np.arange(0, SGDL_epochs) * (SGDL_time / SGDL_epochs)
        MGDL_xaxis = np.arange(0, MGDL_epochs)* (MGDL_time / MGDL_epochs)
        
        single_train_costs = history["train_costs"]
        mul_train_costs = list(itertools.chain.from_iterable(trained_variable['train_costs']))

        # if opt.NORM_K == 1.0 or opt.NORM_K ==5 or opt.NORM_K == 10 or opt.NORM_K == 50:    
        axs[0].plot(SGDL_xaxis, single_train_costs[:SGDL_epochs], label='freq = {}'.format(int(opt.AMP_Z)))
        axs[1].plot(MGDL_xaxis, mul_train_costs[:MGDL_epochs], label='freq = {}'.format(int(opt.AMP_Z)))


    axs[0].set_xlabel("Training time [s]")
    axs[0].set_ylabel("Training loss")
    axs[0].set_ylim(tminnum, tmaxnum)
    axs[0].legend()
    axs[0].set_title('SGDL')

    axs[1].set_xlabel("Training time [s]")
    axs[1].set_ylabel("Training loss")
    axs[1].set_ylim(tminnum, tmaxnum)
    axs[1].legend()
    axs[1].set_title('MGDL')

    plt.tight_layout()
    plt.show()