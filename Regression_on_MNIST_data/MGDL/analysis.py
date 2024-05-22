
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



def results_analysis_bestvalloss(MGDLfile):

    with open(MGDLfile, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)

    opt = trained_variable["opt"]
    
    train_rse = []
    validation_rse = []
    train_time = []

    MGDL_time =  0
    MGDL_epochs = 0
    for i in range(0, len(nn_parameter["mul_layers_dims"])):       
        MGDL_time = MGDL_time + trained_variable['train_time'][i][-1]
        MGDL_epochs = MGDL_epochs + opt_parameter['epochs'][i]    
        train_rse.append(trained_variable['train_rse'][i][-1])
        validation_rse.append(trained_variable['validation_rse'][i][-1])
        
    print('MGDL train rse for each grade is {}'.format(train_rse))
    print('MGDL validation rse for each grade is {}'.format(validation_rse))

    MGDL_xaxis = np.arange(0, MGDL_epochs)* (MGDL_time / MGDL_epochs)     
    MGDL_validation_costs = list(itertools.chain.from_iterable(trained_variable['validation_costs']))
    MGDL_train_costs = list(itertools.chain.from_iterable(trained_variable['train_costs']))

    plt.plot(MGDL_xaxis, MGDL_validation_costs[:MGDL_epochs], label='validation loss')
    plt.plot(MGDL_xaxis, MGDL_train_costs[:MGDL_epochs],  label='train loss')
    plt.xlabel("Training time [s]")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('MGDL: $(\\beta, \kappa) = ({}, {})$'.format(int(opt.AMP_Z), int(opt.NORM_K)))
    plt.tight_layout()
    plt.show()
    
    
def test_model(MGDLfile):
    
    with open(MGDLfile, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)

        
    cleantest_rse = multigrade_dnn_model_predict_cleandata(nn_parameter, opt_parameter, trained_variable)
    print('MGDL test rse one testing data is {}'.format(cleantest_rse))
    
    
    
    
