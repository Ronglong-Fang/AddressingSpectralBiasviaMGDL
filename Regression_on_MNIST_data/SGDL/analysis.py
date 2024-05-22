
import sys
import os.path
import shutil
import os
sys.path.append('/home/rfang002/envs/default-tensorflow-cpu-2.6.0/lib/python3.10/site-packages')
import seaborn as sns
from MNISTLabelNoiseData import get_mnist
sns.set()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from singlegrade_dnn_regression import singlegrade_model_forward, rse 


def results_analysis_bestvalloss(SGDLfile):

    with open(SGDLfile, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f) 
        
    print('train_rse is {}, validation_rses is {}'.format(history['train_rses'][-1], history['validation_rses'][-1]))
    
    

    opt = history["opt"]
    SGDL_epochs = opt_parameter['epochs']
    SGDL_time  = history['time'][-1]
    
    SGDL_xaxis = np.arange(0, SGDL_epochs) * (SGDL_time / SGDL_epochs)
    single_validation_costs = history["validation_costs"]
    single_train_costs = history["train_costs"]


    plt.plot(SGDL_xaxis, single_validation_costs[:SGDL_epochs], label='validation loss')
    plt.plot(SGDL_xaxis, single_train_costs[:SGDL_epochs],  label='train loss')
    plt.xlabel("Training time [s]")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('SGDL: $(\\beta, \kappa) = ({}, {})$'.format(int(opt.AMP_Z), int(opt.NORM_K)))
    plt.tight_layout()
    plt.show()
    
    
def test_model(SGDLfile):
    
    with open(SGDLfile, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f)
   

    opt = history['opt']
    data = get_mnist(opt)
    predict_test_Y, _= singlegrade_model_forward(data['test_X'], nn_parameter["layers_dims"],
                                                 history["parameters"][0], nn_parameter["activation"],
                                                 nn_parameter["sinORrelu"])
    
    
    test_rse = rse(data['test_Y'], predict_test_Y)
    cleantest_rse = rse(data['cleantest_Y'], predict_test_Y)
    print('SGDL test rse on testing data is {}'.format(cleantest_rse))