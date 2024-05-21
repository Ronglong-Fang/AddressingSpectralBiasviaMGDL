
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



def results_analysis(fullfilename):

    with open(fullfilename, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f)     
    


    # print(data["opt"])
    # print(data["train_X"].shape)
    # print(data["test_X"].shape)
    
    print("###########################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)
    
    
    opt = history['opt']
    data = get_mnist(opt)
    print(len(history["parameters"]))
    predict_test_Y, _= singlegrade_model_forward(data['test_X'], nn_parameter["layers_dims"],
                                                 history["parameters"][0], nn_parameter["activation"],
                                                 nn_parameter["sinORrelu"])
    
    
    
    test_rse = rse(data['test_Y'], predict_test_Y)
    cleantest_rse = rse(data['cleantest_Y'], predict_test_Y)
    
    
    


    


    print('train_rse is {}, validation_rses is {}'.format(history['train_rses'][-1], history['validation_rses'][-1]))
    print('test_rse is {}, clean_test_rses is {}'.format(test_rse, cleantest_rse))

    print('the train time is {}'.format(history["time"][-1])) 



        
    return


