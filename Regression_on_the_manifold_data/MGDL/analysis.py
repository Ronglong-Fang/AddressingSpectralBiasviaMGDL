# -*- coding: utf-8 -*-
"""
Created on Wed 12/31 2023

@author: rfang002
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from ManifoldsData import fft
import os.path
import itertools
import sys
from multigrade_dnn_model import multigrade_dnn_model_predict
sys.path.append('/home/rfang002/envs/default-tensorflow-gpu-2.10.0/lib/python3.7/site-packages')
import seaborn as sns
sns.set()



    
def results_analysis(fullfilename, figure):

    with open(fullfilename, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)
        
    
    print("################################################################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)
    
    print(trained_variable["data"]['opt'])


    num_iter = trained_variable["REC_FRQ_iter"]
    data = trained_variable["data"]

    train_rse = []
    validation_rse = []
    train_time = []

    
    total_time =  0

    for i in range(0, len(nn_parameter["mul_layers_dims"])):       
        total_time = total_time + trained_variable['train_time'][i][-1]
        train_rse.append(trained_variable['train_rse'][i][-1])
        validation_rse.append(trained_variable['validation_rse'][i][-1])
        train_time.append(trained_variable['train_time'][i][-1])
        
    test_rse = multigrade_dnn_model_predict(nn_parameter, opt_parameter, trained_variable)

    print('the train rse for each grade is {}'.format(train_rse))
    print('the validation rse for each grade is {}'.format(validation_rse))
    print('the test rse for each grade is {}'.format(test_rse))
    print('the train times for each grade is {}'.format(train_time))
    print('the total train times is {}'.format(total_time))





    for i in range(0, len(nn_parameter["mul_layers_dims"])):
        
        num_iter = trained_variable["REC_FRQ_iter"][i]
        plt.plot(num_iter,  trained_variable['train_costs'][i], label='train cost')
        plt.plot(num_iter,  trained_variable['validation_costs'][i], label='validation cost')
        plt.yscale('log')
        plt.xlabel("Training Iteration")
        plt.ylabel('costs')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()

        plt.plot(num_iter, trained_variable['train_rse'][i], 'b--', label='train rse')
        plt.plot(num_iter, trained_variable['validation_rse'][i], 'r--', label='validation rse')
        plt.yscale('log')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.title('train and validation rse for grade {}'.format(i+1))   
        plt.show()
        
        print('the train time is {}'.format(trained_variable['train_time'][i][-1]))
        

        train_predict_record =  trained_variable["train_predict_record"][i]
        validation_predict_record =  trained_variable["validation_predict_record"][i]
        for l in range(0, len(train_predict_record), 100):
            if i==0:
                predict = train_predict_record[l]
            else:
                predict = train_predict_record[l]+trained_variable["train_predict"][i-1]

            if figure:
                fig = plt.scatter(data['train_X'][0,:].T, data['train_X'][1,:].T, c=data['train_Y'].T)    
                plt.colorbar(fig)
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.title('predict at {} iterations'.format(l*opt_parameter["REC_FRQ"]))     
                plt.show()  


                fig = plt.scatter(data['train_X'][0,:].T, data['train_X'][1,:].T, c=np.abs(data['train_Y'].T - trained_variable["train_predict"][i].T),vmin=0, vmax=3)
                plt.colorbar(fig)
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.title('the error of multi-grade dnn model predict at grade {} on train data'.format(i+1))       
                plt.show()


    return 






def compute_spectra(train_predict_record, pre_train):
    
    dynamics = []
    #compute compute spectra
    for yt in train_predict_record:
        frq, fftyt = fft(yt+pre_train)
        dynamics.append(fftyt)

    return np.array(frq), np.array(dynamics)
        
        

def plot_spectral_dynamics(fullfilename, Amptype):
    
    with open(fullfilename, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)

    data = trained_variable['data']

    ITER = []
    TRAIN_predict_record = []
    for i in range(0, len(nn_parameter["mul_layers_dims"])):
        if i==0:
            ite = trained_variable["REC_FRQ_iter"][i][1:]
            train_predict_record = trained_variable["train_predict_record"][i]
        else:
            ite = trained_variable["REC_FRQ_iter"][i]+np.sum(opt_parameter["epochs"][:i])
            train_predict_record = []
            for s in range(1, len(trained_variable["train_predict_record"][i])):
                train_predict_record.append(trained_variable["train_predict_record"][i][s]+trained_variable["train_predict"][i-1])
        
        ITER.extend(ite)  
        TRAIN_predict_record.extend(train_predict_record)
    
    ITER = reversed(ITER)

    all_dynamics = []
    
    if Amptype == 'increase':
        frq, dynamics = compute_spectra(TRAIN_predict_record, 0)  
        all_dynamics.append(dynamics)
    
        # Average dynamics over multiple history    
        mean_dynamics = np.array(all_dynamics).mean(0)
        # Select the frequencies which are present in the target spectrum
        freq_selected = mean_dynamics[:, np.sum(frq.reshape(-1, 1) == np.array(data['opt'].kappa).reshape(1, -1), axis=-1, dtype='bool')]
        # Normalize by the amplitude. Remember to account for the fact that the measured spectra 
        # are single-sided (positive freqs), so multiply by 2 accordingly
        norm_dynamics = 2 * freq_selected / np.array(data['opt'].alpha).reshape(1, -1)
    elif Amptype == 'vary':
        frq, fftyt =  fft(data["train_Y"])
        new_fftyt = np.zeros([len(data['opt'].kappa)])
        for l in range(len(data['opt'].kappa)):
            for j in range(len(frq)):
                if frq[j]>data['opt'].kappa[l]-10 and frq[j]< data['opt'].kappa[l]+10: 
                    new_fftyt[l] = new_fftyt[l] + fftyt[j]

        if i==0:
            pre_train = 0
        else:
            pre_train = trained_variable["train_predict"][i-1]

        frq, dynamics = compute_spectra(TRAIN_predict_record, 0)  
        all_dynamics.append(dynamics)

        ylicks = trained_variable["REC_FRQ_iter"]
 
        # Average dynamics over multiple history    
        mean_dynamics = np.array(all_dynamics).mean(0)
        print(mean_dynamics.shape)
        new_mean_dynamics = np.zeros([mean_dynamics.shape[0], len(data['opt'].kappa)])
        for l in range(len(data['opt'].kappa)):
            for j in range(len(frq)):
                if frq[j]>data['opt'].kappa[l]-10 and frq[j]< data['opt'].kappa[l]+10: 
                    new_mean_dynamics[:, l] = new_mean_dynamics[:, l] + mean_dynamics[:, j]
            
        # are single-sided (positive freqs), so multiply by 2 accordingly
        norm_dynamics = new_mean_dynamics/new_fftyt.reshape(1, -1)


    

    sns.heatmap(norm_dynamics[::-1], 
                xticklabels = [(kappa if kappa%100 == 0 else '') for kappa in (data['opt'].kappa).astype(int)], 
                yticklabels=[('G '+str(int(ytick/30000)) if (ytick)%30000 == 0 else '') for ytick in ITER], 
                vmin=0., vmax=1.0, 
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("MGDL")
    plt.show() 
    
    
    
def plot_spectral_each_grade(fullfilename, Amp_Y):
    with open(fullfilename, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)
    
    FFTYT = [] 
    for i in range(0, len(nn_parameter["mul_layers_dims"])):
        predict = trained_variable["train_predict_record"][i][-1]
        frq, fftyt = fft(predict)
        FFTYT.append(fftyt)
        
        plt.plot(frq, fftyt)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.xlim(0, 400)
        plt.ylim(0, Amp_Y)
        plt.show()
        
    plt.plot(frq, FFTYT[0], linestyle='solid', label='G 1')
    plt.plot(frq, FFTYT[1], linestyle='solid', label='G 2')
    plt.plot(frq, FFTYT[2], linestyle='solid', label='G 3')
    plt.plot(frq, FFTYT[3], linestyle='solid', label='G 4')
    plt.xlim(0, 400)
    # plt.ylim()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    
    
    
    
    
def merge_all_cost_all(file1, file2):
    
    
    with open(file1, 'rb') as f:
        trained_variable_1, nn_parameter_1, opt_parameter_1 = pickle.load(f)

    with open(file2, 'rb') as f:
        trained_variable_2, nn_parameter_2, opt_parameter_2 = pickle.load(f)

    
    plt.figure(figsize=(6.4, 4.8))
    
    
    
    for i in range(0, len(nn_parameter_1["mul_layers_dims"])):
        if i==0:
            plt.plot(np.arange(30000*i, 30000*(i+1), 100), trained_variable_2['train_costs'][i][0:300], 'b-', label='training, 4')
            plt.plot(np.arange(30000*i, 30000*(i+1), 100), trained_variable_2['validation_costs'][i][0:300], 'b--', label='validation, 4')
        else:
            plt.plot(np.arange(30000*i, 30000*(i+1), 100), trained_variable_2['train_costs'][i][0:300], 'b-')
            plt.plot(np.arange(30000*i, 30000*(i+1), 100), trained_variable_2['validation_costs'][i][0:300], 'b--')    
    
    for i in range(0, len(nn_parameter_1["mul_layers_dims"])):
        if i==0:
            plt.plot(np.arange(30000*i, 30000*(i+1), 100), trained_variable_1['train_costs'][i][0:300], 'r-', label='training, 0')
            plt.plot(np.arange(30000*i, 30000*(i+1), 100), trained_variable_1['validation_costs'][i][0:300], 'r--', label='validation, 0')
        else:
            plt.plot(np.arange(30000*i, 30000*(i+1), 100), trained_variable_1['train_costs'][i][0:300], 'r-')
            plt.plot(np.arange(30000*i, 30000*(i+1), 100), trained_variable_1['validation_costs'][i][0:300], 'r--')




    # Adding vertical dashed lines to separate the grades visually
    plt.axvline(x=30000, color='k', linestyle=':')
    plt.axvline(x=60000, color='k', linestyle=':')
    plt.axvline(x=90000, color='k', linestyle=':')

    # Customizing x-axis tick0s and labels
    plt.xticks([0, 30000, 60000, 90000, 120000], ['0', '30000', '30000', '30000', '30000'])

    # Adding custom labels below the x-axis
    plt.text(15000, 0.5e-4, 'G 1', ha='center', va='center', transform=plt.gca().transData)
    plt.text(45000, 0.5e-4, 'G 2', ha='center', va='center', transform=plt.gca().transData)
    plt.text(75000, 0.5e-4, 'G 3', ha='center', va='center', transform=plt.gca().transData)
    plt.text(105000, 0.5e-4, 'G 4', ha='center', va='center', transform=plt.gca().transData)

    # Adjusting the bottom margin to make room for the labels
    plt.xlabel('Number of training epochs (MGDL)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.ylim([1e-4, 1e1])

    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()
