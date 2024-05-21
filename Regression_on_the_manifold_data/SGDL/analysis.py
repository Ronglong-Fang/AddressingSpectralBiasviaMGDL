
import sys
sys.path.append('/home/rfang002/envs/default-tensorflow-cpu-2.6.0/lib/python3.10/site-packages')
import seaborn as sns
sns.set()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from ManifoldsData import fft
from singlegrade_dnn_regression import singlegrade_model_forward, rse 

def merge_all_cost(file1, file2, file3, file4):
    
    with open(file1, 'rb') as f:
        trained_variable_1, nn_parameter_1, opt_parameter_1 = pickle.load(f)

    with open(file2, 'rb') as f:
        trained_variable_2, nn_parameter_2, opt_parameter_2 = pickle.load(f)

    with open(file3, 'rb') as f:
        trained_variable_3, nn_parameter_3, opt_parameter_3 = pickle.load(f)    

    with open(file4, 'rb') as f:
        trained_variable_4, nn_parameter_4, opt_parameter_4 = pickle.load(f)
    
    ITER = (np.linspace(0, 50000, 501)).astype(int)
    #ITER = (np.linspace(0, 70000, 701)).astype(int)
    COST_1 = trained_variable_1['costs']
    COST_2 = trained_variable_2['costs']
    COST_3 = trained_variable_3['costs']
    COST_4 = trained_variable_4['costs']
    
    plt.figure(figsize=(10, 8))
    plt.plot(ITER, COST_1, label='L=0')
    plt.plot(ITER, COST_2, label='L=4') 
    plt.plot(ITER, COST_3, label='L=10')
    plt.plot(ITER, COST_4, label='L=16') 
    plt.xlabel('Training Iteration')
    plt.ylabel('Train loss')
    # plt.yscale('log')
    plt.ylim([-0.1, 2.5])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def merge_all_cost_all(file1, file2, file3, file4, file5, file6):
    
    with open(file1, 'rb') as f:
        trained_variable_1, nn_parameter_1, opt_parameter_1 = pickle.load(f)

    with open(file2, 'rb') as f:
        trained_variable_2, nn_parameter_2, opt_parameter_2 = pickle.load(f)

    with open(file3, 'rb') as f:
        trained_variable_3, nn_parameter_3, opt_parameter_3 = pickle.load(f)    

    with open(file4, 'rb') as f:
        trained_variable_4, nn_parameter_4, opt_parameter_4 = pickle.load(f)
        
    with open(file5, 'rb') as f:
        trained_variable_5, nn_parameter_5, opt_parameter_5 = pickle.load(f)    

    with open(file6, 'rb') as f:
        trained_variable_6, nn_parameter_6, opt_parameter_6 = pickle.load(f)
    
    #ITER = (np.linspace(0, 70000, 701)).astype(int)
    ITER = (np.linspace(0, 150000, 1501)).astype(int)
    
    COST_1 = trained_variable_1['costs']
    COST_2 = trained_variable_2['costs']
    COST_3 = trained_variable_3['costs']
    COST_4 = trained_variable_4['costs']
    COST_5 = trained_variable_5['costs']
    COST_6 = trained_variable_6['costs']
    
    plt.figure(figsize=(10, 8))
    plt.plot(ITER, COST_1, label='structure (2.28)')
    plt.plot(ITER, COST_2, label='structure (2.29)') 
    plt.plot(ITER, COST_3, label='structure (2.30)')
    plt.plot(ITER, COST_4, label='structure (2.31)')
    plt.plot(ITER, COST_5, label='structure (2.32)')
    plt.plot(ITER, COST_6, label='structure (2.33)') 
    plt.xlabel('Training Iteration')
    plt.ylabel('Train loss')
    plt.yscale('log')
    plt.ylim([1e-6, 1e1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def results_analysis(fullfilename):

    with open(fullfilename, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f)

    data = history["data"]       
    


    # print(data["opt"])
    # print(data["train_X"].shape)
    # print(data["test_X"].shape)
    
    print("###########################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)
    
    test_predict_Y, _ = singlegrade_model_forward(data['test_X'], nn_parameter['layers_dims'], history['parameters'][-1], nn_parameter["activation"], nn_parameter["sinORrelu"])
    test_rse = rse(data['test_Y'], test_predict_Y)

    


    print('train_rse is {}, validation_rses is {}, test_rse is {}'.format(history['train_rses'][-1], history['validation_rses'][-1], test_rse))

    


    

    print('the train time is {}'.format(history["time"][-1])) 



    plt.plot(history["REC_FRQ_iter"], np.array(history["train_costs"]), label="train cost")
    plt.plot(history["REC_FRQ_iter"], np.array(history["validation_costs"]), label="validation cost")
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    
    plt.plot(history['train_rses'], 'b--', label='train rse')
    plt.plot(history['validation_rses'], 'r--', label='validation rse')
    plt.yscale('log')
    plt.xlabel('train iteration')
    plt.ylabel('rse')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.legend()
    plt.show()


    for i in range(0, len(history["REC_FRQ_iter"]), 100):
        print('at epochs {}, train rse {}, validation rse {}'.format(history["REC_FRQ_iter"][i], history['train_rses'][i], history['validation_rses'][i]))
        print('train time at epochs {} is {}'.format(history["REC_FRQ_iter"][i], history["time"][i]))


     
        
    return


def compute_spectra(history):
    train_predict_record =  history["train_predict_record"]
    
    dynamics = []
    #compute compute spectra
    for yt in train_predict_record:
        frq, fftyt = fft(yt)
        dynamics.append(fftyt)


    ylicks = history["REC_FRQ_iter"]
    ylicks.reverse()

    return np.array(frq), np.array(dynamics), np.array(ylicks)
        
def plot_spectral_dynamics(fullfilename):
    
    
    with open(fullfilename, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f) 

    
    data = history["data"]
    frq, dynamics, yticks = compute_spectra(history)  


    # Select the frequencies which are present in the target spectrum
    freq_selected = dynamics[:, np.sum(frq.reshape(-1, 1) == np.array(data['opt'].kappa).reshape(1, -1), axis=-1, dtype='bool')]
    # Normalize by the amplitude. Remember to account for the fact that the measured spectra 
    # are single-sided (positive freqs), so multiply by 2 accordingly
    norm_dynamics = 2 * freq_selected / np.array(data['opt'].alpha).reshape(1, -1)

     
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    # plt.title("Evolution of Frequency Spectrum (Increasing Amplitudes)")
    sns.heatmap(norm_dynamics[::-1], 
                xticklabels = data['opt'].kappa, 
                yticklabels=[(ytick if ytick%10000 == 0 else '') for ytick in yticks], 
                vmin=0., vmax=1.0, 
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Training Iteration")
    plt.show()     

    
    return



def plot_spectral_dynamics_vary(fullfilename):
    
    
    with open(fullfilename, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f) 



    all_dynamics = []
    data = history["data"]
    frq, dynamics, yticks = compute_spectra(history)  
    all_dynamics.append(dynamics)

    frq, fftyt =  fft(data["train_Y"])
    new_fftyt = np.zeros([len(data['opt'].kappa)])
    for l in range(len(data['opt'].kappa)):
        for j in range(len(frq)):
            if frq[j]>data['opt'].kappa[l]-5 and frq[j]< data['opt'].kappa[l]+5: 
                new_fftyt[l] = new_fftyt[l] + fftyt[j]    
    # Average dynamics over multiple history    
    mean_dynamics = np.array(all_dynamics).mean(0)
    # Average dynamics over multiple history    
    new_mean_dynamics = np.zeros([mean_dynamics.shape[0], len(data['opt'].kappa)])
    for l in range(len(data['opt'].kappa)):
        for j in range(len(frq)):
            if frq[j]>data['opt'].kappa[l]-5 and frq[j]< data['opt'].kappa[l]+5: 
                new_mean_dynamics[:, l] = new_mean_dynamics[:, l] + mean_dynamics[:, j]
        
    # are single-sided (positive freqs), so multiply by 2 accordingly
    norm_dynamics = new_mean_dynamics/new_fftyt.reshape(1, -1)
     
    # Plot heatmap
    
    plt.figure(figsize=(10, 8))
    # plt.title("Evolution of Frequency Spectrum (Increasing Amplitudes)")
    sns.heatmap(norm_dynamics[::-1], 
                xticklabels = data['opt'].kappa, 
                yticklabels=[(ytick if ytick%10000 == 0 else '') for ytick in yticks], 
                vmin=0., vmax=1.0, 
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Training Iteration")
    plt.show()     

    
    return 

        

