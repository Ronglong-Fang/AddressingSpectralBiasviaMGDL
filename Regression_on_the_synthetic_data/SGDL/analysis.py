import sys
sys.path.append('/home/rfang002/envs/default-tensorflow-cpu-2.6.0/lib/python3.10/site-packages')
import seaborn as sns
sns.set()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from SpectralBiasData import fft
from singlegrade_dnn_regression import singlegrade_model_forward, rse 



def results_analysis(fullfilename):

    with open(fullfilename, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f)

    data = history["data"]       

    
    print("###########################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)
    

    print('train_rse is {}, validation_rses is {}'.format(history['train_rses'][-1], history['validation_rses'][-1]))
    print('the train time is {}'.format(history["time"][-1])) 



    plt.plot(history["REC_FRQ_iter"], np.array(history["train_costs"]), label="train cost")
    plt.plot(history["REC_FRQ_iter"], np.array(history["validation_costs"]), label="validation cost")
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    


    for i in range(0, len(history["REC_FRQ_iter"]), 100):
        print('at epochs {}, train rse {}, validation rse {}'.format(history["REC_FRQ_iter"][i], history['train_rses'][i], history['validation_rses'][i]))
        print('train time at epochs {} is {}'.format(history["REC_FRQ_iter"][i], history["time"][i]))
        plt.plot(np.squeeze(data['train_X']), np.squeeze(data['train_Y']), label='true label')
        plt.plot(np.squeeze(data['train_X']), np.squeeze(history['train_predict_record'][i]), label='predict label')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.title('SGDL predict at {} iteration'.format(i*opt_parameter["REC_FRQ"]))
        plt.show()
        
    return


def test_model(fullfilename):
    with open(fullfilename, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f)

    data = history["data"]       

    
    test_predict_Y, _ = singlegrade_model_forward(data['test_X'], nn_parameter['layers_dims'], 
                                                  history['parameters'][-1], nn_parameter["activation"], 
                                                  nn_parameter["sinORrelu"])
    test_rse = rse(data['test_Y'], test_predict_Y)
    
    print('the test rse for each grade is {}'.format(test_rse))
                   
    data = history["data"]
    plt.scatter(np.squeeze(data['test_X']), np.squeeze(data['test_Y']), label='true test label')
    plt.scatter(np.squeeze(data['test_X']), np.squeeze(test_predict_Y), label='predict test label')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)  
    plt.title('SGDL predict on testing data')
    plt.show() 

                   
                   

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
    sns.heatmap(norm_dynamics[::-1], 
                xticklabels = [(kappa if kappa%50 == 0 else '') for kappa in (data['opt'].kappa).astype(int)], 
                yticklabels=[(ytick if ytick%10000 == 0 else '') for ytick in yticks], 
                vmin=0., vmax=1.0, 
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Number of training epochs (SGDL)")
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
    sns.heatmap(norm_dynamics[::-1], 
                xticklabels = [(kappa if kappa%50 == 0 else '') for kappa in (data['opt'].kappa).astype(int)], 
                yticklabels=[(ytick if ytick%10000 == 0 else '') for ytick in yticks], 
                vmin=0., vmax=1.0, 
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Number of training epochs (SGDL)")
    plt.show()

    
    return 

        

def plot_spectral(fullfilename, Amp_Y):
    with open(fullfilename, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)

    
    predict = trained_variable['train_predict_record'][-1]
    frq, fftyt = fft(predict)
    plt.plot(frq, fftyt)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.xlim(0, 200)
    plt.ylim(-0.01, Amp_Y+0.01)
    plt.show()    
    
