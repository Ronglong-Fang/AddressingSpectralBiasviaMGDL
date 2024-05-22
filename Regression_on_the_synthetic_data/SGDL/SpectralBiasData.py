import numpy as np
import matplotlib.pyplot as plt

def Spectral_bias_constantORincrease_amplituide(opt):
    """
    generate sample train data and test data
    this example is from the paper : On the Spectral Bias of Neural Networks

    Returns
    -------
    None.

    """
    train_X = np.reshape(np.arange(0, 1, 1./opt.ntrain), (1, opt.ntrain))
    train_Y = np.zeros((1, opt.ntrain))    
    
    random_state = np.random.RandomState(seed=0)
    validation_X = random_state.uniform(low=0, high=1, size=(1,opt.nvalidation))
    validation_Y = np.zeros((1, opt.nvalidation))     
    
    random_state = np.random.RandomState(seed=1)
    test_X = random_state.uniform(low=0, high=1, size=(1, opt.ntest))
    test_Y = np.zeros((1, opt.ntest))    
    
    for i in range(len(opt.kappa)):
        train_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*train_X + opt.phi[i])
        validation_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*validation_X + opt.phi[i])
        test_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*test_X + opt.phi[i])
     
    
    
    data = {}
    data['opt'] = opt    
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['validation_X'] = validation_X
    data['validation_Y'] = validation_Y
    data['test_X'] = test_X
    data['test_Y'] = test_Y
    
    
    plt.plot(train_X.T, train_Y.T)
    plt.title('training data')
    plt.show()

#     plt.scatter(validation_X.T, validation_Y.T)
#     plt.title('validation data')
#     plt.show()
    
#     plt.scatter(test_X.T, test_Y.T)
#     plt.title('testing data')
#     plt.show()
    
    
    frq, fftyt = fft(train_Y)    
    plt.plot(frq, fftyt)
    plt.xlim(0, 200)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.show()
    
    return data


def Spectral_bias_vary_amplituide(opt):
    """
    generate sample train data and test data
    this example is from the paper : On the Spectral Bias of Neural Networks

    Returns
    -------
    None.

    """
    train_X = np.reshape(np.arange(0, 1, 1./opt.ntrain), (1, opt.ntrain))
    train_Y = np.zeros((1, opt.ntrain))    
    
    random_state = np.random.RandomState(seed=0)
    validation_X = random_state.uniform(low=0, high=1, size=(1,opt.nvalidation))
    validation_Y = np.zeros((1, opt.nvalidation))     
    
    random_state = np.random.RandomState(seed=1)
    test_X = random_state.uniform(low=0, high=1, size=(1, opt.ntest))
    test_Y = np.zeros((1, opt.ntest))    


    Ax = []       
    for i in range(len(opt.kappa)):
        Ax.append(lambda X: np.exp(-X) * np.cos(i * X))

    for i in range(len(opt.kappa)):
        train_Y += np.multiply(Ax[i](train_X), np.sin(2*np.pi*opt.kappa[i]*train_X + opt.phi[i]))
        validation_Y += np.multiply(Ax[i](validation_X),  np.sin(2*np.pi*opt.kappa[i]*validation_X + opt.phi[i]))
        test_Y += np.multiply(Ax[i](test_X), np.sin(2*np.pi*opt.kappa[i]*test_X + opt.phi[i]))

        
    data = {}
    data['opt'] = opt
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['validation_X'] = validation_X
    data['validation_Y'] = validation_Y
    data['test_X'] = test_X
    data['test_Y'] = test_Y
    
    plt.plot(train_X.T, train_Y.T)
    plt.title('training data')
    plt.show()

#     plt.scatter(validation_X.T, validation_Y.T)
#     plt.title('validation data')
#     plt.show()
    
#     plt.scatter(test_X.T, test_Y.T)
#     plt.title('testing data')
#     plt.show()
    
    frq, fftyt = fft(train_Y)
    
    plt.plot(frq, fftyt)
    plt.xlim(0, 200)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.show()
       
    return data



def fft(yt):
    
    n = len(np.squeeze(yt)) # length of the signal
    frq = np.arange(n)
    frq = frq[range(n//2)] # one side frequency range
    # -------------
    FFTYT = np.squeeze(np.fft.fft(yt)/n) # fft computing and normalization

    FFTYT = FFTYT[range(n//2)]
    fftyt = abs(FFTYT)


    plt.plot(frq, fftyt)
    plt.xlim(0, 200)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.show()

    return frq, fftyt




    



