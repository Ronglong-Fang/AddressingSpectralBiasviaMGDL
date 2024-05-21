import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def Manifolds_regression_increase_amplituide(opt):
    """
    generate sample train data and test data
    this example is from the paper : On the Spectral Bias of Neural Networks

    Returns
    -------
    None.

    """
    trX = np.reshape(np.arange(0, 1, 1./opt.ntrain), (1, opt.ntrain))
    train_X = np.zeros((2, opt.ntrain)) 
    train_X[0, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*trX),  np.cos(2*np.pi*trX))
    train_X[1, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*trX),  np.sin(2*np.pi*trX))
    train_Y = np.zeros((1, opt.ntrain))    
    
    random_state = np.random.RandomState(seed=0)
    vX = random_state.uniform(low=0, high=1, size=opt.nvalidation)
    validation_X = np.zeros((2, opt.nvalidation)) 
    validation_X[0, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*vX),  np.cos(2*np.pi*vX))
    validation_X[1, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*vX),  np.sin(2*np.pi*vX))
    validation_Y = np.zeros((1, opt.nvalidation))     
    
    random_state = np.random.RandomState(seed=1)
    teX = random_state.uniform(low=0, high=1, size=opt.ntest)
    test_X = np.zeros((2, opt.ntest)) 
    test_X[0, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*teX),  np.cos(2*np.pi*teX))
    test_X[1, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*teX),  np.sin(2*np.pi*teX))
    test_Y = np.zeros((1, opt.ntest))    


    
    
    for i in range(len(opt.kappa)):
        train_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*trX + opt.phi[i])
        validation_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*vX + opt.phi[i])
        test_Y += opt.alpha[i]*np.sin(2*np.pi*opt.kappa[i]*teX + opt.phi[i])
           
    
    data = {}
    data['opt'] = opt
    data['trX'] = trX
    data['vX'] = vX
    data['teX'] = teX
    
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['validation_X'] = validation_X
    data['validation_Y'] = validation_Y
    data['test_X'] = test_X
    data['test_Y'] = test_Y
    
    plt.plot(train_X[0,:], train_X[1,:])
    plt.show()

    plt.plot(trX.T, train_Y.T)
    plt.show()

    plt.scatter(vX.T, validation_Y.T)
    plt.show()
    
    plt.scatter(teX.T, test_Y.T)
    plt.show()
    
    fft(train_Y)
    return data

def Manifolds_regression_vary_amplituide(opt):
    """
    generate sample train data and test data
    this example is from the paper : On the Spectral Bias of Neural Networks

    Returns
    -------
    None.

    """
    trX = np.reshape(np.arange(0, 1, 1./opt.ntrain), (1, opt.ntrain))
    train_X = np.zeros((2, opt.ntrain)) 
    train_X[0, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*trX),  np.cos(2*np.pi*trX))
    train_X[1, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*trX),  np.sin(2*np.pi*trX))
    train_Y = np.zeros((1, opt.ntrain))    
    
    random_state = np.random.RandomState(seed=0)
    vX = random_state.uniform(low=0, high=1, size=opt.nvalidation)
    validation_X = np.zeros((2, opt.nvalidation)) 
    validation_X[0, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*vX),  np.cos(2*np.pi*vX))
    validation_X[1, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*vX),  np.sin(2*np.pi*vX))
    validation_Y = np.zeros((1, opt.nvalidation))     
    
    random_state = np.random.RandomState(seed=1)
    teX = random_state.uniform(low=0, high=1, size=opt.ntest)
    test_X = np.zeros((2, opt.ntest)) 
    test_X[0, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*teX),  np.cos(2*np.pi*teX))
    test_X[1, :] = np.multiply(1+1/2*np.sin(2*np.pi*opt.L*teX),  np.sin(2*np.pi*teX))
    test_Y = np.zeros((1, opt.ntest))    


    Ax = []       
    for i in range(len(opt.kappa)):
        Ax.append(lambda X: np.exp(-X) * np.cos(i * X))

    for i in range(len(opt.kappa)):
        train_Y += np.multiply(Ax[i](trX), np.sin(2*np.pi*opt.kappa[i]*trX + opt.phi[i]))
        validation_Y += np.multiply(Ax[i](vX),  np.sin(2*np.pi*opt.kappa[i]*vX + opt.phi[i]))
        test_Y += np.multiply(Ax[i](teX), np.sin(2*np.pi*opt.kappa[i]*teX + opt.phi[i]))

        
    data = {}
    data['opt'] = opt
    data['trX'] = trX
    data['vX'] = vX
    data['teX'] = teX
    
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['validation_X'] = validation_X
    data['validation_Y'] = validation_Y
    data['test_X'] = test_X
    data['test_Y'] = test_Y
    
    plt.plot(trX.T, train_Y.T)
    plt.show()

    plt.scatter(vX.T, validation_Y.T)
    plt.show()
    
    plt.scatter(teX.T, test_Y.T)
    plt.show()
    
    fft(train_Y)
    
 
    
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
    plt.xlim(0, 400)
    plt.show()

    return frq, fftyt




    



