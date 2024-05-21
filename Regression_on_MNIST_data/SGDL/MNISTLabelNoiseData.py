import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


def get_mnist(opt):
    
    (train_X, Y), (test_X, tY) = mnist.load_data()
    train_Y = np.zeros((Y.shape[0], Y.max() + 1))
    test_Y = np.zeros((tY.shape[0], tY.max() + 1))
    train_Y[np.arange(Y.shape[0]), Y] =  1
    test_Y[np.arange(tY.shape[0]), tY] =  1
    
    train_X = train_X.reshape(train_X.shape[0], 784)/255
    test_X = test_X.reshape(test_X.shape[0], 784)/255

    
    train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, random_state=0, test_size=0.25, shuffle=True)      
        

    cleantrain_Y = train_Y
    cleanvalidation_Y = validation_Y
    cleantest_Y = test_Y
    #add noise to the train, validation, test label
    if opt.NOISE:
        train_Y = train_Y + train_Y * np.reshape(opt.AMP_Z * np.sin(2 * np.pi * opt.NORM_K * np.linalg.norm(train_X, axis=1)), (-1, 1))
        validation_Y = validation_Y + validation_Y * np.reshape(opt.AMP_Z * np.sin(2 * np.pi * opt.NORM_K * np.linalg.norm(validation_X, axis=1)), (-1, 1))
        test_Y = test_Y + test_Y * np.reshape(opt.AMP_Z * np.sin(2 * np.pi * opt.NORM_K * np.linalg.norm(test_X, axis=1)), (-1, 1))

    
        
        
    train_X = train_X.T
    train_Y = train_Y.T
    cleantrain_Y = cleantrain_Y.T
    
    validation_X = validation_X.T
    validation_Y = validation_Y.T
    cleanvalidation_Y = cleanvalidation_Y.T
    
    test_X = test_X.T
    test_Y = test_Y.T
    cleantest_Y = cleantest_Y.T
    
    data = {}
    data['opt'] = opt   
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['cleantrain_Y'] = cleantrain_Y
    
    data['validation_X'] = validation_X
    data['validation_Y'] = validation_Y
    data['cleanvalidation_Y'] = cleanvalidation_Y
    
    data['test_X'] = test_X
    data['test_Y'] = test_Y
    data['cleantest_Y'] = cleantest_Y    
    
    
          
    return data





