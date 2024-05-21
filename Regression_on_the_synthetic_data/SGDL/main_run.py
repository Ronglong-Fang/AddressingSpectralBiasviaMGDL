from singlegrade_dnn_main import single_dnn_main 

Amptype = "constant"                                              # choose the type of amplitude from: constant, decrease, increase, vary
SGD = False                                                       # if use stochastic method in Adam, then SGD is 'True' and set minibatch size
                                                                  # if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
mini_batch_size = False                                           # minibatch size                                
layers_dims = [1, 256, 256, 256, 256, 256, 256, 256, 256, 1]      # this is the structure for SGDL
epochs = 30000                                                    # the number of training epoch 
#epochs = 30

MAX_learning_rate = [1e-3]                                       # the maximum learning rate, denote as t_max in the paper
MIN_learning_rate = [1e-4]                                       # the minimum learning rate, denote as t_min in the paper

for max_learning_rate in MAX_learning_rate:
    for min_learning_rate in MIN_learning_rate:        
        single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, Amptype, SGD)

