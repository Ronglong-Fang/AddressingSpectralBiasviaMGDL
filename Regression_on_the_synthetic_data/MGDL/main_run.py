from multigrade_dnn_main import multi_grade_dnn 


Amptype = 'decrease'                                  # choose the type of amplitude from: constant, decrease, increase, vary
SGD = False                                            # if use stochastic method in Adam, then SGD is 'True' and set minibatch size
                                                      # if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
mini_batch = False                                    # minibatch size
mul_layers_dims = [[1, 256, 256, 1], [256, 256, 256, 1], [256, 256, 256, 1], [256, 256, 256, 1]] # this is the structure for MGDL 
                             
#mul_epochs = [30000, 30000, 30000, 30000]             # the number training epoch in each grade
mul_epochs = [300, 300, 300, 300]  

MAX_learning_rate = [1e-3]                           # the maximum learning rate, denote as t_max in the paper
MIN_learning_rate = [1e-4]                           # the minimum learning rate, denote as t_min in the paper

for max_learning_rate in MAX_learning_rate:
    for min_learning_rate in MIN_learning_rate:
        multi_grade_dnn(max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, Amptype, SGD, mini_batch)

