from multigrade_dnn_main import multi_grade_dnn 

Amptype = 'decrease'                                   # choose the type of amplitude from: constant, decrease, vary, increase
SGD = False                                            # if use stochastic method in Adam, then SGD is 'True' and set minibatch size
                                                       # if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
mini_batch = False                                     # minibatch size

####################################structure for setting: constant, decrease, vary################################
mul_layers_dims = [[1, 256, 256, 1], [256, 256, 256, 1], [256, 256, 256, 1], [256, 256, 256, 1]]                            
mul_epochs = [30000, 30000, 30000, 30000]              # the number training epoch in each grade
lambd_W = [0, 0, 0, 0]                                 # the L2 regularization  for the weight matrix. 
                                                       # In the paper, we did not apply L2 regularization, 
                                                       # thereby set to 0 for this parameter
####################################################################################################################

####################################structure for setting: increase##################################################
#mul_layers_dims = [[1, 256, 256, 1], [256, 256, 256, 1], [256, 256, 256, 1], [256, 256, 256, 1]. [256, 256, 256, 1]]                            
#mul_epochs = [30000, 30000, 30000, 30000, 30000]              
#lambd_W = [0, 0, 0, 0, 0]  
####################################################################################################################

MAX_learning_rate = [1e-3]                           # the maximum learning rate, denote as t_max in the paper
MIN_learning_rate = [1e-4]                           # the minimum learning rate, denote as t_min in the paper

for max_learning_rate in MAX_learning_rate:
    for min_learning_rate in MIN_learning_rate:
        multi_grade_dnn(max_learning_rate, min_learning_rate, mul_layers_dims, lambda_W, mul_epochs, Amptype, SGD, mini_batch)

