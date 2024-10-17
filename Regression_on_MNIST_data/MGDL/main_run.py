from multigrade_dnn_main import multi_grade_dnn 
 
amp = 1                                                                                      # the parameter beta in \tau_{beta, kappa}, chosen from {0.5, 1, 3}
FREQ = [5]                                                                                   # the parameter kappa in \tau_{beta, kappa}, chosen from {1, 5, 10, 50}
SGD = False                                                                                  # if use stochastic method in Adam, then SGD is 'True' and set minibatch size
                                                                                             # if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
mini_batch = False                                                                           # minibatch size 


##############################################Structure 1##################################
mul_layers_dims = [[784, 128, 128, 10], [128, 128, 128, 10], [128, 128, 128, 10]]            # this is the structure for MGDL
mul_epochs = [2000, 2000, 2000]                                                              # the number training epoch in each grade
nn_parameter["lambd_W"] = [0, 0, 0]                                                          # the L2 regularization  for the weight matrix. 
                                                                                             # In the paper, we did not apply L2 regularization, 
                                                                                             # thereby set to 0 for this parameter
############################################################################################



#############################################Structure 2####################################
#mul_layers_dims = [[784, 128, 10], [128, 128, 10], [128, 128, 10], [128, 128, 10], [128, 128, 10], [128, 128, 10]] 
#mul_epochs = [2000, 2000, 2000, 2000, 2000, 2000]
#mul_lambd_W = [0, 0, 0, 0, 0, 0]
############################################################################################

MAX_learning_rate = [1e-3]                                                                   # the maximum learning rate, denote as t_max in the paper
MIN_learning_rate = [1e-4]                                                                   # the minimum learning rate, denote as t_min in the paper

for freq in FREQ:
    for max_learning_rate in MAX_learning_rate:
        for min_learning_rate in MIN_learning_rate:
            multi_grade_dnn(max_learning_rate, min_learning_rate, mul_layers_dims, mul_lambd_W, mul_epochs, SGD, mini_batch, freq, amp)
