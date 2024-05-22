from singlegrade_dnn_main import single_dnn_main 

amp = 1                                                              # the parameter beta in \tau_{beta, kappa}, chosen from {0.5, 1, 3}                   
FREQ = [5]                                                           # the parameter kappa in \tau_{beta, kappa}, chosen from {1, 5, 10, 50}

SGD = False                                                          # if use stochastic method in Adam, then SGD is 'True' and set minibatch size
                                                                     # if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
mini_batch_size = False                                              # minibatch size                                           
layers_dims = [784, 128, 128, 128, 128, 128, 128, 10]                # this is the structure for SGDL
#epochs = 2000                                                       # the number training epoch in each grade
epochs = 30  

MAX_learning_rate = [1e-3]                                           # the maximum learning rate, denote as t_max in the paper                                      
MIN_learning_rate = [1e-4]                                            # the minimum learning rate, denote as t_min in the paper

for freq in FREQ:
    for max_learning_rate in MAX_learning_rate:
        for min_learning_rate in MIN_learning_rate:        
            single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, freq, amp)

