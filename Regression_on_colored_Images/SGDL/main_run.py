from sgdlmodel import train_model, data_setup
from argparse import Namespace


###################################Cat###########################
image = 'Cat'
Learning_rate = [5e-3]
#################################################################

# ##################################Sea############################
# image = 'Sea'
# Learning_rate = [1e-3]
# #################################################################

# ##################################Building########################
# image = 'Building'
# Learning_rate = [1e-3]
# ##################################################################

Learning_rate = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
Num_layers = [12]

for learning_rate in Learning_rate:
    for num_layer in Num_layers:
        opt = Namespace()
        opt.epoch = 10000
        opt.learning_rate = learning_rate
        opt.num_layers = num_layer
        opt.num_channels = 256
        opt.interval = 25
        opt.image = image

        test_data, train_data = data_setup(opt)
        
        history_dict = train_model(opt, train_data, test_data)
