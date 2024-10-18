from sgdlmodel import train_model, data_setup
from argparse import Namespace

test_data, train_data = data_setup()

Learning_rate = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
Num_layers = [6, 8, 9, 10, 12, 15, 18]

for learning_rate in Learning_rate:
    for num_layer in Num_layers:
        opt = Namespace()
        opt.epoch = 10000
        opt.learning_rate = learning_rate
        opt.num_layers = num_layer
        opt.num_channels = 256
        opt.interval = 25
        
        history_dict = train_model(opt, train_data, test_data)