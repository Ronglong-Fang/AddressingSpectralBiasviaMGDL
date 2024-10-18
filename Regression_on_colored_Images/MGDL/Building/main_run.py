from argparse import Namespace
from mgdlmodel import MGDLmodel


learning_rate1 = {'grade1': 5e-3, 'grade2': 5e-3, 'grade3': 1e-3, 'grade4': 1e-3}
Learning_rate = [learning_rate1]

for learning_rate in Learning_rate:

    opt = Namespace()
    opt.epoch = {'grade1': 10000, 'grade2': 10000, 'grade3': 10000, 'grade4': 10000}
    opt.num_channel = {'grade1': 256, 'grade2': 256, 'grade3': 256, 'grade4': 256}
    opt.num_layer = {'grade1': 3, 'grade2': 3, 'grade3': 3, 'grade4': 3}
    opt.epsilon = False
    opt.grade = 4
    opt.interval = 100
    opt.learning_rate =  learning_rate
    
    print(opt)
    
    MGDLmodel(opt)