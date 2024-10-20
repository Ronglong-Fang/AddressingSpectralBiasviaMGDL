import jax.numpy as np
from jax import jit, grad, random
from jax.example_libraries import stax, optimizers
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os, imageio


#set up data
def data_setup(opt):

    if opt.image == 'Cat':
        # Download image, take a square crop from the center
        image_path = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    elif opt.image == 'Sea':
        image_path = 'image/sea.png'  # Replace with your image file path
    elif opt.image == 'Building': 
        image_path = 'image/building.png'  
        
    img = imageio.imread(image_path)[..., :3] / 255.

    print(f'the shape of image is: {np.shape(img)}')
    
    plt.imshow(img)
    plt.show()
    
    # Create input pixel coordinates in the unit square
    coords_x = np.linspace(0, 1, img.shape[0], endpoint=False)
    coords_y = np.linspace(0, 1, img.shape[1], endpoint=False)
    test_x = np.stack(np.meshgrid(coords_y, coords_x), -1)

    test_y = img
    train_x = test_x[::2, ::2]
    train_y = img[::2, ::2]

    return test_x, test_y, train_x, train_y


def snn(grade, input_shape_x, opt, seed=42):
    # Initialize the random seed
    key = random.PRNGKey(seed)
    
    # Define the layers for the x input
    layers_x = []
    for i in range(opt.num_layer['grade' + str(grade)]):
        layers_x.append(stax.Dense(opt.num_channel['grade' + str(grade)]))
        layers_x.append(stax.Relu)
    
    # Final dense layer for x
    _, snn_feature = stax.serial(*layers_x)
    
    layers_x.append(stax.Dense(3))
    
    # Define the network for x and y
    init_fn, snn_no_identity = stax.serial(*layers_x)
    _, params = init_fn(key, input_shape_x)
    
    # Define the final model structure
    def model_fn(params, inputs_x, inputs_y):
        x = snn_no_identity(params, inputs_x)  # Apply network to inputs_x
        y = inputs_y
        outputs = np.add(x, y)  # Combine x and y
        return stax.Identity[1](None, outputs)  # Apply sigmoid activation
    
    return snn_feature, snn_no_identity, model_fn, params


# Train model with given hyperparameters and data
def train_model(grade, input_shape_x, train_data, test_data, opt):

    snn_feature, snn_no_identity, model_fn, params = snn(grade, input_shape_x, opt)


    model_pred = jit(lambda params, inputs_x, inputs_y : model_fn(params, inputs_x, inputs_y))   
    snn_feature_pred = jit(lambda params, inputs_x : snn_feature(params, inputs_x))
    snn_no_identity_pred = jit(lambda params, inputs_x : snn_no_identity(params, inputs_x))
    model_loss = jit(lambda params, inputs_x, inputs_y, output: .5 * np.mean((model_pred(params, inputs_x, inputs_y) - output) ** 2))
    model_psnr = jit(lambda params, inputs_x, inputs_y, output: -10 * np.log10(2.*model_loss(params, inputs_x, inputs_y, output)))
    model_grad_loss = jit(lambda params, inputs_x, inputs_y, output: grad(model_loss)(params, inputs_x, inputs_y, output))

    opt_init, opt_update, get_params = optimizers.adam(opt.learning_rate['grade'+str(grade)])
    opt_update = jit(opt_update)

    opt_state = opt_init(params)

    train_psnrs = []
    test_psnrs = []
    xs = []
    for i in range(opt.epoch['grade'+str(grade)]):
        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), *train_data), opt_state)

        if i % opt.interval == 0:
            train_psnrs.append(model_psnr(get_params(opt_state), *train_data))
            test_psnrs.append(model_psnr(get_params(opt_state), *test_data))
            xs.append(i)

    train_features = snn_feature_pred(get_params(opt_state), train_data[0])
    test_features = snn_feature_pred(get_params(opt_state), test_data[0])

    train_no_identity = snn_no_identity_pred(get_params(opt_state), train_data[0])
    test_no_identity = snn_no_identity_pred(get_params(opt_state), test_data[0])

    
    pred_imgs = model_pred(get_params(opt_state), test_data[0], test_data[1])
    loss = model_loss(get_params(opt_state), *train_data)
    
    

    return {
        'params': get_params(opt_state),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs': pred_imgs,
        'xs': xs,
        'train_features': train_features,
        'test_features': test_features,
        'train_no_identity': train_no_identity,
        'test_no_identity': test_no_identity,
        'loss': loss
    }


def MGDLmodel(opt):
    
    test_x, test_y, train_x, train_y = data_setup(opt)
    
    train_features = train_x
    test_features = test_x
    train_accumulations = np.zeros_like(train_y)
    test_accumulations = np.zeros_like(test_y)

    SaveHistory = {}

    current_epsilon = 1 

    for grade in range(1, opt.grade+1):
        input_shape_x = np.shape(train_features)[1:]
        train_data = [train_features, train_accumulations, train_y]
        test_data = [test_features, test_accumulations, test_y]
        s_time = time.time() 
        history = train_model(grade, input_shape_x, train_data, test_data, opt)
        e_time = time.time()
        train_features = history['train_features']
        test_features = history['test_features']
        train_no_identity = history['train_no_identity']
        test_no_identity = history['test_no_identity']

        train_accumulations += current_epsilon * train_no_identity
        test_accumulations += current_epsilon * test_no_identity


        if opt.epsilon:
            current_epsilon = history['loss']
        else:
            current_epsilon = 1
        
        SaveHistory['grade'+str(grade)] = {
            'params': history['params'],
            'train_psnrs': history['train_psnrs'],
            'test_psnrs': history['test_psnrs'],
            'pred_imgs': history['pred_imgs'],
            'xs': history['xs'],
            'time': e_time - s_time,
            'current_epsilon': current_epsilon
        }
        
        print(f"At grade {grade}, train time: {e_time - s_time},  train psnr: {history['train_psnrs'][-1]}, test psnr: {history['test_psnrs'][-1]}\n")
        

        
    picklename = 'results/MGDL_img%s_epsilon%s_grade%d_trainpsnr%.4e_testpsnr%.4e.pickle' % (
        opt.image, str(opt.epsilon), opt.grade, history['train_psnrs'][-1], history['test_psnrs'][-1]
    )

    
    with open(picklename, 'wb') as f:
        pickle.dump([SaveHistory, opt], f)        


def analysis(filepath, LossPsnr_print, Fig_print, grade):

    with open(filepath, 'rb') as f:
        [SaveHistory, opt] = pickle.load(f)

    
    print(opt)

    
    TRAIN_psnr = []
    TEST_psnr = []



    current_epoch = 0
    MUL_EPOCH = [0]
    total_time = 0
    for grade in range(1, grade+1):

        current_epoch += opt.epoch['grade'+str(grade)]
        MUL_EPOCH.append(current_epoch)
        history_dic = SaveHistory['grade'+str(grade)]
        train_psnr = history_dic['train_psnrs']
        test_psnr = history_dic['test_psnrs']

        pred_imgs = history_dic['pred_imgs']
        
        xs = history_dic['xs']
        time = history_dic['time']
        current_epsilon = history_dic['current_epsilon']
        TRAIN_psnr.extend(train_psnr)
        TEST_psnr.extend(test_psnr)
        total_time += time

        print(f'at grade {grade}, current epsilon is {current_epsilon} train time is {time}, train PSNR is {train_psnr[-1]}, test PSNR is {test_psnr[-1]}')




        if Fig_print:
            test_img = pred_imgs
            plt.imshow(test_img)
            plt.show()

    print(f'the total time is {total_time}')


    if LossPsnr_print:

        epochs = [opt.interval * (i+1) for i in range(len(TRAIN_psnr))]
        # print(epochs)
        plt.plot(epochs, TRAIN_psnr, label='Training PSNR')
        plt.plot(epochs, TEST_psnr, label='Testing PSNR')
        plt.title('MGDL', fontsize=20)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('PSNR', fontsize=20)
        # plt.ylim([14, 24])
        plt.legend(fontsize=20)
        for x in MUL_EPOCH:
            plt.axvline(x, color='k', linestyle=':')
        plt.xticks(MUL_EPOCH) 
        plt.xticks(fontsize=18) 
        plt.yticks(fontsize=18)     
        plt.tight_layout()
        plt.show()
    