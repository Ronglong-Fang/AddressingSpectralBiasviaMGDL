import jax.numpy as np
from jax import jit, grad, random
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
import os, imageio
# import seaborn as sns
# sns.set()





#set up data
def data_setup():

    # Download image, take a square crop from the center
    image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]
    
    # plt.imshow(img)
    # plt.show()
    
    # Create input pixel coordinates in the unit square
    coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1)

    test_data = [x_test, img]
    train_data = [x_test[::2, ::2], img[::2, ::2]]
    
    # plt.imshow(img[::2, ::2])
    # plt.show()
    
    return test_data, train_data


# JAX network definition
def make_network(opt):
    layers = []
    for i in range(opt.num_layers):
        layers.append(stax.Dense(opt.num_channels))
        layers.append(stax.Relu)
    layers.append(stax.Dense(3))
    layers.append(stax.Identity)
    return stax.serial(*layers)

# Train model with given hyperparameters and data
def train_model(opt, train_data, test_data):
    key = random.PRNGKey(0)

    s_time = time.time() 

    init_fn, apply_fn = make_network(opt)

    model_pred = jit(lambda params, x: apply_fn(params, x))
    model_loss = jit(lambda params, x, y: .5 * np.mean((model_pred(params, x) - y) ** 2))
    model_psnr = jit(lambda params, x, y: -10 * np.log10(2.*model_loss(params, x, y)))
    model_grad_loss = jit(lambda params, x, y: grad(model_loss)(params, x, y))

    opt_init, opt_update, get_params = optimizers.adam(opt.learning_rate)
    opt_update = jit(opt_update)

    _, params = init_fn(key, (-1, train_data[0].shape[-1]))
    opt_state = opt_init(params)

    train_psnrs = []
    test_psnrs = []
    pred_imgs = []
    residual_imgs = []
    xs = []
    for i in range(opt.epoch):
        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), *train_data), opt_state)

        if i % 25 == 0:
            train_psnrs.append(model_psnr(get_params(opt_state), *train_data))
            test_psnrs.append(model_psnr(get_params(opt_state), *test_data))
            pred_imgs.append(model_pred(get_params(opt_state), test_data[0]))
            xs.append(i)

    pred_imgs = model_pred(get_params(opt_state), test_data[0])
    residual_imgs = np.abs( test_data[1] -  model_pred(get_params(opt_state), test_data[0]))

    e_time = time.time() 

    history_dict =  {
        'state': get_params(opt_state),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs': pred_imgs,
        'residual_imgs': residual_imgs,
        'xs': xs,
        'train_time': e_time - s_time
    }

    
    print(f"train time: {e_time - s_time},  train psnr: {train_psnrs[-1]}, test psnr: {test_psnrs[-1]}\n")

    picklename = 'results/SGDL_numlayer%d_learningrate%.4e_trainpsnr%.4e_testpsnr%.4e.pickle' % (opt.num_layers, opt.learning_rate, train_psnrs[-1], test_psnrs[-1])

    
    with open(picklename, 'wb') as f:
        pickle.dump([history_dict, opt], f)      

    return 



def analysis(filepath):


    # Plot training and validation loss
    with open(filepath, 'rb') as f:
        [history_dic, opt] = pickle.load(f)


    train_psnr = history_dic['train_psnrs']
    test_psnr = history_dic['test_psnrs']
    xs = history_dic['xs']
    time = history_dic['train_time']
    pred_imgs = history_dic['pred_imgs'] 


    print(opt)
    print(f'train time is {time}')
    print(f'train PSNR is {train_psnr[-1]}, test PSNR is {test_psnr[-1]}')

    

    plt.plot(xs, train_psnr, label='Training PSNR')
    plt.plot(xs, test_psnr, label='Testing PSNR')
    plt.title('SGDL', fontsize=20)
    plt.xlabel('Epochs',  fontsize=20)
    plt.ylabel('PSNR',  fontsize=20)
    plt.ylim([15, 29])
    plt.legend(fontsize=20)
    plt.xticks(fontsize=18)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=18) 

    plt.tight_layout()
    plt.show()

    test_img = pred_imgs
    test_img = test_img.reshape(512, 512, 3)
    plt.imshow(test_img)
    plt.show()

