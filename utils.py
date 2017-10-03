from __future__ import print_function

import numpy as np
from keras import backend as K
import cv2
import os
from sklearn import metrics as ms

# img_width = 32
# img_height = 32

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
# layer_name = 'conv2d_3'

def computeBaselineMSE(ylabel):
    ymean = np.mean(ylabel)
    mean_mse = ms.mean_squared_error(ylabel, np.ones(ylabel.shape)*ymean)
    return mean_mse

def generateWindowFromIdx(win_size,win_idx,sourceDir, meanImg):
    X = []
    for i in range(win_size):
        localImg = cv2.imread(sourceDir + 'frame_{:06d}.jpg'.format(win_idx+ i+ 1))
        X.append((localImg-meanImg)/255.0)
    return np.asarray(X)

def computeMeanImageFromFolder(sourceDir, ext, img_shape):
    mean_img = np.zeros(img_shape)
    img_list = [f for f in os.listdir(sourceDir) if f.endswith(ext)]
    if img_list:
        for k in img_list:
            local_img = cv2.imread(sourceDir + k)
            mean_img += 1.0/len(img_list) * local_img
    return mean_img

def reshapeImages(targetW,targetH,sourceDir,targetDirRoot):
    targetDir = targetDirRoot + "data/train_" + '{:04d}'.format(targetW) + "_" + '{:04d}'.format(targetH)
    try:
        os.stat(targetDir)
    except:
        os.mkdir(targetDir) 
    file_list = [filein for filein in os.listdir(sourceDir) if filein.endswith(".jpg")]
    for i in file_list:
        localImg = cv2.imread(sourceDir + i)
        res = cv2.resize(localImg,(targetW, targetH), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(targetDir + "/" + i,res)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def getFiltersAsImg(model, layer_name, get_best):
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[layer_name].output
    input_img = model.input
    if K.image_data_format() == 'channels_first':
        no_filters = layer_output.shape.as_list()[1]
        img_width = model.input.shape.as_list()[2]
        img_height = model.input.shape.as_list()[3]
    else:
        no_filters = layer_output.shape.as_list()[3]
        img_width = model.input.shape.as_list()[1]
        img_height = model.input.shape.as_list()[2]
    print("img dims: {:03d} x {:03d}".format(img_width, img_height))
    kept_filters = []
    for filter_index in range(0, no_filters):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])
            
        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        grads = normalize(grads)
        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])
        # step size for gradient ascent
        step = 1.
        
        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128
        
        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
                        
            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break
                
        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
    if get_best is not None and get_best<no_filters:
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:get_best]
    return kept_filters

