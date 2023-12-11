import os
import subprocess
import random
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from PIL import Image
import config
import re
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras import metrics
from keras import losses

#first need to read in the scroll data
#The team has already made segmentation of the data which is stored in dl.ash2txt.org
#username password for access is in config.py

def download_files(username, password, path, save_dir): 
    url = path
    # wget command
    cmd = [
        "wget",
        "--no-parent",
        "-r",
        "-l",
        "1",
        "--user=" + username,
        "--password=" + password,
        url,
        "-np",
        "-nd",
        "-nc",  # Don't overwrite files
        "-P",
        save_dir
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        error_output = e.output.decode('utf-8')  # Decode the error output as UTF-8
        print(f"Command failed with exit code {exit_code}.\nError output:\n{error_output}")

def to_8bit(dir_path):#change 16bit data into 8 bit data
    for file in os.listdir(dir_path):
        if file.endswith('.tif'): #we change any tiff file 
            try:
                pathname = os.path.join(dir_path, file)
                img = Image.open(pathname)
                img_float = Image.fromarray(np.array(img) / 2**8-1)
                im = img_float.convert('L')
                im.save(pathname)
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
def to_8bit_training_data(dir_path): #exact same as above, but doesn't replace the original data
    path_to_volume = os.path.join(dir_path, 'surface_volume')
    path_to_8bit = os.path.join(dir_path,'surface_volume_8bit')
    os.makedirs(path_to_8bit)
    for file in os.listdir(path_to_volume):
        if file.endswith('.tif'):
            try:
                pathname = os.path.join(path_to_volume, file)
                pathname_8bit = os.path.join(path_to_8bit, file)
                img = Image.open(pathname)
                img_float = Image.fromarray(np.array(img) / 2**8-1)
                im = img_float.convert('L')
                im.save(pathname_8bit) 
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
def quick_test(dir_path='train'):
    for number in os.listdir(dir_path):
        print(number)
        if not number.startswith('.'):
            path_to_data = os.path.join(dir_path,number)
            for splits in os.listdir(path_to_data):
                print(splits)
def sorted_alphanumeric(data): #to be able to sort like 1,2,10: not 1,10,2
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def make_dataset_onelayer(dir_path = 'train', train=True, layer='32'):
    x_train = []
    y_train = []
    dimensions = []
    for number in sorted(os.listdir(dir_path)):
        if not number.startswith('.'):
            path_to_data = os.path.join(dir_path,number)
            dimensions.append(read_rows_and_columns(path_to_data))
            for splits in os.listdir(path_to_data):
                if splits.startswith('surface_split'):
                    surface_path = os.path.join(path_to_data, splits)
                    for image in os.listdir(surface_path): #first numbers
                        if image.startswith(layer):
                            image_path = os.path.join(surface_path, image)
                            image_chunks = sorted_alphanumeric(os.listdir(image_path))
                            for image_chunk in image_chunks:
                                if image_chunk.endswith('.tif'):
                                    chunk_path = os.path.join(image_path, image_chunk)
                                    img = Image.open(chunk_path)
                                    imarray = np.array(img) / 255
                                    x_train.append(np.pad(imarray,12,mode='constant'))
                if train:                    
                    if splits.startswith('label_split'):
                        label_path = os.path.join(path_to_data, splits)
                        chunk_dir = sorted_alphanumeric(os.listdir(label_path))
                        for chunk in chunk_dir:
                            if chunk.endswith('.tif'):
                                chunk_path_ = os.path.join(label_path, chunk)
                                label = Image.open(chunk_path_)
                                label_array = np.array(label) 
                                y_train.append(label_array)
    return np.array(x_train),np.array(y_train), dimensions
def multilayer_dataset(layers = [31,32,33]):
    x_train = []
    y_train = []
    d_train = []
    lays = len(layers)
    for layer in layers:
        print('on layer ', layer - layers[0], ' of ', lays)
        x,y,d = make_dataset_onelayer(layer = str(layer))
        x_train.append(x)
        if not d_train: #each layer should be the same dimensions, only need to save once
            d_train.append(d)
            y_train.append(y)
    x_train = np.transpose(np.array(x_train), (1,2,3,0))
    y_train = np.transpose(np.array(y_train),(1,2,3,0))
    d_train = np.array(d_train).squeeze()
    return x_train, y_train, d_train

def read_training(x_train,y_train):
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    fig.tight_layout(pad=2)
    ax1.set_title('fragment')
    image1 = ax1.imshow(x_train[4])
    fig.colorbar(image1, ax = ax1, fraction=0.04)
    ax2.set_title('ink label')
    image2 = ax2.imshow(y_train[4])
    fig.colorbar(image2, ax = ax2,fraction=0.04)
    plt.show()
"""
test_path = "http://dl.ash2txt.org/hari-seldon-uploads/chuck-paths/scroll1/path-1/layers/"
dir_test = "testing_layers"
os.makedirs(dir_test, exist_ok=True)
with open('mydata.json') as f:
    CONFIG = json.load(f)
DATA = CONFIG["data"]
print(CONFIG["data"]["testing"]["url"])
#download_files(config.username, config.password, test_path, dir_test)
"""

def make8bittrain():
    for number in os.listdir('train'):
        if not number.startswith('.'):#avoid hidden files
            try:
                pathname = os.path.join('train',number)
                to_8bit_training_data(pathname)
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
def split_image_view(tiles, x=10,y=15):
    fig, axarr = plt.subplots(ncols=x,nrows=y,layout='compressed')
    for row in range(y):
        for col in range(x):
            im = axarr[row,col].imshow(tiles[col + row*x],
                                       cmap = plt.cm.get_cmap('gray').reversed(),
                                        vmin = 0.0, vmax = 1.0 )
    plt.setp(axarr, xticks=[],yticks=[])
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax = cb_ax )
    plt.show(block=False)
def read_rows_and_columns(dir): 
    rows, columns = 0,0
    for dirs in os.listdir(dir):
        if dirs.startswith('surface_split'):
            values = dirs.split('surface_split',1)[1].split(' ',1)
            rows = int(values[1][:-1])
            columns = int(values[0][1:])
    return rows,columns
 

def split_image(path,N=500,M=500): 
    #split the image into more managable chunks
    # save each chunk into a folder within that image layer
    # even the 2cm^2 test layer is 2596 x 2226
    # splitting into chunks of 500 x 500 should be managable
    pathname = os.path.join(path, 'surface_volume_8bit')
    split_path = os.path.join(path, 'surface_split')
    os.makedirs(split_path,exist_ok=True)
    for file in os.listdir(pathname):
        if file.endswith('.tif'):
            image_path = os.path.join(pathname, file)
            img = Image.open(image_path)
            img_array = np.array(img)
            #now split this array into chunks
            columns = int(img_array.shape[0]/M)
            rows = int(img_array.shape[1]/N)
                            
            tiles = [img_array[x:x+M,y:y+N] 
                     for x in range(0,M * columns,M)
                     for y in range(0,N * rows,N)]
            filename = file.replace('.tif','')
            image_dir = os.path.join(split_path, filename)
            os.makedirs(image_dir)
            for index,tile in enumerate(tiles):
                part_name = '_' + str(index) + '.tif'
                im = Image.fromarray(tile)
                im_ = im.convert('L')
                part_path = os.path.join(image_dir, part_name)
                im_.save(part_path)
    new_splitname = split_path + '(' + str(rows) + ' ' + str(columns) + ')'
    os.rename(split_path, new_splitname)
    #print(type(tiles))
    #print(len(tiles))
    #print(tiles[0].shape)

def split_label(path='train', N=500, M=500): 
    #same as the split image function, 
    # but in this case we just need to grab the inklabels.png
    # for every labeled sataset in train
    for number in os.listdir(path):
        if not number.startswith('.'):
            label_path = os.path.join(path,number,'inklabels.png')
            folder_path = os.path.join(path, number, 'label_split')
            img = Image.open(label_path)
            img_array = np.array(img)
            #now split this array into chunks
            columns = int(img_array.shape[0]/M)
            rows = int(img_array.shape[1]/N)
                            
            tiles = [img_array[x:x+M,y:y+N] 
                     for x in range(0,M * columns,M)
                     for y in range(0,N * rows,N)]
            os.makedirs(folder_path, exist_ok=True)
            for index,tile in enumerate(tiles):
                part_name = str(index) + '.tif'
                im = Image.fromarray(tile)
                im_ = im.convert('L')
                part_path = os.path.join(folder_path, part_name)
                im_.save(part_path)



    


def split_all_images(path = 'train'):
    #go through every layer 
    # split and put into folders
    for number in os.listdir(path):
        if not number.startswith('.'):
            #print(number)
            pathname = os.path.join(path, number)
            split_image(pathname)

def read_multiple_chunks(x_train, y_train):
    fig, axarr = plt.subplots(ncols=12, nrows=4, layout='compressed')
    for i in range(12):
        axarr[0,i].imshow(x_train[i])
        axarr[1,i].imshow(x_train[i+12])
        axarr[2,i].imshow(y_train[i])
        axarr[3,i].imshow(y_train[i+12])
        plt.setp(axarr, xticks=[])
    plt.show()
def create_a_test_dataset(path='test'):
    for letter in os.listdir(path):
        if not letter.startswith('.'):
            letter_path = os.path.join(path,letter)
            to_8bit_training_data(letter_path)
            split_image(letter_path)

def test_getting_data_onlylayer32():
    x_train,y_train, dimensions = make_dataset_onelayer()
    print('x_train shape: ', x_train.shape, '\n y_train shape: ',y_train.shape)
    

def making_sense_of_data2D():
    x_train, y_train, d_train = make_dataset_onelayer()
    x_test, y_test, d_test = make_dataset_onelayer(dir_path='test', train=False)
    print(d_train, '\n', d_test)
    print(x_train.shape, y_train.shape)


    
    #try to read each full image in training data
    for i in range(len(d_train)):
        start = 0
        for prev_image in d_train[:i]:
            start += prev_image[0] * prev_image[1]
        end = start + d_train[i][0]* d_train[i][1]
        #if i == 0:
        split_image_view(x_train[start:end], x = d_train[i][1], y = d_train[i][0])
        split_image_view(y_train[start:end], x = d_train[i][1], y = d_train[i][0])
    plt.show()
    
def binary_ratio(y_sample):
    zeros = 0
    for arr in y_sample:
        zeros += np.count_nonzero(arr)
    return float(zeros)/ len(y_sample.flatten())
def filter_samples(y_train):
    choice = []
    for index, y in enumerate(y_train):
        if binary_ratio(y) > 0.1:
            choice.append(index)
    return choice
    


def create_model2D():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 4, kernel_size = (25,25), input_shape = (500,500,1), activation = 'sigmoid', padding = 'same'),
    tf.keras.layers.Conv2D(filters = 4, kernel_size = (12,12), activation = 'sigmoid', padding = 'same'),
    tf.keras.layers.Conv2D(filters = 1, kernel_size = (3,3), activation = 'sigmoid', padding = 'same')
])
    model.compile(optimizer='adam',
              loss=losses.BinaryCrossentropy(),
              metrics=[metrics.FalseNegatives(),metrics.FalsePositives(), 'accuracy'])
    return model

def create_model3D():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(filters = 16, kernel_size = (20,20,5), activation = 'relu', padding = 'valid', input_shape = (524,524,7,1)),
    tf.keras.layers.Conv3D(filters = 16, kernel_size = (6,6,3), activation = 'sigmoid', padding = 'valid'),
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (10,10), activation = 'sigmoid', padding = 'same'),
    tf.keras.layers.Conv2D(filters = 1, kernel_size = (4,4), activation = 'sigmoid', padding = 'same')
])
    model.compile(optimizer='adam',
              loss=losses.BinaryCrossentropy(),
              metrics=[metrics.FalseNegatives(),metrics.FalsePositives(), 'accuracy'])
    return model
def read_model():
    model = create_model3D()
    model.summary()
def main2D():
    x_train,y_train, dimensions = make_dataset_onelayer()
    print('x_train shape: ', x_train.shape, '\n y_train shape: ',y_train.shape)
    indexs = filter_samples(y_train)
    print(indexs)
    X_train = np.array([x_train[i] for i in sorted(indexs)])
    Y_train = np.array([y_train[i] for i in sorted(indexs)])
  
    
    model = create_model2D()
    model.summary()
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    #
    # history = model.fit(x_train[:743],y_train[:743], epochs = 5, callbacks = [cp_callback])
    history = model.fit(X_train, Y_train, epochs = 5, callbacks = [cp_callback])

    predictions_ = model.predict(x_train[743:893])
    #predictions = model.predict(x_train[:192])
    
    split_image_view(x_train[743:893],x=10,y=15)
    split_image_view(y_train[743:893],x=10,y=15)
    #split_image_view(x_train[:192],x=12,y=16)
    #split_image_view(y_train[:192],x=12,y=16)
    #split_image_view(predictions[:,:,:,0],x=12,y=16)
    split_image_view(predictions_[:,:,:,0],x=10,y=15)
    plt.show()

def main3D():
    layers = range(28,35)
    x_train,y_train, dimensions = multilayer_dataset(layers = layers)
    print('x_train shape: ', x_train.shape, '\n y_train shape: ',y_train.shape)
    indexs = filter_samples(y_train[0:743,:,:,0])
    X_train = np.array([x_train[i] for i in sorted(indexs)])
    Y_train = np.array([y_train[i] for i in sorted(indexs)])
    print('X_train shape: ', X_train.shape, '\n Y_train shape: ',Y_train.shape)
    model = create_model3D()
    model.summary()
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    #history = model.fit(x_train[:743],y_train[:743], epochs = 5, callbacks = [cp_callback])
    history = model.fit(X_train, Y_train, epochs = 5, callbacks = [cp_callback])

    x_test,y_test, test_dimensions = make_dataset_onelayer(dir_path='test', train=False)
    predictions_ = model.predict(x_train[743:893])
    #predictions = model.predict(x_train[:192])
    
    split_image_view(x_train[743:893, :,:,int(x_train.shape[3]/2)],x=10,y=15) #image the middle layer
    split_image_view(y_train[743:893],x=10,y=15)
    #split_image_view(x_train[:192],x=12,y=16)
    #split_image_view(y_train[:192],x=12,y=16)
    #split_image_view(predictions[:,:,:,0],x=12,y=16)
    split_image_view(predictions_[:,:,:,0],x=10,y=15)
    plt.show()
def read_predictions(predictions):
    print('The shape of predictions: ', predictions.shape)
def download_some_scroll_data(): #WIP
    #include link to what data you want to download
    dir_ = 'test_scrolls'
    os.makedirs(dir_, exist_ok=True)
    path = 'http://dl.ash2txt.org/hari-seldon-uploads/chuck-paths/scroll1/path-1/layers/00.tif' #path to a small segmentation of scroll 1
    username = 'registeredusers'
    password = 'only'
    download_files(username=username, password=password, path=path, save_dir=dir_)

if __name__ == '__main__':
    main3D()

