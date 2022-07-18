import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
import os, time, random

import tensorflow as tf
from PIL import Image
from skimage import io

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#########################
# DL Libraries
#########################
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomFlip, RandomZoom, Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras import layers



    
class Model():
    dir_path ="./chest_xray_dataset/"
    train_path = dir_path+'train/'
    valid_path = dir_path+'val/'
    test_path = dir_path+'test/'
    label_dict={'NORMAL':0,'PNEUMONIA':1}

    #The sizes of the images vary over a wide range. So best option is to fix a size for all.
    target_size = (300, 300)

    input_shape = (300, 300, 1)

    train_dir = None
    val_dir = None
    test_dir = None

    batch_size = 64
    def __init__(self):
        # class initialization method
        self.temp = 0
        print("Hello, train done")
        self.train()
    
    ### check number of train, valid and test images 
    labels = ['NORMAL', 'PNEUMONIA']
    ## method to create file systenm structure for the code to work
    ## run when some file systems breaks
    def create_file_system(self,folder,labels):
        ## list items
        items = os.listdir(folder)
        if not os.path.exists(folder+labels[0]+"/"):
            os.mkdir(folder+labels[0]+"/")
        if not os.path.exists(folder+labels[1]+"/"):
            os.mkdir(folder+labels[1]+"/")
        for item in items:
            if self.contains(item, labels[0]):
                os.rename(folder+item, folder+labels[0]+"/"+item)
            elif self.contains(item, labels[1]):
                os.rename(folder+item, folder+labels[1]+"/"+item)
            else:
                print('error')
    def contains(self,file_name, label):
        return file_name.__contains__(label) 

    def check_ims_in_folder(self,labels):
        '''returns tuples of images in each folder'''
        if not os.path.exists(self.train_path+labels[0]+"/"):
            self.create_file_system(self.train_path, labels)
        train_ims_normal = os.listdir(self.train_path+labels[0]+'/')
        train_ims_pneumonia = os.listdir(self.train_path+labels[1]+'/')

        # no valid 
        if not os.path.exists(self.valid_path+labels[0]+"/"):
            self.create_file_system(self.valid_path, labels)
        valid_ims_normal = os.listdir(self.valid_path+labels[0]+'/')
        valid_ims_pneumonia = os.listdir(self.valid_path+labels[1]+'/')

        ## fixgins ifile system 
        if not os.path.exists(self.test_path+labels[0]+"/"):
            self.create_file_system(self.test_path, labels)
        test_ims_normal = os.listdir(self.test_path+labels[0]+'/')
        test_ims_pneumonia = os.listdir(self.test_path+labels[1]+'/')

        #test photos in other file system
        _test_ims_= os.listdir(self.test_path)
        # create two list for labels[0] and labels[1]
        # then loop over test_ims_ and check if it is in the list
        _test_ims_normal = list()
        _test_ims_pneumonia = list()
        for im in _test_ims_:
            if self.contains(im, labels[0]):
                _test_ims_normal.append(im)
            elif self.contains(im, labels[1]):
                _test_ims_pneumonia.append(im)
            else:
                print('error')


        return (train_ims_normal, train_ims_pneumonia), (valid_ims_normal, valid_ims_pneumonia), (test_ims_normal, test_ims_pneumonia)

    
    def preprocess_image(open_image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.per_image_standardization(image)
        return image
        
    def load_and_preprocess_image(self,path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

        #### define a function that will be added as lambda layer later
    def standardize_layer(self,tensor):
        tensor_mean = tf.math.reduce_mean(tensor)
        tensor_std = tf.math.reduce_std(tensor)
        new_tensor = (tensor-tensor_mean)/tensor_std
        return new_tensor
    
    def train(self):
        (train_im_n, train_im_p), (valid_im_n, valid_im_p), (test_im_n, test_im_p) = self.check_ims_in_folder(self.labels)
        train_dir = tf.keras.preprocessing.image_dataset_from_directory(self.train_path, 
                                                                image_size=self.target_size, 
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                color_mode='grayscale',
                                                                label_mode='binary')



        """ print("Val Dataset....")
        val_dir = tf.keras.preprocessing.image_dataset_from_directory(valid_path, 
                                                              image_size=target_size, 
                                                              batch_size=batch_size,
                                                              color_mode='grayscale',
                                                              label_mode='binary') """

        print("Test Datast...")
        test_dir = tf.keras.preprocessing.image_dataset_from_directory(self.test_path, 
                                                               image_size=self.target_size, 
                                                               batch_size=self.batch_size, 
                                                               color_mode='grayscale',
                                                               label_mode='binary')
        tot_normal_train = len(train_im_n) + len(valid_im_n) 
        tot_pneumonia_train = len(train_im_p) + len(valid_im_p)
        
        # Get the classs nanes from the train_dir
        class_names = train_dir.class_names
        new_train_ds = train_dir

        print (new_train_ds, train_dir)

        train_size = int(0.8 * 83) # 83 is the elements in dataset (train + valid)
        val_size = int(0.2 * 83)

        train_ds = new_train_ds.take(train_size)
        val_ds = new_train_ds.skip(train_size).take(val_size)


        #### check the dataset size back again 
        num_elements_train = tf.data.experimental.cardinality(train_ds).numpy()
        print (num_elements_train)
        num_elements_val_ds = tf.data.experimental.cardinality(val_ds).numpy()
        print (num_elements_val_ds)
        rescale_layer = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])

        data_augmentation = tf.keras.Sequential([
          layers.experimental.preprocessing.RandomFlip(),
          layers.experimental.preprocessing.RandomRotation(10), 
          layers.experimental.preprocessing.RandomZoom(0.1)
        ])
        autotune = tf.data.AUTOTUNE ### most important function for speed up training
        train_data_batches = train_ds.cache().prefetch(buffer_size=autotune)
        valid_data_batches = val_ds.cache().prefetch(buffer_size=autotune)
        test_data_batches = test_dir.cache().prefetch(buffer_size=autotune)
        #### check the numbers again
        print (train_data_batches, valid_data_batches)

        num_elements_train_data_batches = tf.data.experimental.cardinality(train_data_batches).numpy()
        print (num_elements_train_data_batches)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.2, patience=3, min_lr=1e-8, verbose=1)

        mcp_save = ModelCheckpoint(filepath="best_model_weights.h5",
                           save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

        ##restore best weights added after 2nd training
        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        ### added after 2nd training 

        METRICS = ['accuracy', 
           tf.keras.metrics.Precision(name='precision'), 
           tf.keras.metrics.Recall(name='recall'), 
           tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
          ]

        freq_neg = tot_normal_train/(tot_normal_train + tot_pneumonia_train)
        freq_pos = tot_pneumonia_train/(tot_normal_train + tot_pneumonia_train)

        pos_weights = np.array([freq_neg])
        neg_weights = np.array([freq_pos])

        print ('check positive weight: ', pos_weights, len(pos_weights))
        print ('check negative weight: ', neg_weights)


        def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
            """
            Return weighted loss function given negative weights and positive weights.

            Args:
              pos_weights (np.array): array of positive weights for each class, size (num_classes)
              neg_weights (np.array): array of negative weights for each class, size (num_classes)

            Returns:
              weighted_loss (function): weighted loss function
            """
            def weighted_loss(y_true, y_pred):
                """
                Return weighted loss value. 

                Args:
                    y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
                    y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
                Returns:
                    loss (float): overall scalar loss summed across all classes
                """
                # initialize loss to zero
                loss = 0.0

                for i in range(len(pos_weights)): # we have only 1 class 
                    # for each class, add average weighted loss for that class 
                    loss += - (K.mean((pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon)) + 
                                      (neg_weights[i] * (1-y_true[:, i]) * K.log(1-y_pred[:, i] + epsilon)) ) )
                return loss
            return weighted_loss
        input_shape = (300, 300, 3)
        inception_resnet_v2 = InceptionResNetV2(
            include_top=False,
            # dowload on the go 
            weights=None,
            # dowladed to te model to work
            #weights="./inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
            input_shape=input_shape)
        def build_model():
            inputs = Input((300, 300, 1))

            x = preprocess_input(inputs) # necessary as per keras documentation 
            x = layers.Lambda(rescale_layer)(x) # rescale incoming images
            x = layers.Lambda(self.standardize_layer)(x) # standardize incoming images
            x = layers.Lambda(data_augmentation)(x) # data augmentation layers
            x = Conv2D(3, (3,3), padding='same')(x) 
            # this is to fool the network that instead of rgb image we passed grayscale image but still have shape 3 at last axis (none, x, x, 3). 



            ###### InceptionResNetV2 + Some Top Layers
            x = BatchNormalization()(x)
            x = inception_resnet_v2(x)

            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(256, (1, 1), activation=LeakyReLU())(x)
            x = BatchNormalization()(x)

            x = Flatten()(x)
            x = Dropout(0.75)(x)

            x = Dense(256, activation=LeakyReLU())(x)
            x = Dropout(0.80)(x)
            x = BatchNormalization()(x)

            outputs = Dense(1, activation="sigmoid")(x)

            model = KerasModel(inputs, outputs)

        #     model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
        #                   loss="binary_crossentropy", 
        #                   metrics=METRICS)
        # added weighted cross entropy loss for the loss instead of 
        # "binary_crossentropy"

            model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                         loss=get_weighted_loss(pos_weights, neg_weights), 
                         metrics=METRICS)




            return model
        model = build_model()
        model.summary()
        start_time = time.time()
        with tf.device("/gpu:0"):
            history = model.fit(train_data_batches, 
                            epochs=100, 
                            validation_data=valid_data_batches,
                            callbacks=[mcp_save, es, reduce_lr])

        end_time = time.time()
        print ('total time taken: in Minutes', (end_time-start_time)/60.)



    def predict(file_path):
    #methid to predict the output
        print("prediction")
    

class customCallbacks(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    self.epoch = epoch + 1
    if self.epoch % 2 == 0:
      print (
          'epoch num {}, train loss: {}, validation loss: {}'.format(epoch, logs['loss'], logs['val_loss']))


ModelTest = Model()