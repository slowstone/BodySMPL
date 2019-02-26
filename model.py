import re
import multiprocessing
import tensorflow as tf
import numpy as np
from config import Config
from dataset import imread
from tf_smpl.batch_smpl import SMPL

def mse(y_true,y_pred):
    return tf.keras.losses.mean_squared_error(y_true,y_pred)

def mae(y_true,y_pred):
    return tf.keras.losses.mean_absolute_error(y_true,y_pred)

def mape(y_true,y_pred):
    return tf.keras.losses.mean_absolute_percentage_error(y_true,y_pred)

def mse_point_loss(a, b):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(a - b), 2)),name='mse_point')

def smplfuc(tensorlist):
    shape = tensorlist[0]
    pose = tensorlist[1]
    smpl_model = SMPL('./tf_smpl/models/neutral_smpl_with_cocoplus_reg.pkl')
    verts_ori, J, R = smpl_model(shape, pose, get_skin=True)

    return verts_ori

class Model(object):
    def __init__(self,config=Config()):
        self.config = config
        if config.mode == 'train':
            self.model,losses = self.build()
            if self.config.param['MODEL_PATH'] is not None:
                self.model.load_weights(self.config.param['MODEL_PATH'],by_name=True)
            metrics = ['mse','mae','mape']
            loss_weights = self.config.param['LOSS_WEIGHTS']
            self.compile_fuc(losses=losses,metrics=metrics)
        if config.mode == 'eval':
            self.model = self.build()
            if self.config.param['MODEL_PATH'] is not None:
                self.model.load_weights(self.config.param['MODEL_PATH'],by_name=True)

    def build(self):
        if self.config.param['MODEL_NAME'] == 'i_spv':
            inputs,outputs = self.get_layers_i_spv()
            losses={"shape":mse,"pose":mse,"verts":mse_point_loss}
        if self.config.param['MODEL_NAME'] == 'ip_sv':
            inputs,outputs = self.get_layers_ip_sv()
            losses={"shape":mse,"verts":mse_point_loss}
        model = tf.keras.models.Model(inputs = inputs,outputs = outputs, name = self.config.param['MODEL_NAME'])
        try:
            model = tf.keras.utils.multi_gpu_model(
                                    model,
                                    gpus=len(self.config.param['GPUS'].split(',')),
                                    cpu_merge=False)
            print("Training using multiple GPUs..")
        except:
            print("Training using single GPU or CPU..")
        return model,losses
    
    def get_layers_i_spv(self):
        image = tf.keras.Input(shape = (None,None,3),name='image',dtype='float32')
        
        C1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',name='sC1',padding='same')(image)
        P1 = tf.keras.layers.AveragePooling2D(name='sP1')(C1)
        C2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',name='sC2',padding='same')(P1)
        P2 = tf.keras.layers.AveragePooling2D(name='sP2')(C2)
        
        sC3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',name='sC3',padding='same')(P2)
        sP3 = tf.keras.layers.GlobalAveragePooling2D(name='sP3')(sC3)
        shape = tf.keras.layers.Dense(self.config.param['BETA_NUMS'],activation='softmax',name='shape')(sP3)
        
        pC3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',name='pC3',padding='same')(P2)
        pP3 = tf.keras.layers.GlobalAveragePooling2D(name='pP3')(pC3)
        pose = tf.keras.layers.Dense(self.config.param['POSE_NUMS'],activation='softmax',name='pose')(pP3)     
        
        verts = tf.keras.layers.Lambda(smplfuc,name='verts')([shape,pose])
#         verts = tf.keras.layers.Flatten(name='verts')(verts_ori)
        
        inputs = [image]
        outputs =[shape,pose,verts]
        return inputs,outputs
    
    def get_layers_ip_sv(self):
        image = tf.keras.Input(shape = (None,None,3),name='image',dtype='float32')
        
        sC1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',name='sC1',padding='same')(image)
        sP1 = tf.keras.layers.AveragePooling2D(name='sP1')(sC1)
        sC2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',name='sC2',padding='same')(sP1)
        sP2 = tf.keras.layers.AveragePooling2D(name='sP2')(sC2)
        sC3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',name='sC3',padding='same')(sP2)
        sP3 = tf.keras.layers.GlobalAveragePooling2D(name='sP3')(sC3)
        shape = tf.keras.layers.Dense(self.config.param['BETA_NUMS'],activation='softmax',name='shape')(sP3)

        pose = tf.keras.Input(shape = (72,),name='pose',dtype='float32')
        
        verts = tf.keras.layers.Lambda(smplfuc,name='verts')([shape,pose])
#         verts = tf.keras.layers.Flatten(name='verts')(verts_ori)
        
        inputs = [image,pose]
        outputs =[shape,verts]
        return inputs,outputs
    
    def compile_fuc(self,losses,metrics=None,loss_weights=None):
        if self.config.param['OPT_STRING'] == "momentum":
            opt = tf.keras.optimizers.SGD(lr=self.config.param['LEARNING_RATE'],
                                momentum=self.config.param['MOMENTUM'])
        if self.config.param['OPT_STRING'] == "nesterov":
            opt = tf.keras.optimizers.SGD(lr=self.config.param['LEARNING_RATE'],
                                momentum=self.config.param['MOMENTUM'],
                                nesterov=True)
        if self.config.param['OPT_STRING'] == "adam":
            opt = tf.keras.optimizers.Adam(lr=self.config.param['LEARNING_RATE'],
                                beta_1=self.config.param['ADAM_BETA_1'],
                                beta_2=self.config.param['ADAM_BETA_2'])
        self.model.compile(optimizer=opt,loss=losses,metrics=metrics,loss_weights=loss_weights)

    def train(self,train_generator,val_generator,callbacks):
        self.model.fit_generator(train_generator,
                   steps_per_epoch = self.config.param['TRAIN_STEPS'],
                   epochs = self.config.param['EPOCHS'],
                   validation_data=val_generator,
                   validation_steps=self.config.param['VALIDATION_STEPS'],
                   callbacks=callbacks,
                   workers = multiprocessing.cpu_count(),
                   max_queue_size = 10,
                   shuffle = True,
                   use_multiprocessing = True)
    
    def train_dataset(self,train_dataset,val_dataset,callbacks):
        self.model.fit(train_dataset, 
                    validation_data=val_dataset,
                    epochs=self.config.param['EPOCHS'],
                    steps_per_epoch = self.config.param['TRAIN_STEPS'],
                    validation_steps= self.config.param['VALIDATION_STEPS'],
                    callbacks=callbacks)
