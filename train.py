# coding: utf-8
import json
import os
import numpy as np
import datetime
from config import Config
model_config = Config()
IS_SAVE = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = model_config.param['GPUS'] # "0, 1" for multiple

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

from callbacks import *

from dataset import surreal_sequence,TFRecordDataset
from model import Model

base_dir = './logs'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

model_name = model_config.param['MODEL_NAME']
image_type = model_config.param['IMAGE_TYPE']
data_version = model_config.param['DATA_VERSION']

if data_version == 'surreal':
    bodyseg_dir = '../dataset/SURREAL/summary/bodyseg'
    mask_dir = '../dataset/SURREAL/summary/bodyseg'
    label_dir = '../dataset/SURREAL/summary/labels'

    train_json_path = '../dataset/SURREAL/summary/train_frame_sample.json'
    train_generator = surreal_sequence(train_json_path,mask_dir,bodyseg_dir,label_dir,model_config)
    val_json_path = '../dataset/SURREAL/summary/test_frame_sample.json'
    val_generator = surreal_sequence(val_json_path,mask_dir,bodyseg_dir,label_dir,model_config)

    model_config.set_param(['TRAIN_JSON_PATH','VAL_JSON_PATH'],[train_json_path,val_json_path])
    
    train_nums = train_generator.names_num
    val_nums = val_generator.names_num

if data_version == 'surreal_tfrecord':
    train_tfrecord = '/media/hdc/xuyan/dataset/SURREAL_tfrecord/sample-noshape/train'
    val_tfrecord = '/media/hdc/xuyan/dataset/SURREAL_tfrecord/sample-noshape/test'
    train_dataset = TFRecordDataset(train_tfrecord,model_config).it
    val_dataset = TFRecordDataset(train_tfrecord,model_config).it
    
#     train_nums = train_dataset.total_num  # too slow
    train_nums = 519829
#     val_nums = val_dataset.total_num    # too slow
    val_nums = 4241

print("====================dataset scale====================")
print("=========> The number of train dataset: {}".format(train_nums))
print("=========> The number of val dataset: {}".format(val_nums))
print("=====================================================\n")

train_steps = int(train_nums/model_config.param['BATCH_SIZE'])
# train_steps = 5000
val_steps = int(val_nums/model_config.param['BATCH_SIZE'])
# val_steps = 500

model_config.set_param(['TRAIN_STEPS','VALIDATION_STEPS'],[train_steps,val_steps])
model_config.show_config()

model = Model(model_config)

now = datetime.datetime.now()

log_dir = os.path.join(base_dir, "{}_{}_{:%Y%m%dT%H%M}".format(image_type,model_name,now))
checkpoint_path = os.path.join(log_dir, "ep_*epoch*.h5")
checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")
min_lr = model_config.param['LEARNING_RATE'] / pow(model_config.param['LR_DECAY'],2)

if IS_SAVE:
    callbacks = [
            LRTensorBoard(log_dir=log_dir,
                    histogram_freq=0, write_graph=True, write_images=False),
            # tf.keras.callbacks.TensorBoard(log_dir=log_dir,
            #          histogram_freq=0, write_graph=True, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                    verbose=1, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                    factor=model_config.param['LR_DECAY'],
                                    patience=5,
                                    verbose=1,
                                    mode='auto',
                                    epsilon=0.0001,
                                    cooldown=5,
                                    min_lr=min_lr)
        ]
    model_config.save_config(out_dir=log_dir)
else:
    callbacks = [
#             tf.keras.callbacks.LearningRateScheduler(lrdecay),
        ]
if data_version == 'surreal_tfrecord':
    model.train_dataset(train_dataset,val_dataset,callbacks)
else:
    model.train(train_generator,val_generator,callbacks)
