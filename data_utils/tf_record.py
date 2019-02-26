import tensorflow as tf
import numpy as np
import os
import cv2
import json
import multiprocessing
import time
import traceback

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_image(im_path):
    im = cv2.imread(im_path)

    if len(im.shape)==2:#若是灰度图则转为三通道
        print("Warning:gray image",filename)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)#将BGR转为RGB
    return im,im.shape

def convert_to_example(image,
               human,
               bodyseg,
               mask,
               ori_mask,
               labels,
               im_shape,
               human_shape,
               bodyseg_shape,
               mask_shape,
               ori_mask_shape):
    pose = np.array(labels['pose'],dtype=np.float)
    shape = np.array(labels['shape'],dtype=np.float)
    gt2d = np.array(labels['joints2D'],dtype=np.float)
    gt3d = np.array(labels['joints3D'],dtype=np.float)

    feat_dict = {
        'image': bytes_feature(image),
        'image/h': int64_feature(im_shape[0]),
        'image/w': int64_feature(im_shape[1]),
        'image/c': int64_feature(im_shape[2]),
        'image/bodyseg': bytes_feature(bodyseg),
        'image/bodyseg/h': int64_feature(bodyseg_shape[0]),
        'image/bodyseg/w': int64_feature(bodyseg_shape[1]),
        'image/bodyseg/c': int64_feature(bodyseg_shape[2]),
        'image/human': bytes_feature(human),
        'image/human/h': int64_feature(human_shape[0]),
        'image/human/w': int64_feature(human_shape[1]),
        'image/human/c': int64_feature(human_shape[2]),
        'image/mask': bytes_feature(mask),
        'image/mask/h': int64_feature(mask_shape[0]),
        'image/mask/w': int64_feature(mask_shape[1]),
        'image/mask/c': int64_feature(mask_shape[2]),
        'image/ori_mask': bytes_feature(ori_mask),
        'image/ori_mask/h': int64_feature(ori_mask_shape[0]),
        'image/ori_mask/w': int64_feature(ori_mask_shape[1]),
        'image/ori_mask/c': int64_feature(ori_mask_shape[2]),
        'label/pose': float_feature(pose),
        'label/shape': float_feature(shape),
        'label/gt2d': float_feature(gt2d.ravel()),
        'label/gt3d': float_feature(gt3d.ravel()),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))

    return example

def get_example(im_dir,bodyseg_dir,human_dir,mask_dir,label_dir,base_name):
#通过读取read_image将单通道图直接转为3通道，避免出现通道数为1的情况
    im,im_shape = read_image(os.path.join(im_dir,base_name+'.jpg'))
    res,im_f = cv2.imencode('.jpg',im)
    im_f = im_f.tostring()

    bodyseg,bodyseg_shape = read_image(os.path.join(bodyseg_dir,base_name+'.png'))
    res,bodyseg_f = cv2.imencode('.jpg',bodyseg)
    bodyseg_f = bodyseg_f.tostring()

    human,human_shape = read_image(os.path.join(human_dir,base_name+'.jpg'))
    res,human_f = cv2.imencode('.jpg',human)
    human_f = human_f.tostring()

    mask,mask_shape = read_image(os.path.join(mask_dir,base_name+'.png'))
    res,mask_f = cv2.imencode('.jpg',mask)
    mask_f = mask_f.tostring()

    ori_mask = bodyseg.copy()
    ori_mask[ori_mask != 0] = 1
    ori_mask_shape = ori_mask.shape
    res,ori_mask_f = cv2.imencode('.jpg',ori_mask)
    ori_mask_f = ori_mask_f.tostring()

    with open(os.path.join(label_dir,base_name+'.json'),'r') as f:
        labels = json.load(f)

# 这样处理可能存在通道数为1的情况
#     with tf.gfile.GFile(os.path.join(im_dir,base_name+'.jpg'),'rb') as f:
#         im_f = f.read()
#     with tf.gfile.GFile(os.path.join(bodyseg_dir,base_name+'.png'),'rb') as f:
#         bodyseg_f = f.read()
#     with tf.gfile.GFile(os.path.join(human_dir,base_name+'.jpg'),'rb') as f:
#         human_f = f.read()
#     with tf.gfile.GFile(os.path.join(mask_dir,base_name+'.png'),'rb') as f:
#         mask_f = f.read()
#     with open(os.path.join(label_dir,base_name+'.json'),'r') as f:
#         labels = json.load(f)
    return convert_to_example(im_f,
                      human_f,
                      bodyseg_f,
                      mask_f,
                      ori_mask_f,
                      labels,
                      im_shape,
                      human_shape,
                      bodyseg_shape,
                      mask_shape,
                      ori_mask_shape)

def multi_save_to_tfrecord(tfrecord_path,
                        names,
                        im_dir,
                        bodyseg_dir,
                        human_dir,
                        mask_dir,
                        label_dir):
    print('Begin writting',tfrecord_path.split('/')[-2],tfrecord_path.split('/')[-1])
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    start_t = time.time()
    try:
        for i,name in enumerate(names):
            ex = get_example(im_dir,bodyseg_dir,human_dir,mask_dir,label_dir,name)
            writer.write(ex.SerializeToString())
        cost_t = time.time()-start_t
        print('Write',tfrecord_path.split('/')[-2],tfrecord_path.split('/')[-1],'end.',len(names),'Example Cost',cost_t)
    except Exception as e:
        print('Error:',e)
    finally:
        writer.close()

def save_to_tfrecord(pool,save_dir,names,im_dir,bodyseg_dir,human_dir,mask_dir,label_dir,num_per_tfrecord=50000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_name = os.path.basename(save_dir)
    tfrecord_files = '{}.tfrecords'
    cur_file_index = 1
    names_list = []
    for i,name in enumerate(names):
        names_list.append(name)
        if len(names_list) == num_per_tfrecord:
            tfrecord_path = os.path.join(save_dir,tfrecord_files.format(cur_file_index))
            pool.apply_async(
                             multi_save_to_tfrecord,
                            (tfrecord_path,names_list,im_dir,bodyseg_dir,human_dir,mask_dir,label_dir,))
            cur_file_index += 1
            names_list = []
    if len(names_list) != 0:
        tfrecord_path = os.path.join(save_dir,tfrecord_files.format(cur_file_index))
        pool.apply_async(
                        multi_save_to_tfrecord,
                        (tfrecord_path,names_list,im_dir,bodyseg_dir,human_dir,mask_dir,label_dir,))
        cur_file_index += 1
        names_list = []

if __name__ == "__main__":
    im_dir = '../../dataset/SURREAL/summary/image'
    bodyseg_dir = '../../dataset/SURREAL/summary/bodyseg'
    human_dir = '../../dataset/SURREAL/summary/human'
    mask_dir = '../../dataset/SURREAL/summary/mask'
    label_dir = '../../dataset/SURREAL/summary/labels'

    tfrecord_base_dir = '../../dataset/SURREAL_tfrecord'
    cur_save_dir = 'surreal-2019-2-26'
    save_dir = os.path.join(tfrecord_base_dir,cur_save_dir)
    train_save_dir = os.path.join(save_dir,'train')
    val_save_dir = os.path.join(save_dir,'val')
    test_save_dir = os.path.join(save_dir,'test')

    train_names_fpath = '../../dataset/SURREAL/summary/train_frame_sample.json'
    val_names_fpath = '../../dataset/SURREAL/summary/val_frame_sample.json'
    test_names_fpath = '../../dataset/SURREAL/summary/test_frame_sample.json'
    print('Write',train_names_fpath,'to',train_save_dir)
    print('Write',val_names_fpath,'to',val_save_dir)
    print('Write',test_names_fpath,'to',test_save_dir)
    s_t = time.time()
    pool = multiprocessing.Pool(processes = 11)
    with open(val_names_fpath,'r') as f:
        val_names = json.load(f)
        save_to_tfrecord(pool,val_save_dir,val_names,im_dir,bodyseg_dir,human_dir,mask_dir,label_dir)
    with open(test_names_fpath,'r') as f:
        test_names = json.load(f)
        save_to_tfrecord(pool,test_save_dir,test_names,im_dir,bodyseg_dir,human_dir,mask_dir,label_dir)
    with open(train_names_fpath,'r') as f:
        train_names = json.load(f)
        save_to_tfrecord(pool,train_save_dir,train_names,im_dir,bodyseg_dir,human_dir,mask_dir,label_dir)
    pool.close()
    pool.join()
    print("All End. Cost",time.time()-s_t)
