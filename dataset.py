import json
import os
import numpy as np
import cv2
import tensorflow as tf
import random
from config import Config
from tf_smpl.smpl_np import SMPLModel
from tf_smpl.smpl_tf import smpl_model

def imread(path,im_shape):
    im = cv2.imread(path)
    if im is None:
        return None

    im = im/255
    im_pad = np.zeros(im_shape,dtype=np.float64)
    h,w = im.shape[:2]
    if h/w > im_shape[0]/im_shape[1]:
        re_h = im_shape[0]
        re_w = int(w * (re_h / h))
    else:
        re_w = im_shape[1]
        re_h = int(h * (re_w / w))
    re_im = cv2.resize(im,(re_w,re_h))
    im_pad[:re_h,:re_w,:] = re_im.copy()
    return im_pad

class TFRecordDataset(object):
    def __init__(self,files,config=Config()):
        files = self._preprocess_files(files)
        self.config = config
        self.dataset = tf.data.TFRecordDataset(files)
#         self.total_num = self._total_sample(files)
        self.dataset = self.dataset.map(lambda x: self._parse_example_proto(x))
        self.dataset = self.dataset.batch(self.config.param['BATCH_SIZE'])
        self.dataset = self.dataset.repeat()
        self.it = self.dataset.make_one_shot_iterator()
    def _preprocess_files(self,files):
        if not isinstance(files, list):
            files = [files]
        assert len(files) != 0,"Input can't be empty"
        new_files = []
        for file in files:
            if os.path.isdir(file):
                tfrecord_names = os.listdir(file)
                for name in tfrecord_names:
                    if name.endswith('tfrecords'):
                        new_files.append(os.path.join(file,name))
            elif os.path.isfile(file):
                if file.endswith('tfrecords'):
                    new_files.append(file)
        assert len(files) != 0,"Can't find tfrecord file from input. Input can include tfrecord file or dir which has tfrecord file or both file and dir."
        return new_files
    def _parse_example_proto(self,example_serialized):
        feature_map = {
            'label/pose': tf.FixedLenFeature((72,), dtype=tf.float32),
            'label/shape': tf.FixedLenFeature((10,), dtype=tf.float32),
#             'label/gt2d': tf.FixedLenFeature((24 * 2,), dtype=tf.float32),
#             'label/gt3d': tf.FixedLenFeature((24 * 3,), dtype=tf.float32),
        }
        if 'ori' == self.config.param['IMAGE_TYPE']:
            key_string = 'image'
        if 'body' == self.config.param['IMAGE_TYPE']:
            key_string = 'image/bodyseg'
        if 'human' == self.config.param['IMAGE_TYPE']:
            key_string = 'image/human'
        if 'mask' == self.config.param['IMAGE_TYPE']:
            key_string = 'image/mask'
        if 'ori_mask' == self.config.param['IMAGE_TYPE']:
            key_string = 'image/ori_mask'

        feature_map[key_string] = tf.FixedLenFeature([], dtype=tf.string, default_value='')
        feature_map[key_string + '/h'] = tf.FixedLenFeature([], dtype=tf.int64)
        feature_map[key_string + '/w'] = tf.FixedLenFeature([], dtype=tf.int64)
        feature_map[key_string + '/c'] = tf.FixedLenFeature([], dtype=tf.int64)

        features = tf.parse_single_example(example_serialized, feature_map)

        image = tf.image.decode_image(features[key_string])
        h = tf.cast(features[key_string + '/h'],tf.int64)
        w = tf.cast(features[key_string + '/w'],tf.int64)
        c = tf.cast(features[key_string + '/c'],tf.int64)
        image = tf.reshape(image, [h, w, c])

        pose = tf.cast(features['label/pose'], dtype=tf.float32)
        shape = tf.cast(features['label/shape'], dtype=tf.float32)
        trans = np.zeros(3)
        trans = tf.constant(trans, dtype=tf.float32)
        verts_ori = smpl_model('./tf_smpl/models/neutral_smpl_with_cocoplus_reg.pkl',shape,pose,trans)
#         verts = tf.reshape(verts_ori,shape=(-1,))
        verts = verts_ori
#         gt2d = tf.reshape(tf.cast(features['label/gt2d'], dtype=tf.float16), [24, 2])
#         gt3d = tf.reshape(tf.cast(features['label/gt3d'], dtype=tf.float16), [24, 3])
        if self.config.param['MODEL_NAME'] == 'ip_sv':
            return ({"image":image,"pose":pose},{"shape":shape,"verts":verts})
        if self.config.param['MODEL_NAME'] == 'i_spv':
            return ({"image":image},{"shape":shape,"pose":pose,"verts":verts})
    def _total_sample(self,file_names):
        if not isinstance(file_names, list):
            file_names = [file_names]
        sample_nums = 0
        for file_name in file_names:
            for record in tf.python_io.tf_record_iterator(file_name):
                sample_nums += 1
        return  sample_nums

class surreal_sequence(tf.keras.utils.Sequence):
    def __init__(self,json_path,mask_dir,bodyseg_dir,label_dir,config = Config()):
        self.config = config
        self.smpl = SMPLModel('./tf_smpl/models/neutral_smpl_with_cocoplus_reg.pkl')
        
        self.mask_dir = mask_dir
        self.bodyseg_dir = bodyseg_dir
        self.label_dir = label_dir
        
        f = open(json_path,'r')
        self.names = json.load(f)
        f.close()
        self.names_num = len(self.names)
        self.index = 0

    def __len__(self):
        return int(len(self.names) / self.config.param['BATCH_SIZE']) # the length is the number of batches

    def __getitem__(self, batch_id):
        images = []
        pose_list = []
        shape_list = []
        verts_list = []
        cur_size = 0
        while cur_size < self.config.param['BATCH_SIZE']:
            self.index = self.index % self.names_num
            if self.index >= self.names_num:
                self.index = 0
            base_name = self.names[self.index]
            mask_im_path = os.path.join(self.mask_dir,base_name+'.png')
            bodyseg_im_path = os.path.join(self.bodyseg_dir,base_name+'.png')
            label_path = os.path.join(self.label_dir,base_name+'.json')
            if not os.path.exists(label_path):
                self.index += 1
                continue
            if 'mask' == self.config.param['IMAGE_TYPE']:
                im = imread(bodyseg_im_path,self.config.param['INPUT_SHAPE'])
                if im is None:
                    self.index += 1
                    continue
                im[im != 0] = 1
            if 'body' == self.config.param['IMAGE_TYPE']:
                im = imread(bodyseg_im_path,self.config.param['INPUT_SHAPE'])
                if im is None:
                    self.index += 1
                    continue
            if 'human' == self.config.param['IMAGE_TYPE']:
                im = imread(mask_im_path,self.config.param['INPUT_SHAPE'])
                if im is None:
                    self.index += 1
                    continue
            if 'ori_mask' == self.config.param['IMAGE_TYPE']:
                im = imread(bodyseg_im_path,self.config.param['INPUT_SHAPE'])
                if im is None:
                    self.index += 1
                    continue
                a1,a2,a3 = np.where(im != 0)
                im[a1,a2,2] = 1
            with open(label_path,'r') as f:
                info = json.load(f)
#             label = [ i * 100 for i in info['shape'] ]
            shape = np.array(info['shape'])
            pose = np.array(info['pose'])
            trans = np.zeros(self.smpl.trans_shape)
            self.smpl.set_params(beta=shape, pose=pose, trans=trans)
            verts = self.smpl.verts
            images.append(im)
            shape_list.append(shape)
            pose_list.append(pose)
            verts_list.append(verts.flatten())
            self.index += 1
            cur_size += 1
        images = np.array(images)
        shape_np = np.array(shape_list)
        pose_np = np.array(pose_list)
        verts_np = np.array(verts_list)
        if self.config.param['MODEL_NAME'] == 'ip_sv':
            return ({"image":images,"pose":pose_np},{"shape":shape_np,"verts":verts_np})
        if self.config.param['MODEL_NAME'] == 'i_spv':
            return ({"image":images},{"shape":shape_np,"pose":pose_np,"verts":verts_np})
