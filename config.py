import json
import os

class Config(object):
    def __init__(self,mode='train'):
        self.mode = mode
        self.param = {}
        self.param['INPUT_SHAPE'] = (240,240,3)
        # h w c
        self.param['IMAGE_TYPE'] = 'ori_mask'
        # ori body human mask ori_mask
        self.param['MODEL_NAME'] = 'ip_sv'
        # input_***-output_***
        # ip_sv:   image_pose-shape_verts
        # i_spv:   image-shape_pose_verts
        self.param['BETA_NUMS'] = 10
        self.param['POSE_NUMS'] = 72
#         self.param['MODEL_PATH'] = './logs/regress_up_20181205T0934/ep_0100.h5'
#         self.param['MODEL_PATH'] = './logs/mask-body_ip_sv_20190118T2011/ep_0002.h5'
        self.param['MODEL_PATH'] = None

        if mode == 'train':
            self.train_init()

    def train_init(self):
        self.param['GPUS'] = '0,1'
        self.param['BATCH_SIZE'] = 400
        self.param['DATA_VERSION'] = 'surreal_tfrecord'
        """
        # surreal, surreal_tfrecord
        """
        """
        # adam, momentum or nesterov
        """
        if self.param['MODEL_NAME'] == 'ip_sv':
            self.param['LOSS_WEIGHTS'] = [1,10]
        if self.param['MODEL_NAME'] == 'i_spv':
            self.param['LOSS_WEIGHTS'] = [1,1,10]
        self.param['OPT_STRING'] = "momentum"

        self.param['LEARNING_RATE'] = 1
        #using in lr decay
        self.param['LR_DECAY'] = 0.1
        #using in momentum and nesterov
        self.param['MOMENTUM'] = 0.9
        #using in adam
        self.param['ADAM_BETA_1'] = 0.9
        self.param['ADAM_BETA_2'] = 0.99

        self.param['L2_SCALE'] = 0.01

        #base hyper-parameter
        self.param['MEAN_PIXEL'] = [93.2,104.6,116.6]


        self.param['TRAIN_STEPS'] = 10000
        self.param['VALIDATION_STEPS'] = 1000
        self.param['EPOCHS'] = 100

    def show_config(self):
        print("\n============== Param =============")
        print("===========>config mode",self.mode)
        for key in self.param.keys():
            print(key,":",self.param[key])
        print("==================================\n")

    def set_config(self,file_path):
        print("\n======set config from file=======")
        print(file_path)
        f = open(file_path,'r')
        json_infos = json.load(f)
        for key in self.param:
            self.param[key] = json_infos[key]
        print("==================================\n")

    def save_config(self,out_dir = './',out_name = 'config.json'):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = os.path.join(out_dir,out_name)
        f = open(out_path,'w')
        f.write(json.dumps(self.param,indent=2))

    def set_param(self,keys,datas):
        for i,key in enumerate(keys):
            self.param[key] = datas[i]
            print("set",key,"to",datas[i])
