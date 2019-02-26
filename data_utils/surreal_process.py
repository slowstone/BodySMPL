import os
import scipy.io as sio
import json
import cv2
import logging
import subprocess
import numpy as np
import multiprocessing
import random

def statistic(pool,dir_s = False):
    image_dir = './summary/image/'
    label_dir = './summary/labels/'
    bodyseg_dir = './summary/bodyseg'
    mask_dir = './summary/mask'
    human_dir = './summary/human'
    dir_list = [image_dir,label_dir,bodyseg_dir,mask_dir,human_dir]
    logging.info("=============> Statistic")
    if dir_s:
        for dir_name in dir_list:
            pool.apply_async(multi_process_st,(dir_name,))
    base_dir = './summary'
    names = os.listdir(base_dir)
    for name in names:
        if 'json' in name:
            pool.apply_async(multi_process_st_json,(base_dir,name,))

def multi_process_st(dir_name):
    names = os.listdir(dir_name)
    logging.info("=============> The number in {}: {}".format(dir_name,len(names)))

def multi_process_st_json(base_dir,name):
    json_path = os.path.join(base_dir,name)
    with open(json_path,'r') as f:
        info = json.load(f)
    logging.info("=============> The number in {}: {}".format(name,len(info)))

def video_cut(pool):
    root_path = "./data/cmu/"
    outputs = os.walk(root_path)

    stdfile_dir = './stdout'
    if not os.path.exists(stdfile_dir):
        os.mkdir(stdfile_dir)

    output_im_dir = './summary/image/'
    output_json_dir = './summary/labels/'
    if not os.path.exists(output_im_dir):
        os.mkdir(stdfile_dir)
    if not os.path.exists(output_json_dir):
        os.mkdir(stdfile_dir)
    for output in outputs:
        pre_dir = output[0].split('/')[-2]
        if len(output[1]) != 0:
            continue
        pool.apply_async(multi_process_sbmh,(output,stdfile_dir,pre_dir,output_im_dir,output_json_dir,))

def multiprocess_vc(output,stdfile_dir,pre_dir,output_im_dir,output_json_dir):
    logging.info("=============> Running in {}".format(output[0]))
    for f_name in output[2]:
        try:
            file_path = os.path.join(output[0],f_name)
            if '.mp4' in f_name:
                frame_path = os.path.join(output_im_dir,"{}_{}_frame%4d.jpg".format(pre_dir,f_name[:-4]))
                stdfile = os.path.join(stdfile_dir,"{}_{}".format(pre_dir,f_name[:-4]))
                os.system('ffmpeg -i {} {} > {} 2>&1'.format(file_path,frame_path,stdfile))
                os.system('rm {}'.format(stdfile))
            if '_info.mat' in f_name:
                data = sio.loadmat(file_path)
                for index in range(data['pose'].shape[1]):
                    cur_f_name = '{}_{}_frame{:0>4}.json'.format(pre_dir,f_name[:-9],index+1)
                    json_path = os.path.join(output_json_dir,cur_f_name)
                    json_infos = {}
                    json_infos['shape'] = data['shape'][:,index].tolist()
                    json_infos['pose'] = data['pose'][:,index].tolist()
                    json_infos['joints3D'] = data['joints3D'][:,:,index].tolist()
                    json_infos['joints2D'] = data['joints2D'][:,:,index].tolist()
                    f = open(json_path,'w')
                    f.write(json.dumps(json_infos,indent=2))
                    f.close()
        except Exception as e:
            logging.error("-------> {} {}".format(file_path,e))

def filter_image(pool,Threshold=0.1,buffer_num=1000):
    bodyseg_dir = './summary/bodyseg'
    bodyseg_names = os.listdir(bodyseg_root_path)

    names = []
    for name in bodyseg_names():
        base_name = name[:-4]
        names.append(base_name)
        if len(names) == buffer_num:
            pool.apply_async(multi_process_fi,(names,Threshold,))
            names = []

def multi_process_fi(names,Threshold):
    bodyseg_dir = './summary/bodyseg'
    for base_name in names:
        path = os.path.join(bodyseg_dir,base_name+'.png')
        try:
            im = cv2.imread(path,-1)
            scale = np.sum(im) / (im.shape[0] * im.shape[1])
            if scale < Threshold :
                remove_file(base_name)
        except Exception as e:
            logging.error("---------> {} {}".format(path.split('/')[-1][:-4],e))

def remove_file(base_name):
    image_dir = './summary/image/'
    label_dir = './summary/labels/'
    bodyseg_dir = './summary/bodyseg'
    mask_dir = './summary/mask'
    human_dir = './summary/human'
    image_path = os.path.join(image_dir,base_name+".jpg")
    label_path = os.path.join(label_dir,base_name+".json")
    bodyseg_path =  os.path.join(bodyseg_dir,base_name+".png")
    mask_path = os.path.join(mask_dir,base_name+".png")  
    human_path =  os.path.join(human_dir,base_name+".jpg")
    logging.info('=========> Remove {}'.format(base_name))
    os.system("rm {}".format(image_path))
    os.system("rm {}".format(label_path))
    os.system("rm {}".format(bodyseg_path))
    os.system("rm {}".format(mask_path))
    os.system("rm {}".format(human_path))

def match(pool,buffer_num=1000):
    image_dir = './summary/image/'
    label_dir = './summary/labels/'
    bodyseg_dir = './summary/bodyseg'
    mask_dir = './summary/mask'
    human_dir = './summary/human'
    dir_list = [image_dir,label_dir,bodyseg_dir,mask_dir,human_dir]

    for dir_name in dir_list:
        logging.info("===============> Begin match statistics: base dir {}".format(dir_name))
        names_in_dir = os.listdir(dir_name)
        im_len = len(names_in_dir)
        logging.info("===============> The number in {}: {}".format(dir_name,im_len))

        names = []
        for i,name in enumerate(names_in_dir):
            base_name = name[:-4]
            names.append(base_name)
            if len(names) == buffer_num:
                pool.apply_async(multiprocess_match,(names,image_dir,label_dir,bodyseg_dir,mask_dir,human_dir,))
                names = []

def multiprocess_match(names,image_dir,label_dir,bodyseg_dir,mask_dir,human_dir):
    for base_name in names:
        try:
            flag = False
            image_path = os.path.join(image_dir,base_name+".jpg")
            label_path = os.path.join(label_dir,base_name+".json")
            bodyseg_path =  os.path.join(bodyseg_dir,base_name+".png")
            mask_path = os.path.join(mask_dir,base_name+".png")  
            human_path =  os.path.join(human_dir,base_name+".jpg")
            if flag or not os.path.exists(image_path):
                flag = True
            if flag or not os.path.exists(label_path):
                flag = True
            if flag or not os.path.exists(bodyseg_path):
                flag = True
            if flag or not os.path.exists(mask_path):
                flag = True
            if flag or not os.path.exists(human_path):
                flag = True
            if flag:
                remove_file(base_name)
        except Exception as e:
            logging.error("---------> {} {}".format(base_name,e))

def save_body_mask_humanim(pool):
    bodyseg_save_dir = './summary/bodyseg'
    mask_save_dir = './summary/mask'
    ori_save_dir = './summary/image'
    human_save_dir = './summary/human'
    root_path = './data/cmu/'
    if not os.path.exists(bodyseg_save_dir):
        os.mkdir(stdfile_dir)
    if not os.path.exists(mask_save_dir):
        os.mkdir(stdfile_dir)
    if not os.path.exists(human_save_dir):
        os.mkdir(stdfile_dir)
    outputs = os.walk(root_path)

    for output in outputs:
        pre_dir = output[0].split('/')[-2]
        if len(output[1]) != 0:
            continue
        pool.apply_async(multi_process_sbmh,(output,pre_dir,ori_save_dir,bodyseg_save_dir,mask_save_dir,human_save_dir,))


def multi_process_sbmh(output,pre_dir,ori_save_dir,bodyseg_save_dir,mask_save_dir,human_save_dir):
    logging.info("==============> Running in {}".format(output[0]))
    for f_name in output[2]:
        if '_segm.mat' in f_name:
            file_path = os.path.join(output[0],f_name)
            data = sio.loadmat(file_path)
            for key in data.keys():
                if 'segm' not in key:
                    continue
                try:
                    ori_im_path = os.path.join(ori_save_dir,'{}_{}_frame{:0>4}.jpg'.format
                                      (pre_dir,f_name[:-9],key.split('_')[-1]))
                    ori_im = cv2.imread(ori_im_path)
                    if ori_im is None:
                        continue
                    bodyseg = data[key]
                    f_path = os.path.join(bodyseg_save_dir,'{}_{}_frame{:0>4}.png'.format
                                      (pre_dir,f_name[:-9],key.split('_')[-1]))
                    if os.path.exists(f_path):
                        continue
                    cv2.imwrite(f_path,bodyseg)
                    im = bodyseg.copy()
                    im[im !=0] = 1
                    human_im = ori_im.copy()
                    human_im[:,:,0] = ori_im[:,:,0] * im
                    human_im[:,:,1] = ori_im[:,:,1] * im
                    human_im[:,:,2] = ori_im[:,:,2] * im
                    x_list,y_list = np.where(im == 1)
                    x = np.min(x_list)
                    y = np.min(y_list)
                    xh = np.max(x_list)
                    yh = np.max(y_list)
                    im = im[x:xh,y:yh]
                    human_im = human_im[x:xh,y:yh,:]
                    im_path = os.path.join(mask_save_dir,'{}_{}_frame{:0>4}.png'.format
                                      (pre_dir,f_name[:-9],key.split('_')[-1]))
                    cv2.imwrite(im_path,im)
                    human_path = os.path.join(human_save_dir,'{}_{}_frame{:0>4}.jpg'.format
                                      (pre_dir,f_name[:-9],key.split('_')[-1]))
                    cv2.imwrite(human_path,human_im)
                except Exception as e:
                    logging.error("---------> {},{},{}".format(file_path,key,e))

def patition(val_rate = None):
    logging.info("=======================>Getting base names")
    f = open('./summary/all_names.json','r')
    base_names = json.load(f)
    f.close()
    """
    im_dir = './summary/image/'
    logging.info("=======================>Getting all names in {}".format(im_dir))
    names = os.listdir(im_dir)
    base_names = []
    logging.info("=======================>Getting base names")
    for i,name in enumerate(names):
        base_names.append(name[:-4])
    """
    num = len(base_names)
    base_names.sort()

    if val_rate is None:
        val_num = 10000
    else:
        val_num = num*val_rate
    logging.info("=======================>Getting val names")
    val_names = base_names[:val_num]
#    val_names = random.sample(base_names,val_num)
#    train_names = []
    logging.info("=======================>Getting train names")
    train_names = base_names[val_num:100000]
    """
    for i,name in enumerate(base_names):
        if name not in val_names:
            train_names.append(name)
    """
    logging.info("===================>all:{},train:{},val:{}".format(num,len(val_names),len(train_names)))
    """
    f = open('./summary/all_names.json','w')
    f.write(json.dumps(base_names))
    f.close()
    """
    f = open('./summary/train_names_sort_100000.json','w')
    f.write(json.dumps(train_names))
    f.close()
    f = open('./summary/val_names_sort_100000.json','w')
    f.write(json.dumps(val_names))
    f.close()

def ori_patition():
    logging.info("=======================>Start")
    root_path = "./data/cmu/"
    outputs = os.walk(root_path)
    res = {}
    for output in outputs:
        try:
            pre_dir = output[0].split('/')[-2]
            if len(output[1]) != 0:
                continue
            ty = output[0].split('/')[-3]
            if ty not in res.keys():
                res[ty] = []
                logging.info("=======================>Add: key {}".format(ty))
            for f_name in output[2]:
                if '.mp4' in f_name:
                    res[ty].append("{}_{}".format(pre_dir,f_name[:-4]))
        except Exception as e:
            logging.error("---------------------> {},{}".format(output,e))
    for key in res.keys():
        with open('./summary/{}.json'.format(key),'w') as f:
            logging.info("==========================> Write to {}, number: {}".format(key,len(res[key])))
            f.write(json.dumps(res[key]))

def ori_patition_frame():
    logging.info("=======================>Start")
    json_path = "./summary/all_names.json"
    with open(json_path,'r') as f:
        names = json.load(f)
    train_path = "./summary/train.json"
    with open(train_path,'r') as f:
        train_video_names = json.load(f)
    test_path = "./summary/test.json"
    with open(test_path,'r') as f:
        test_video_names = json.load(f)
    val_path = "./summary/val.json"
    with open(val_path,'r') as f:
        val_video_names = json.load(f)
    train_names = []
    test_names = []
    val_names = []
    for name in names:
        for train_video_name in train_video_names:
            if train_video_name in name:
                train_names.append(name)
        for test_video_name in test_video_names:
            if test_video_name in name:
                test_names.append(name)
        for val_video_name in val_video_names:
            if val_video_name in name:
                val_names.append(name)
    with open("./summary/train_frame.json",'w') as f:
        logging.info("==========================> Write to train_frame, number: {}".format(len(train_names)))
        f.write(json.dumps(train_names))
    with open("./summary/test_frame.json",'w') as f:
        logging.info("==========================> Write to test_frame, number: {}".format(len(test_names)))
        f.write(json.dumps(test_names))
    with open("./summary/val_frame.json",'w') as f:
        logging.info("==========================> Write to val_frame, number: {}".format(len(val_names)))
        f.write(json.dumps(val_names))

def ori_patition_sampling():
    logging.info("=======================>Start")
    json_path = "./summary/all_names.json"
    with open(json_path,'r') as f:
        names = json.load(f)
    train_path = "./summary/train.json"
    with open(train_path,'r') as f:
        train_video_names = json.load(f)
    test_path = "./summary/test.json"
    with open(test_path,'r') as f:
        test_video_names = json.load(f)
    val_path = "./summary/val.json"
    with open(val_path,'r') as f:
        val_video_names = json.load(f)
    train_names = []
    test_names = []
    val_names = []
    for name in train_video_names:
        for i in random.sample(range(100),10):
            frame_name = "{}_frame{:0>4}".format(name,i)
            if frame_name in names:
                train_names.append(frame_name)
    for name in test_video_names:
        for i in random.sample(range(100),10):
            frame_name = "{}_frame{:0>4}".format(name,i)
            if frame_name in names:
                test_names.append(frame_name)
    for name in val_video_names:
        for i in random.sample(range(100),10):
            frame_name = "{}_frame{:0>4}".format(name,i)
            if frame_name in names:
                val_names.append(frame_name)
    with open("./summary/train_frame_sample.json",'w') as f:
        logging.info("==========================> Write to train_frame_sample, number: {}".format(len(train_names)))
        f.write(json.dumps(train_names))
    with open("./summary/test_frame_sample.json",'w') as f:
        logging.info("==========================> Write to test_frame_sample, number: {}".format(len(test_names)))
        f.write(json.dumps(test_names))
    with open("./summary/val_frame_sample.json",'w') as f:
        logging.info("==========================> Write to val_frame_sample, number: {}".format(len(val_names)))
        f.write(json.dumps(val_names))


def cv_multi(names,json_dir,new_json_dir,smpl):
    for name in names:
        try:
            path = os.path.join(json_dir,name+'.json')
            with open(path,'r') as f:
                json_info = json.load(f)
            trans = np.zeros(smpl.trans_shape)
            shape = np.array(json_info['shape'])
            pose = np.array(json_info['pose'])
            verts = smpl.set_params(beta=shape, pose=pose, trans=trans)
            json_info['verts'] = verts.tolist()
            new_path = os.path.join(new_json_dir,name+'.json')
            with open(new_path,'w') as f:
                f.write(json.dumps(json_info))
        except Exception as e:
            logging.error("========================> {}".format(e))

def computer_verts(pool,buffernum):
    from smpl_np import SMPLModel
    logging.info("=======================>Getting computer verts")
    smpl = SMPLModel('./base_model.pkl')
    f = open('./summary/all_names.json','r')
    base_names = json.load(f)
    f.close()
    json_dir = './summary/labels/'
    new_json_dir = './summary/labels-verts'
    names = []
#     pathes = []
    for base_name in base_names:
#         path = os.path.join(json_dir,base_name+'.json')
#         pathes.append(path)
#         if len(pathes) == buffernum:
#             pool.apply_async(cv_multi,(pathes,))
#             pathes = []
        names.append(base_name)
        if len(names) == buffernum:
            pool.apply_async(cv_multi,(names,json_dir,new_json_dir,smpl,))
            names = []
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','--m', type=str, help='  1. s: statistic \
                                    2. vc:  video_cut. \
                                    3. fi: filter_image(Threshold) \
                                    4. match: match \
                                    5. sbmh: save_body_mask_human (multi) \
                                    6. p: partition. \
                                    7. cv: computer_verts \
                                    8. op: ori_patition \
                                    9. opf: ori_patition_frame \
                                    10. ops: ori_patition_sampling'
                       )
    parser.add_argument('--threshold','--t',type=float, help='Threshold of scale',default=0.1)
    parser.add_argument('--buffernum','--bn',type=int, help='The buffernum of per process',default=1000)
    parser.add_argument('--num_workers','--nw',type=int, help='The number of processes',default=10)
    parser.add_argument('--val_rate','--vr',type=float, help='The rate of val/all',default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s %(filename)s[line:%(lineno)d]-{} %(levelname)s %(message)s'.format(args.model),datefmt='%a, %d %b %Y %H:%M:%S')

    pool = multiprocessing.Pool(processes = args.num_workers)
    if args.model == 's':
        statistic(pool)
    elif args.model == 'vc':
        video_cut(pool)
    elif args.model == 'fi':
        filter_image(pool,args.threshold,args.buffernum)
    elif args.model == 'match':
        match(pool,args.buffernum)
    elif args.model == 'sbmh':
        save_body_mask_humanim(pool)
    elif args.model == 'p':
        patition(args.val_rate)
    elif args.model == 'op':
        ori_patition()
    elif args.model == 'opf':
        ori_patition_frame()
    elif args.model == 'ops':
        ori_patition_sampling()
    elif args.model == 'cv':
        computer_verts(pool,args.buffernum)
    else:
        parser.print_help()
    pool.close()
    pool.join()

