
# coding: utf-8
import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
from PIL import Image

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer

import skimage.io
import skimage.transform
import skimage.color
import skimage


def init_dataset(image_csv_file, class_csv_file):
    '''
        创建数据对象
        参数：
            image_csv_file： 需要测试图片信息，可以包含标注信息，也可以不包括，例如：
                ./dataset/chongyin/15050400/2_d68e121b909ee9c2.jpg,,,,,,
                ./dataset/biguashikongtiao/176249224_0070163647_1.jpg,18,98,377,233,biguashikongtiao
            class_csv_file： label信息，例如：
                baozhuang,0
                biguashikongtiao,1
                kongtiaoshan,2
                liguishikongtiao,3
                yaokongqi,4
    '''
    # 初始化dataset对象
    the_dataset = CSVDataset(train_file=image_csv_file, class_list=class_csv_file, transform=transforms.Compose([Normalizer(), Resizer()]))
    return the_dataset

def init_dataloader(the_dataset, batch_size=1, num_worker=1):
    '''
        创建数据加载对象
        参数：
            image_csv_file： 需要测试图片信息，可以包含标注信息，也可以不包括，例如：
                ./dataset/chongyin/15050400/2_d68e121b909ee9c2.jpg,,,,,,
                ./dataset/biguashikongtiao/176249224_0070163647_1.jpg,18,98,377,233,biguashikongtiao
            class_csv_file： label信息，例如：
                baozhuang,0
                biguashikongtiao,1
                kongtiaoshan,2
                liguishikongtiao,3
                yaokongqi,4
            batch_size : 设置batch_size大小
            num_worker： 加载训练数据的线程数量
    '''
    # 初始化Sampler对象
    the_sampler = AspectRatioBasedSampler(the_dataset, batch_size=batch_size, drop_last=False)
    # 创建dataloader对象
    the_dataloader = DataLoader(the_dataset, num_workers=num_worker, collate_fn=collater, batch_sampler=the_sampler)
    return the_dataloader

def init_model(model_file):
    '''
        创建模型对象
        参数：
            model_file： 模型文件保存路径
    '''
    use_gpu = True
    retinanet = torch.load(model_file)

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda() # 设置为多GPU的并行模式
    else:
        retinanet = torch.nn.DataParallel(retinanet)
    return retinanet


def detector_images(retinanet, the_dataset, the_data, thresh_score = 0.5):
    '''
        对输入数据，进行目标检测
        参数：
            the_dataset： 数据集对象，可以从中获取分类信息
            the_data： 字典类型，存储需要进行目标检测数据 {'img': 图片数据; 'image_path': 图片路径; 'scale': 缩放比例}
            thresh_score: 过滤box时使用的box
        返回：
            result_dict     key值： 图片路径        value：[[x1,y1,x2,y2, classname, score], .....]
    '''
    result_dict = dict()

    with torch.no_grad():
        st = time.time()
        if torch.cuda.is_available():
            the_result = retinanet(the_data['img'].cuda().float())
        else:
            the_result = retinanet(the_data['img'].float())

        print('Elapsed time: {}'.format(time.time()-st))
        for image_index, (scores, classification, transformed_anchors) in enumerate(the_result):
            idxs = np.where(scores.cpu()>thresh_score)
            
            image_path = the_data['image_path'][image_index]
            scale = the_data['scale'][image_index]
            
            if idxs[0].shape[0]==0:
                result_dict[image_path] = [[0,0,0,0, "None", .0]]
            else:
                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0]/scale)
                    y1 = int(bbox[1]/scale)
                    x2 = int(bbox[2]/scale)
                    y2 = int(bbox[3]/scale)
                    label_name = the_dataset.labels[int(classification[idxs[0][j]])]
                    if image_path in result_dict:
                        result_dict[image_path].append([x1,y1,x2,y2, label_name, scores[j]])
                    else:
                        result_dict[image_path]= [[x1,y1,x2,y2, label_name, scores[j]]]
    return result_dict

def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    # b[1]-20防止label超过上边界
    cv2.putText(image, caption, (b[0], b[1]-10 if b[1]-20>0 else 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1]-10 if b[1]-20>0 else 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.
    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')

def read_class_file(class_file):
    # parse the provided class file
    class_dict, label_dict = {}, {}
    try:
        with open_for_csv(class_file) as csv_reader:
            for line, row in enumerate(csv_reader):
                line += 1
                try:
                    class_name, class_id = row.strip().split(',')
                except ValueError:
                    raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
                class_id = int(class_id)

                if class_name in class_dict:
                    raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
                class_dict[class_name] = class_id
                label_dict[class_id] = class_name
    except ValueError as e:
        raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)
    return class_dict, label_dict

if __name__ == "__main__":
    '''
        该程序是为了验证 pytorch的数据加载逻辑
        通过程序，可以看到程序通过scikit包括读入图片数据后，通过tranform对象，转换数据，然后通过collater变成batch形式，然后统一输入到网络中
        另外可以看到pytorch的网络会根据输入的图片的数量，动态的调整输入占用显存，进行推理计算。
        另外这里的transform使用的转换函数都是在dataloader中自定义的。可能与pytorch的官方实现不一样
    '''
    # 去取class文件
    class_dict, label_dict =read_class_file("datasetv3/classes.csv")

    # 创建图像transform对象
    transform=transforms.Compose([Normalizer(), Resizer()])

    # 创建网络
    model_file = "csv_retinanet_65.pt"
    use_gpu = True
    retinanet = torch.load(model_file)

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda() # 设置为多GPU的并行模式
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    # 读取图片
    image_path = "datasetv3/add/badcase/404859041667234040256125_x.jpg"
    img = skimage.io.imread(image_path)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)
    img = img.astype(np.float32)/255.0

    # 创建图片信息
    im_dict = {'img':img, 'annot': np.array([[0,0,0,0,-1]], dtype='float64'), 'image_path':image_path}
    im_tensor = transform(im_dict)
    
    im_tensors = [im_tensor for i in range(10)]
    im_tensors = collater(im_tensors)

    # 前向传播
    result_dict = dict()
    with torch.no_grad():
        st = time.time()
        if torch.cuda.is_available():
            the_result = retinanet(im_tensors['img'].cuda().float())
        else:
            the_result = retinanet(im_tensors['img'].float())

        print('Elapsed time: {}'.format(time.time()-st))
        for image_index, (scores, classification, transformed_anchors) in enumerate(the_result):
            idxs = np.where(scores.cpu()>0.5)
            
            image_path = im_tensors['image_path'][image_index]
            scale = im_tensors['scale'][image_index]
            
            if idxs[0].shape[0]==0:
                im_tensors[image_path] = [[0,0,0,0, "None", .0]]
                #print("no bouding box in {}".format(result_dict[image_path]))
            else:
                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0]/scale)
                    y1 = int(bbox[1]/scale)
                    x2 = int(bbox[2]/scale)
                    y2 = int(bbox[3]/scale)
                    label_name = label_dict[int(classification[idxs[0][j]])]
                    if image_path in result_dict:
                        result_dict[image_path].append([x1,y1,x2,y2, label_name, scores[j]])
                    else:
                        result_dict[image_path]= [[x1,y1,x2,y2, label_name, scores[j]]]

    for index, image_path in enumerate(result_dict):
        
        img = cv2.imread(image_path)
        bboxes = result_dict[image_path]
        for box in bboxes:
            if box[4] != "None":
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                class_name = box[4]
                score = box[5]
                # 打印检测框信息
                print("image path: {}, box local: x1= {}, x2= {}, y1= {}, y2{}, class label= {}, score={}".format(image_path, x1, y1,x2, y2, class_name, score))
                
                # 在图像中显示检测框
                txt_draw = "%s %.2f" % (class_name, score)
                draw_caption(img, (x1, y1, x2, y2), txt_draw)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                
            result_path = os.path.join("result111",image_path)
            result_dir = os.path.dirname(result_path)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            cv2.imwrite(result_path, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)