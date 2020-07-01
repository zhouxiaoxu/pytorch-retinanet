
# coding: utf-8
import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer


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

if __name__ == "__main__":
    # 创建数据集对象
    test_image_csv = "datasetv3/annotations_test.csv"
    class_csv = "datasetv3/classes.csv"
    the_dataset = init_dataset(test_image_csv, class_csv)

    # 创建数据加载对象
    num_workes = 1
    batch_size = 2
    the_dataloader = init_dataloader(the_dataset, batch_size, num_workes)

    # 创建模型对象
    model_file = "csv_retinanet_65.pt"
    retinanet = init_model(model_file)

    # 按批次进行目标检测
    for idx, data in enumerate(the_dataloader):
        result = detector_images(retinanet, the_dataset, data)   

        
        for index, image_path in enumerate(result):
            
            img = cv2.imread(image_path)
            bboxes = result[image_path]
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
                    
                result_path = os.path.join("result_val",image_path)
                result_dir = os.path.dirname(result_path)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                cv2.imwrite(result_path, img)
                #cv2.imshow('img', img)
                #cv2.waitKey(0)
