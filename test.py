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


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	'''
		test.py 会计算原始图片中的box的位置，而visualize.py返回的是resize和padding后的图片boudning box 位置
		另外test.py支持对识别结果的保存
	'''
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')
	parser.add_argument('--resultsavepath', help='path to save detection images')
	parser.add_argument('--thresh_score', help="thresh score", type=float, default=.5)

	parser = parser.parse_args(args)

    # 创建结果的保存路径
	if parser.resultsavepath:
		os.makedirs(parser.resultsavepath, exist_ok=True)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		#dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))	# 提示错误
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	retinanet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		if torch.cuda.is_available():
			retinanet = retinanet.cuda()

	
	if torch.cuda.is_available():
		retinanet = torch.nn.DataParallel(retinanet).cuda() # 设置为多GPU的并行模式
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.eval()# 设置为评估模式

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		# b[1]-20防止label超过上边界
		cv2.putText(image, caption, (b[0], b[1]-10 if b[1]-20>0 else 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1]-10 if b[1]-20>0 else 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
	
	if parser.resultsavepath:
		result_csv = "{}_result.csv".format(os.path.splitext(os.path.basename(parser.csv_val))[0])
		result_csv = os.path.join(parser.resultsavepath, result_csv)
		print(result_csv)
		result_csv_fd = open(result_csv, 'w')	
		
	for idx, data in enumerate(dataloader_val):
		#print("data shape:", data.shape)
		print(data['image_path'])
		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				the_result = retinanet(data['img'].cuda().float())
			else:
				the_result = retinanet(data['img'].float())

			print('Elapsed time: {}'.format(time.time()-st))
			for image_index, (scores, classification, transformed_anchors) in enumerate(the_result):
				idxs = np.where(scores.cpu()>parser.thresh_score)
				
				image_path = data['image_path'][image_index]
				scale = data['scale'][image_index]
				
					
				img = cv2.imread(image_path)
				
				if idxs[0].shape[0]==0:
					result_csv_fd.write("{},,,,,\n".format(image_path))
				else:
					for j in range(idxs[0].shape[0]):
						bbox = transformed_anchors[idxs[0][j], :]
						x1 = int(bbox[0]/scale)
						y1 = int(bbox[1]/scale)
						x2 = int(bbox[2]/scale)
						y2 = int(bbox[3]/scale)
						label_name = dataset_val.labels[int(classification[idxs[0][j]])]
						txt_draw = "%s %.2f" %(label_name, scores[j])
						draw_caption(img, (x1, y1, x2, y2), txt_draw)
						
						cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
						if parser.resultsavepath:
							result_csv_fd.write("{},{},{},{},{},{}\n".format(image_path,x1,y1,x2,y2,label_name))
						print(label_name)
				if parser.resultsavepath:
					new_dir = os.path.join(parser.resultsavepath, os.path.dirname(image_path))
					if not os.path.exists(new_dir):
						os.makedirs(new_dir)
					new_path = os.path.join(parser.resultsavepath, image_path)
					cv2.imwrite(new_path, img)
					
					#new_path = os.path.join(parser.resultsavepath, os.path.basename(image_path))
					#cv2.imwrite(new_path, img)
					
					#print("create result image:{} ".format(new_path))

				else:
					cv2.imshow('img', img)
					cv2.waitKey(0)
	if parser.resultsavepath:
		result_csv_fd.close()
	



if __name__ == '__main__':
 main()