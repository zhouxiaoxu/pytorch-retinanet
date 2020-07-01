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
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')
	parser.add_argument('--imagesavepath', help='path to save detection images')

	parser = parser.parse_args(args)

	if parser.imagesavepath:
		os.makedirs(parser.imagesavepath, exist_ok=True)
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
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		# b[1]-20防止label超过上边界
		cv2.putText(image, caption, (b[0], b[1]-10 if b[1]-20>0 else 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1]-10 if b[1]-20>0 else 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
	'''
	for idx, data in enumerate(dataloader_val):
		#print(data)
		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			else:
				scores, classification, transformed_anchors = retinanet(data['img'].float())
			print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores.cpu()>0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
			img_path = data['image_path'][0]
			
			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				txt_draw = "%s %.2f" %(label_name, scores[j])
				draw_caption(img, (x1, y1, x2, y2), txt_draw)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

				print(label_name)
			if parser.imagesavepath:
				image_dir = os.path.join(parser.imagesavepath, os.path.dirname(img_path))
				if not os.path.exists(image_dir):
					os.makedirs(image_dir)
				cv2.imwrite(os.path.join(parser.imagesavepath, "{}.jpg".format(img_path)), img)
				print("create result image:{} ".format(os.path.join(parser.imagesavepath, img_path)))
				print("**** Final funtion： img {},  shape: {}".format(img_path, img.shape))
			else:
				cv2.imshow('img', img)
				cv2.waitKey(0)
				
	'''
	for idx, data in enumerate(dataloader_val):
		#print(data)
		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			else:
				scores, classification, transformed_anchors = retinanet(data['img'].float())
			print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores.cpu()>0.5)
			#img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
			image_path = data['image_path'][0]
			scale = data['scale'][0]
			
			#img[img<0] = 0
			#img[img>255] = 255

			#img = np.transpose(img, (1, 2, 0))

			#img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
			img = cv2.imread(image_path)

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

				print(label_name)
			if parser.imagesavepath:
				new_path = os.path.join(parser.imagesavepath, os.path.basename(image_path))
				cv2.imwrite(new_path, img)
				#print("create result image:{} ".format(new_path))
				#print("**** Final funtion： img {},  shape: {}".format(image_path, img.shape))
			else:
				cv2.imshow('img', img)
				cv2.waitKey(0)
	



if __name__ == '__main__':
 main()