
# coding: utf-8

# In[74]:

import numpy as np
import os
import pandas as pd
from bs4 import BeautifulSoup
import voc_utils
from more_itertools import unique_everseen
import cv2
import pdb

# In[3]:


root_dir = '/home/menna/Datasets/VIVID_DARPA/'
img_dir = os.path.join(root_dir, 'images')
ann_dir = os.path.join(root_dir, 'Annotations')
masks_dir = os.path.join(root_dir, 'masks')

# category name is from above, dataset is either "train" or
# "val" or "train_val"
def imgs_from_category(cat_name, dataset):
    filename = os.path.join(set_dir, cat_name + "_" + dataset + ".txt")
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=['filename', 'true'])
    return df

def imgs_from_category_as_list(cat_name, dataset):
    df = imgs_from_category(cat_name, dataset)
    df = df[df['true'] == 1]
    return df['filename'].values

def annotation_file_from_img(img_name):
    return os.path.join(ann_dir, img_name) + '.xml'

# annotation operations
def load_annotation(img_filename):
    xml = ""
    with open(annotation_file_from_img(img_filename)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml)

def get_all_obj_and_box(objname, img_set):
    img_list = imgs_from_category_as_list(objname, img_set)

    for img in img_list:
        annotation = load_annotation(img)

# image operations
def load_img(img_filename):
    return io.load_image(os.path.join(img_dir, img_filename + '.jpg'))

def draw_boxes(img_filename, bboxes):
    img= cv2.imread(img_filename)
    img_motion= np.zeros(img.shape[:2], dtype=np.uint8)
    for bb in bboxes:
        cv2.rectangle(img, (bb[0],bb[1]), (bb[2],bb[3]),(255,0,0))
        cv2.ellipse(img_motion, ((bb[2]+bb[0])//2,(bb[3]+bb[1])//2), (bb[2]-bb[0],bb[3]-bb[1]),0,0,360,(255,255,255),-1)
    return img, img_motion

def draw_polygons(img_filename, bboxes):
    img= cv2.imread(img_filename)
    img_motion= np.zeros(img.shape[:2], dtype=np.uint8)
    for xs, ys in bboxes:
        cv2.line(img, (xs[0],ys[0]), (xs[1],ys[1]),(255,0,0))
        cv2.line(img, (xs[1],ys[1]), (xs[2],ys[2]),(255,0,0))
        cv2.line(img, (xs[2],ys[2]), (xs[3],ys[3]),(255,0,0))
        cv2.line(img, (xs[3],ys[3]), (xs[0],ys[0]),(255,0,0))
        pts= np.concatenate((np.expand_dims(xs,axis=1),np.expand_dims(ys,axis=1)), axis=1)
        pts = pts.reshape((-1,1,2))
        #cv2.polylines(img_motion,[pts],True,(255,255,255))
        cv2.fillPoly(img_motion,[pts],(255,255,255))
    return img, img_motion

def parse_annotations():
    for d in os.listdir(img_dir):
        for count in range(0, 1820, 5):
            if not os.path.exists(ann_dir+'/'+d+'/frame%05d.xml'%count):
                continue

            anno = load_annotation(d+'/frame%05d'%count)
            objs = anno.findAll('object')
            boxes=[]
            for obj in objs:
                f_boxes= []
                obj_names = obj.findChildren('name')
                for name_tag in obj_names:
                    xs= []
                    ys= []
                    if str(name_tag.contents[0]) == 'Car':
                        fname = anno.findChild('filename').contents[0]
                        bbox = obj.findChildren('polygon')[0].findChildren('pt')

                        xs.append(int(float(bbox[0].findChildren('x')[0].contents[0])))
                        ys.append(int(float(bbox[0].findChildren('y')[0].contents[0])))

                        xs.append(int(float(bbox[1].findChildren('x')[0].contents[0])))
                        ys.append(int(float(bbox[1].findChildren('y')[0].contents[0])))

                        xs.append(int(float(bbox[2].findChildren('x')[0].contents[0])))
                        ys.append(int(float(bbox[2].findChildren('y')[0].contents[0])))

                        xs.append(int(float(bbox[3].findChildren('x')[0].contents[0])))
                        ys.append(int(float(bbox[3].findChildren('y')[0].contents[0])))

                        xs= np.asarray(xs)
                        ys= np.asarray(ys)
                        print('frame ',[fname, xs.shape, ys.shape])
                        boxes.append((xs,ys ))

            img_boxes, img_motion= draw_polygons(img_dir+'/'+d+'/frame%05d.jpg'%count, boxes)

            cv2.imshow('BBoxes ', img_boxes)
            cv2.imshow('motion ', img_motion)
            if not os.path.exists(masks_dir+'/'+d):
                os.mkdir(masks_dir+'/'+d)
            cv2.imwrite(masks_dir+'/'+d+'/frame%05d.jpg'%count, img_motion)
            cv2.waitKey(10)

parse_annotations()
