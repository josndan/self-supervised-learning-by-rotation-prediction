import json
from os import listdir
from os.path import isfile, join
from pathlib import Path
from PIL import Image
from PIL import ImageFile
import numpy as np
import glob
import os.path
from torch.utils.data import Dataset
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True
INPUT_SIZE = 128

class COCO(Dataset):
  #dog =1,cat =0
  def __init__(self,root,split,transform):
    self.x_path = root+'/'+ split + '/data'
    self.split = split
    if split !="test":
      self.y_path = root +'/'+ split + '/labels.json'
      f = open(self.y_path)
      self.labels_dir = json.load(f)
    
    self.transform = transform


    self.x_file_paths = [f for f in listdir(self.x_path) if isfile(join(self.x_path, f))]
    dog_image_ids = set()
    cat_image_ids = set()
    dog_bounding_boxes = {}
    cat_bounding_boxes = {}

    if split !="test":
      for ann in self.labels_dir["annotations"]:
        f_name = self.labels_dir["images"][ann["image_id"]-1]["file_name"]

        image_width = self.labels_dir["images"][ann["image_id"]-1]["width"]
        image_height = self.labels_dir["images"][ann["image_id"]-1]["height"]
        x_ratio = INPUT_SIZE/image_width
        y_ratio = INPUT_SIZE/image_height

        x,y,width,height = ann["bbox"]
        x_modified, y_modified, width_modified, height_modified = x*x_ratio,y*y_ratio,width*x_ratio,height*y_ratio

        if ann["category_id"] == 18:
          dog_image_ids.add(f_name)
          dog_bounding_boxes[f_name] = [x_modified, y_modified, width_modified, height_modified]
        elif ann["category_id"] == 17:
          cat_image_ids.add(f_name)
          cat_bounding_boxes[f_name] =  [x_modified, y_modified, width_modified, height_modified]

    self.imgs = []
    self.boxes = []
    
    labels = []
    for i,img_path in enumerate(self.x_file_paths):
      if split != "test":
        if img_path in dog_image_ids and img_path not in cat_image_ids:
          labels.append((1,0))
        elif img_path in cat_image_ids and img_path not in dog_image_ids :
          labels.append((0,1))
        else:
          labels.append((1,1))
        

      annotations = []

      if img_path in dog_bounding_boxes:
        annotations.append(dog_bounding_boxes[img_path])
      
      if img_path in cat_bounding_boxes:
        annotations.append(cat_bounding_boxes[img_path])

      im = Image.open(self.x_path+'/'+img_path).convert('RGB')
      im.draft('RGB',(INPUT_SIZE,INPUT_SIZE))
      im_np = np.asarray(im)#.transpose((2,0,1))
      im = self.transform(im_np)

      self.imgs.append(im)
      self.boxes.append(annotations)


      # im_cv = cv2.imread(self.x_path+'/'+img_path,flags=cv2.IMREAD_COLOR)
      
      # im = np.transpose(im_cv,(2,0,1)).astype(np.float32)
      # im = Image.open(self.x_path+'/'+img_path)
      # im.draft('RGB',(size,size))
      # print(i)
      # imgs.append(np.asarray(im).astype(np.float32))
    
    # self.X = transform(np.stack(imgs,axis=0))
    self.y = np.asarray(labels)
  
  def __getitem__(self, index):
    # im = Image.open(self.x_path+'/'+self.x_file_paths[index])
    # # im.draft('RGB',(self.size,self.size))
    # im.draft('RGB',(INPUT_SIZE,INPUT_SIZE))
    # im_np = np.asarray(im)#.transpose((2,0,1))
    # im = self.transform(im_np)
    # # print(im.shape)
    # return (im,self.y[index])
    if self.split == "test":
      return self.imgs[index],"No label"

    return self.imgs[index],self.y[index], json.dumps(self.boxes[index])

  def __len__(self):
    return len(self.x_file_paths)


class cityscape(Dataset):
  def __init__(self,root,split,transform):
    self.x_path = root+'/leftImg8bit/'+ split
    self.transform = transform

    self.x_file_paths = list(Path(self.x_path).rglob("*.png"))
    

    # self.x_file_paths = glob.glob(self.x_path + '/*.png', recursive=True)

    self.imgs = []
    
    for i,img_path in enumerate(self.x_file_paths):
      im = Image.open(img_path).convert('RGB')
      im.draft('RGB',(INPUT_SIZE,INPUT_SIZE))
      im_np = np.asarray(im)#.transpose((2,0,1))
      im = self.transform(im_np)

      self.imgs.append(im)

  def __getitem__(self,index):
    # im = Image.open(self.x_file_paths[index])
    # im.draft('RGB',(INPUT_SIZE,INPUT_SIZE))
    # im_np = np.asarray(im)
    # im = self.transform(im_np)
    # return (im,"No label")
    return (self.imgs[index],"No label")

  def __len__(self):
    return len(self.x_file_paths)


class lfw(Dataset):
  def __init__(self,root,split,transform,paths=None):
    self.x_path = root
    self.transform = transform

    if paths is None:
      self.x_file_paths = glob.glob(self.x_path + '/**/*.jpg', recursive=True)
    else:
      self.x_file_paths = paths

    
    if split == "train":
      self.start = 0
      self.end = int (len(self.x_file_paths) * 0.6)
    elif split == "val":
      self.start = int(len(self.x_file_paths) * 0.6)
      self.end = int(len(self.x_file_paths) * 0.8)
    else:
      self.start = int(len(self.x_file_paths) * 0.8)
      self.end = len(self.x_file_paths)
      
    self.split_path = self.x_file_paths[self.start:self.end]

    self.imgs = []
    
    for i,img_path in enumerate(self.split_path):
      im = Image.open(img_path).convert('RGB')
      im.draft('RGB',(INPUT_SIZE,INPUT_SIZE))
      im_np = np.asarray(im)#.transpose((2,0,1))
      im = self.transform(im_np)

      self.imgs.append(im)
    

  def __getitem__(self,index):
    # im = Image.open(self.split_path[index])
    # im.draft('RGB',(INPUT_SIZE,INPUT_SIZE))
    # label = os.path.basename(os.path.dirname(self.split_path[index]))
    # im_np = np.asarray(im)
    # im = self.transform(im_np)
    # return (im,label)
    # return self.X[index],self.y[index]
    return (self.imgs[index],"No label")

  def __len__(self):
    return len(self.split_path)

