import cv2
import numpy
import torch.utils.data
import os

class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_root='', transform=None):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        self.file_list = open(file_root + '/' + dataset + '/list/' + dataset + '.txt').read().splitlines()
        self.pre_images = [file_root + '/' + dataset + '/A/' + x for x in self.file_list]
        self.post_images = [file_root + '/' + dataset + '/B/' + x for x in self.file_list]
        self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        #self.file_root=file_root
        #self.dataset=dataset
        #self.file_list = os.listdir(file_root+ '/' + dataset +"/A/")#the name of training data in temporal A
        self.transform = transform
        #if (dataset=="train"):
        #    self.title="train"
        #else:
        #   self.title="test"

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        #base_name=self.file_list[idx].split(self.title+"1_")[1].split(".tif")[0]
        #print(base_name)
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]
        #pre_image = cv2.imread(self.file_root+ "/" + self.dataset +"/A/"+self.file_list[idx],-1)
        #label = cv2.imread(self.file_root+ "/" + self.dataset +"/label/label_"+base_name+".tif", 0)
        #post_image = cv2.imread(self.file_root+ "/" + self.dataset +"/B/"+self.title+"2_"+base_name+".tif",-1)
        pre_image = cv2.imread(pre_image_name)
        label = cv2.imread(label_name, 0)
        post_image = cv2.imread(post_image_name)
        #print(pre_image)
        #print("shape",pre_image.shape)
        #print(label)
        img = numpy.concatenate((pre_image, post_image), axis=2)
        # if self.transform:
        #     [pre_image, label, post_image] = self.transform(pre_image, label, post_image)
        #
        # return pre_image, label, post_image
        if self.transform:
            [img, label] = self.transform(img, label)

        return img, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}

class DatasetSentinel(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, adaptation, file_root='', transform=None):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        #self.file_list = open(file_root + '/' + dataset + '/list/' + dataset + '.txt').read().splitlines()
        #self.pre_images = [file_root + '/' + dataset + '/A/' + x for x in self.file_list]
        #self.post_images = [file_root + '/' + dataset + '/B/' + x for x in self.file_list]
        #self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        self.file_root=file_root
        print(self.file_root)
        self.dataset=dataset
        self.adaptation=adaptation
        self.file_list = os.listdir(file_root+ '/' + dataset +"/A/"+self.adaptation+"/")#the name of training data in temporal A
        #print(self.file_list)
        self.transform = transform
        if (dataset=="train"):
            self.title="train"
        else:
            self.title="test"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #if (self.title=="train"):
        base_name=self.file_list[idx].split(self.title+"1_")[1].split(".tif")[0]
        #print(base_name)
        #pre_image_name = self.pre_images[idx]
        #label_name = self.gts[idx]
        #post_image_name = self.post_images[idx]
        pre_image = cv2.imread(self.file_root+ "/" + self.dataset +"/A/"+self.adaptation+"/"+self.file_list[idx],-1)
        label = cv2.imread(self.file_root+ "/" + self.dataset +"/label/label_"+base_name+".tif", 0)
        post_image = cv2.imread(self.file_root+ "/" + self.dataset +"/B/"+self.title+"2_"+base_name+".tif",-1)
        # elif (self.title=="test"):
        #     base_name=self.file_list[idx].split(self.title+"1_")[1].split(".png")[0]
        #     pre_image = cv2.imread(self.file_root+ "/" + self.dataset +"/A/"+self.adaptation+"/"+self.file_list[idx],-1)
        #     label = cv2.imread(self.file_root+ "/" + self.dataset +"/label/label_"+base_name+".png", 0)
        #     post_image = cv2.imread(self.file_root+ "/" + self.dataset +"/B/"+self.title+"2_"+base_name+".png",-1)
        #print(pre_image)
        #print("shape",pre_image.shape)
        #print(label)
        img = numpy.concatenate((pre_image, post_image), axis=2)
        # if self.transform:
        #     [pre_image, label, post_image] = self.transform(pre_image, label, post_image)
        #
        # return pre_image, label, post_image
        if self.transform:
            [img, label] = self.transform(img, label)

        return img, label, base_name

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
