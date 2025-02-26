from lib2to3.pgen2.token import AMPER
from data.base_dataset import BaseDataset, get_transform, get_transformp, get_transformp1
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import os
import torch
from osgeo import gdal

class SingledoubleGDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.task=opt.taskname
        self.gpu_ids = opt.gpu_ids
        self.device=torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        
        self.cpXlist=[]
        maxindex=self.load_size-self.crop_size#
        minindex=0
        self.cpYlist=[]
        for _ in range(0,10000):
            self.cpYlist.append(random.uniform(minindex,maxindex))
        for _ in range(0,10000):
            self.cpXlist.append(random.uniform(minindex,maxindex))
        self.transform = get_transform(opt, grayscale=(input_nc == 1))
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        
        achoicex=random.randint(0,9999)
        achoicey=random.randint(0,9999)
        A_path = self.A_paths[index]
        print(A_path)

        if self.task=="GESD":
            Am_path1=os.path.split(A_path)[1].split('.png')[0].split('A_')[1]
            m_folder=os.path.split(os.path.split(A_path)[0])[0]
            Am_path2=m_folder+"/Amask/A_mask_"+Am_path1+".png"
        elif self.task=="SHCD":
            phase=os.path.split(A_path)[1].split('1_')[0]
            Am_path1=os.path.split(A_path)[1].split('.tif')[0].split(phase+'1_')[1]
            m_folder=os.path.split(os.path.split(A_path)[0])[0]
            Am_path2=m_folder+"/"+phase+"Amask/"+phase+"1_"+Am_path1+"_label.tif"

        A_imggdal=gdal.Open(A_path)
        A_imgarr=A_imggdal.ReadAsArray()

        if self.task=="GESD":
            A_nparr=np.rollaxis(A_imgarr,0,3)/255
        elif self.task=="SHCD":
            A_nparr=np.rollaxis(A_imgarr,0,3)/5000
        #A_img = Image.open(A_path).convert('RGB')
        A_mask = Image.open(Am_path2).convert('RGB')
        #self.transform0=get_transformp(self.opt,crop_pos=[self.cpXlist[achoicex],self.cpYlist[achoicey]], grayscale=(False),convert=True)
        self.transform1 = get_transformp1(self.opt,crop_pos=[self.cpXlist[achoicex],self.cpYlist[achoicey]], grayscale=(False),convert=True)
        A=self.transform1(np.float32(A_nparr))##
        self.transform_m0 = get_transformp(self.opt,crop_pos=[self.cpXlist[achoicex],self.cpYlist[achoicey]], grayscale=(False),convert=False)
        Am= np.array(self.transform_m0(A_mask))
        Amp=np.transpose(Am,[2,0,1])
        return {'A': A, 'A_mask':Amp,'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)