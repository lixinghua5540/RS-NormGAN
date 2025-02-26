import os
from data.base_dataset import BaseDataset, get_transformp, get_transformp1
from data.image_folder import make_dataset
from PIL import Image
from osgeo import gdal
import random
import numpy as np

class UnalignedDataset(BaseDataset):#unpaired
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.task=opt.taskname
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'delete
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'delete
        self.dir_Amask=os.path.join(opt.dataroot, opt.phase + 'Amask')
        self.dir_Bmask=os.path.join(opt.dataroot, opt.phase + 'Bmask')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.Amask_paths = sorted(make_dataset(self.dir_Amask, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.Bmask_paths = sorted(make_dataset(self.dir_Bmask, opt.max_dataset_size)) 
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        maxindex=self.load_size-self.crop_size
        minindex=0
        #create adequate number to confirm crop area
        self.cpXlist=[]
        for _ in range(0,10000):
            self.cpXlist.append(random.uniform(minindex,maxindex))
        self.cpYlist=[]
        for _ in range(0,10000):
            self.cpYlist.append(random.uniform(minindex,maxindex))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        if self.task=="GESD":
            imgA_name = os.path.split(A_path)[1].split('.png')[0].split('A_')[1]
            imgB_name = os.path.split(B_path)[1].split('.png')[0].split('B_')[1]
            imga_name=self.root+"/trainAmask/A_mask_"+imgA_name+".png"
            imgb_name=self.root+"/trainBmask/B_mask_"+imgB_name+".png"
        elif self.task=="SHCD":
            imgA_name = os.path.split(A_path)[1].split('.tif')[0]
            imgB_name = os.path.split(B_path)[1].split('.tif')[0]
    
            imga_name=self.root+"/trainAmask/"+imgA_name+"_mask.png"
            imgb_name=self.root+"/trainBmask/"+imgB_name+"_mask.png"

        A_imggdal=gdal.Open(A_path)
        A_imgarr=A_imggdal.ReadAsArray()
        B_imggdal=gdal.Open(B_path)
        B_imgarr=B_imggdal.ReadAsArray()

        if self.task=="GESD":
            A_nparr=np.rollaxis(A_imgarr,0,3)/255
            B_nparr=np.rollaxis(B_imgarr,0,3)/255
        elif self.task=="SHCD":
            A_nparr=np.rollaxis(A_imgarr,0,3)/5000
            B_nparr=np.rollaxis(B_imgarr,0,3)/5000

        #A_img = Image.open(A_path).convert('RGB')
        #B_img = Image.open(B_path).convert('RGB')
        A_mask = Image.open(imga_name).convert('RGB')
        B_mask = Image.open(imgb_name).convert('RGB')
        achoicex=random.randint(0,9999)#the left upper corner
        achoicey=random.randint(0,9999)
        bchoicex=random.randint(0,9999)
        bchoicey=random.randint(0,9999)
        # apply image transformation
        self.transform_A1 = get_transformp1(self.opt,crop_pos=[self.cpXlist[achoicex],self.cpYlist[achoicey]], grayscale=(False),convert=True)
        self.transform_B1 = get_transformp1(self.opt,crop_pos=[self.cpXlist[bchoicex],self.cpYlist[bchoicey]], grayscale=(False),convert=True)
        #self.transform_A2 = get_transformp2(self.opt,crop_pos=[self.cpXlist[achoicex],self.cpYlist[achoicey]], grayscale=(False),convert=False)
        #self.transform_B2 = get_transformp2(self.opt,crop_pos=[self.cpXlist[bchoicex],self.cpYlist[bchoicey]], grayscale=(False),convert=False)
        self.transform_Am = get_transformp(self.opt,crop_pos=[self.cpXlist[achoicex],self.cpYlist[achoicey]], grayscale=(False),convert=False)
        self.transform_Bm = get_transformp(self.opt,crop_pos=[self.cpXlist[bchoicex],self.cpYlist[bchoicey]], grayscale=(False),convert=False)
        #print(np.array(A_img))
        #A = self.transform_A2(A_img)
        #print(np.array(A))
        A=self.transform_A1(np.float32(A_nparr))
        B = self.transform_B1(np.float32(B_nparr))
        Am = np.array(self.transform_Am(A_mask))
        Bm = np.array(self.transform_Bm(B_mask))
        Amp=np.transpose(Am,[2,0,1])
        Bmp=np.transpose(Bm,[2,0,1])
        return {'A': A, 'B': B, 'A_mask': Amp, 'B_mask': Bmp,'A_paths': A_path, 'B_paths': B_path,'Am_paths': imga_name, 'Bm_paths': imgb_name}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
