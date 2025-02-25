from .base_model import BaseModel
from . import networks
import torch
import numpy as np
from models.WFusion import Weightfusion
import torch.nn as nn

class TestdoubleGModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='singledoubleG')
        parser.add_argument('--model_suffix', type=str, default='_A', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.task=opt.taskname
        if self.task=="GESD":
            self.subtask=opt.subtask
        if self.task=="SHCD":
            self.c_code = ['RED','BLUE','YELLOW','WHITE','BLACK','PINK','BROWN','DGREEN','GREEN']
            self.c_code_var = ['PINK','BROWN','DGREEN','GREEN']
            self.c_code_inv = ['RED','BLUE','YELLOW','WHITE','BLACK']
            self.AttWeight = 1.2
            self.AttBias = 0.0
        elif self.task=="GESD":
            self.c_code = ['RED','BLUE','GREEN','DGREEN']
            self.AttWeight = 0.6
            self.AttBias = 0.6

            if self.subtask=="S2A":
                self.c_code_var = ['GREEN','DGREEN']
                self.c_code_inv = ['RED','BLUE']
            elif self.subtask=="S2W":
                self.c_code_var = ['DGREEN']
                self.c_code_inv = ['RED','BLUE','GREEN']
        
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix+'v','G' + opt.model_suffix,'G_GAM' + opt.model_suffix]  # only generator is needed.
        #self.model_names = ['G' + opt.model_suffix]
        #self.model_names = ['G' + opt.model_suffix+'v','G' + opt.model_suffix]
        self.netGv = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,#6
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_GAM=networks.define_Att(opt.input_nc, opt.output_nc,opt.ngf,opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # print(opt.netG)
        self.Fusion=Weightfusion(radius=3)
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix+'v', self.netGv)  # store netG in self.
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG_C in self.
        setattr(self, 'netG_GAM' + opt.model_suffix, self.netG_GAM)  # store netG_att in self.

    def load_seginv(self,content_seg,c_code):
        #color_codes = ['RED','WHITE','GREEN','BLUE','YELLOW','BLACK','DGREEN','BROWN','PINK']
        #color_codes = ['RED','BLUE','YELLOW','WHITE','BLACK']
        #color_codes = ['RED',"BLUE"]
        color_codes = c_code
        def _extract_mask(seg, color_str):
            b, c, h, w = np.shape(seg)
            if color_str == "RED":
                mask_r = (seg[:,0,:, :] > 240)
                mask_g = (seg[:,1,:, :] < 15)
                mask_b = (seg[:,2,:, :] < 15)
            elif color_str == "WHITE":
                mask_r = (seg[:,0,:, :] > 240)
                mask_g = (seg[:,1,:, :] > 240)
                mask_b = (seg[:,2,:, :] > 240)
            elif color_str == "BLUE":
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = (seg[:,1,:, :] < 15)
                mask_b = (seg[:,2,:, :] > 240)
            elif color_str == "BLACK":
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = (seg[:,1,:, :] < 15)
                mask_b = (seg[:,2,:, :] < 15)
            elif color_str == "YELLOW":
                mask_r = (seg[:,0,:, :] > 240)
                mask_g = (seg[:,1,:, :] > 240)
                mask_b = (seg[:,2,:, :]  < 15)
            elif color_str == "GREEN":
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = (seg[:,1,:, :] > 240)
                mask_b = (seg[:,2,:, :] < 15)
            return torch.mul(torch.mul(mask_r, mask_g), mask_b)
        cmaski=[]
        smaski=[]
        for i in range(len(color_codes)):
            cmaski.append(torch.unsqueeze(_extract_mask(content_seg, color_codes[i]), 1).float())
            #smaski.append(torch.unsqueeze(_extract_mask(style_seg, color_codes[i]), 1).float())
        #b,c,h,w,
        cmaski_onehot=torch.zeros_like(cmaski[0])
        for i in range(len(color_codes)):
            cmaski_onehot+=cmaski[i]
        #print(cmaski_onehot)
        return cmaski,cmaski_onehot#,smaski

    def load_segvar(self,content_seg,c_code):
        #color_codes = ['RED','WHITE','GREEN','BLUE','YELLOW','BLACK','DGREEN','BROWN','PINK']
        #color_codes = ['PINK','BROWN','DGREEN','GREEN']
        #color_codes = ['DGREEN','GREEN']
        color_codes = c_code
        def _extract_mask(seg, color_str):
            b, c, h, w = np.shape(seg)
            if color_str == "GREEN":
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = (seg[:,1,:, :] > 240)
                mask_b = (seg[:,2,:, :] < 15)
            elif color_str == "DGREEN":
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = torch.mul((seg[:,1,:, :] > 130),
                                     (seg[:,1,:, :] < 145))
                mask_b = (seg[:,2,:, :] < 15)
            elif color_str == "BROWN":
                mask_r = torch.mul((seg[:,0,:, :] > 130),
                                     (seg[:,0,:, :] < 145))
                mask_g = torch.mul((seg[:,1,:, :] > 80),
                                     (seg[:,1,:, :] < 100))
                mask_b = (seg[:,2,:, :] <15)
            elif color_str == "PINK":
                mask_r =  (seg[:,0,:, :] > 240)
                mask_g = (seg[:,1,:, :] < 15)
                mask_b = (seg[:,2,:, :] > 240)
            # elif color_str == "BLUE":
            #     mask_r = (seg[:,0,:, :] < 15)
            #     mask_g = (seg[:,1,:, :] < 15)
            #     mask_b = (seg[:,2,:, :] > 240)
            # elif color_str == "BLACK":
            #     mask_r = (seg[:,0,:, :] < 15)
            #     mask_g = (seg[:,1,:, :] < 15)
            #     mask_b = (seg[:,2,:, :] < 15)
            # elif color_str == "YELLOW":
            #     mask_r = (seg[:,0,:, :] > 240)
            #     mask_g = (seg[:,1,:, :] > 240)
            #     mask_b = (seg[:,2,:, :]  < 15)
            return torch.mul(torch.mul(mask_r, mask_g), mask_b)
        cmaskv=[]
        smaskv=[]
        for i in range(len(color_codes)):
            cmaskv.append(torch.unsqueeze(_extract_mask(content_seg, color_codes[i]), 1).float())
        #b,c,h,w,
        cmaskv_onehot=torch.zeros_like(cmaskv[0])
        for i in range(len(color_codes)):
            #print("part",cmaskv[i])
            cmaskv_onehot+=cmaskv[i]
        return cmaskv,cmaskv_onehot#,smaskv

    def Net_Gpart(self,img,mask,netG,netGp):
        _,invmask=self.load_seginv(mask)#new
        imaskdilate= torch.nn.functional.max_pool2d(invmask, kernel_size=5, stride=1, padding=2)#dilate rate can not too big
        RGBinv=torch.mul(img,imaskdilate)#new
        maskinv=torch.mul(mask,imaskdilate)#new
        dst1=netGp(torch.cat([RGBinv,maskinv],1))
        _,vmask=self.load_segvar(mask)
        vmaskdilate= torch.nn.functional.max_pool2d(vmask, kernel_size=5, stride=1, padding=2)
        RGBv=torch.mul(img,vmaskdilate)
        maskv=torch.mul(mask,vmaskdilate)
        RGBpredvar=netG(torch.cat([RGBv,maskv],1))
        RGBfusion=self.Fusion(dst1,RGBpredvar,vmask)
        return RGBfusion
        #return RGBfusion

    # def Net_Gpart1(self,img,mask,netG,netGp):#所以你说这个netG和netGC有什么不一样#前一个是变化的，后一个是不变的
    #     _,invmask=self.load_seginv(mask)#new
    #     imaskdilate= torch.nn.functional.max_pool2d(invmask, kernel_size=5, stride=1, padding=2)#dilate rate can not too big
    #     RGBinv=torch.mul(img,imaskdilate)#new
    #     maskinv=torch.mul(mask,imaskdilate)#new
    #     dst1=netGp(torch.cat([RGBinv,maskinv],1))
    #     _,vmask=self.load_segvar(mask)
    #     vmaskdilate= torch.nn.functional.max_pool2d(vmask, kernel_size=5, stride=1, padding=2)#avgpooling会不会好一些,这个padding 不宜太大，会造成干扰
        
    #     RGBv=torch.mul(img,vmaskdilate)#动的部分，也即要用gram约束的部分,就关键在于给他黑一下，然后,黑的部分是否影响
    #     maskv=torch.mul(mask,vmaskdilate)
    #     #RGBpredvar=netG(torch.cat([RGBv,maskv],1))
    #     RGBpredvar=netG(torch.cat([img,mask],1))#区别在这里，不用那个分开的变动部分
    #     RGBfusion=self.Fusion(dst1,RGBpredvar,vmask)
    #     return RGBfusion
    def Net_Gpart1(self,img,mask,netGv,netG):
        #pred=torch.zeros_like(img)
        _,invmask=self.load_seginv(mask)#new
        imaskdilate= torch.nn.functional.max_pool2d(invmask, kernel_size=5, stride=1, padding=2)
        RGBinv=torch.mul(img,imaskdilate)#new
        maskinv=torch.mul(mask,imaskdilate)#new
        #dst1=netGp(torch.cat([img,mask],1))
        dst1=netG(RGBinv)
        _,vmask=self.load_segvar(mask)
        vmaskdilate= torch.nn.functional.max_pool2d(vmask, kernel_size=5, stride=1, padding=2)
        RGBv=torch.mul(img,vmaskdilate)
        maskv=torch.mul(mask,vmaskdilate)
        RGBpredvar=netGv(RGBv)
        #RGBpredvar=netGv(torch.cat([img,mask],1))
        RGBfusion=self.Fusion(dst1,RGBpredvar,vmask)
        return RGBfusion
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.mask=input['A_mask'].to(self.device)
        self.sum=torch.cat([self.real,self.mask],1)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        #self.fake = self.Net_Gpart(self.real,self.mask,self.netGv,self.netG)  # G(real)
        self.fake=(self.AttWeight*self.netG_GAM(self.real)+self.AttBias)*self.Net_Gpart(self.real,self.mask,self.netGv,self.netG)+(1-self.AttBias-self.AttWeight*self.netG_GAM(self.real))*self.real
        #self.fake = self.netG(torch.cat([self.real,self.mask],1))
        #self.fake = self.netG_att(self.real)*self.Net_Gpart(self.real,self.mask,self.netG,self.netGC) + (1-self.netG_att(self.real))*self.real
        #self.fake = (1-self.netG_att(self.real)/2)*self.Net_Gpart(self.real,self.mask,self.netG,self.netGC) + (self.netG_att(self.real)/2)*self.real
        #self.fake = self.netG_att(torch.cat([self.real,self.mask],1))*self.Net_Gpart(self.real,self.mask,self.netG,self.netGC) + (1-self.netG_att(torch.cat([self.real,self.mask],1)))*self.real

    def optimize_parameters(self):
        """No optimization for test model."""
        pass