from .base_model import BaseModel
from . import networks
from data.base_dataset import RGB2HSInormbatch,HSI2RGBnormbatch
import torch
import numpy as np
from models.WFusion import Weightfusion

class TestdoubleG8bitModel(BaseModel):
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
        parser.set_defaults(dataset_mode='singledoubleG8bit')#这里定义了dataset_mode
        parser.add_argument('--model_suffix', type=str, default='_A', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        #self.model_names = ['G' + opt.model_suffix,'G' + opt.model_suffix+'C']
        self.model_names = ['G' + opt.model_suffix,'G' + opt.model_suffix+'C','G_att' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netGC = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,#强行定义一个值
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_att=networks.define_Att(opt.input_nc, opt.output_nc,opt.ngf,opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.Fusion=Weightfusion(radius=3)
        # print(opt.netG)
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.
        setattr(self, 'netG' + opt.model_suffix+'C', self.netGC)  # store netG in self.
        setattr(self, 'netG_att' + opt.model_suffix, self.netG_att)  # store netG_att in self.
    def load_seginv(self,content_seg):
        #color_codes = ['RED','WHITE','GREEN','BLUE','YELLOW','BLACK','DGREEN','BROWN','PINK']
        #color_codes = ['RED','BLUE','YELLOW','WHITE','BLACK']#前五个吧为不变地物#肯定也是要提取前五个的分别的模板
        color_codes = ['RED','BLUE']
        #color_codes = ['RED']
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
            return torch.mul(torch.mul(mask_r, mask_g), mask_b)
        cmaski=[]
        smaski=[]#对应位置相加得到总的onehot
        for i in range(len(color_codes)):
            cmaski.append(torch.unsqueeze(_extract_mask(content_seg, color_codes[i]), 1).float())
            #smaski.append(torch.unsqueeze(_extract_mask(style_seg, color_codes[i]), 1).float())
        #b,c,h,w,
        cmaski_onehot=torch.zeros_like(cmaski[0])
        for i in range(len(color_codes)):
            cmaski_onehot+=cmaski[i]
        #print(cmaski_onehot)
        return cmaski,cmaski_onehot#,smaski#就怕有黑色部分影响结果

    def load_segvar(self,content_seg):
        #color_codes = ['RED','WHITE','GREEN','BLUE','YELLOW','BLACK','DGREEN','BROWN','PINK']
        #color_codes = ['PINK','BROWN','DGREEN','GREEN']#前五个吧为不变地物#肯定也是要提取前五个的分别的模板
        color_codes = ['DGREEN','GREEN']
        def _extract_mask(seg, color_str):
            b, c, h, w = np.shape(seg)
            if color_str == "GREEN":
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = (seg[:,1,:, :] > 240)
                mask_b = (seg[:,2,:, :] < 15)
            elif color_str == "DGREEN":#139度灰 树
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = torch.mul((seg[:,1,:, :] > 130),
                                     (seg[:,1,:, :] < 145))
                mask_b = (seg[:,2,:, :] < 15)
            elif color_str == "BROWN":#空地
                mask_r = torch.mul((seg[:,0,:, :] > 130),
                                     (seg[:,0,:, :] < 145))
                mask_g = torch.mul((seg[:,1,:, :] > 80),
                                     (seg[:,1,:, :] < 100))
                mask_b = (seg[:,2,:, :] <15)
            elif color_str == "PINK":#空地
                mask_r =  (seg[:,0,:, :] > 240)
                mask_g = (seg[:,1,:, :] < 15)
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
            return torch.mul(torch.mul(mask_r, mask_g), mask_b)
        cmaskv=[]
        smaskv=[]#对应位置相加得到总的onehot
        for i in range(len(color_codes)):
            cmaskv.append(torch.unsqueeze(_extract_mask(content_seg, color_codes[i]), 1).float())
        #b,c,h,w,
        cmaskv_onehot=torch.zeros_like(cmaskv[0])
        for i in range(len(color_codes)):
            #print("part",cmaskv[i])
            cmaskv_onehot+=cmaskv[i]
        return cmaskv,cmaskv_onehot#,smaskv#就怕有黑色部分影响结果
    def load_segcity(self,img):
        color_codes = ['RED','WHITE','GREEN','BLUE','YELLOW','BLACK','DGREEN','BROWN','PINK']
        def _extract_mask(seg, color_str):
            b, c, h, w = np.shape(seg)
            if color_str == "RED":
                mask_r = (seg[:,0,:, :] > 240)
                mask_g = (seg[:,1,:, :] < 15)
                mask_b = (seg[:,2,:, :] < 15)
            return torch.mul(torch.mul(mask_r, mask_g), mask_b)
        cmask=torch.unsqueeze(_extract_mask(img, color_codes[0]), 1).float()
        return cmask
    def Net_Gpart(self,img,mask,netG,netGp):#所以你说这个netG和netGC有什么不一样#前一个是变化的，后一个是不变的
        #pred=torch.zeros_like(img)
        _,invmask=self.load_seginv(mask)#new
        #imaskdilate= torch.nn.functional.max_pool2d(invmask, kernel_size=7, stride=1, padding=3)#dilate rate can not too big
        imaskdilate= torch.nn.functional.max_pool2d(invmask, kernel_size=5, stride=1, padding=2)#dilate rate can not too big
        RGBinv=torch.mul(img,imaskdilate)#new
        maskinv=torch.mul(mask,imaskdilate)#new
        #dst1=netGp(torch.cat([img,mask],1))
        dst1=netGp(torch.cat([RGBinv,maskinv],1))
        _,vmask=self.load_segvar(mask)
        #vmaskdilate= torch.nn.functional.max_pool2d(vmask, kernel_size=7, stride=1, padding=3)#avgpooling会不会好一些,这个padding 不宜太大，会造成干扰
        vmaskdilate= torch.nn.functional.max_pool2d(vmask, kernel_size=5, stride=1, padding=2)#avgpooling会不会好一些,这个padding 不宜太大，会造成干扰
        RGBv=torch.mul(img,vmaskdilate)#动的部分，也即要用gram约束的部分,就关键在于给他黑一下，然后,黑的部分是否影响
        maskv=torch.mul(mask,vmaskdilate)
        #RGBpredvar=netG(torch.cat([img,mask],1))
        RGBpredvar=netG(torch.cat([RGBv,maskv],1))#
        RGBfusion=self.Fusion(dst1,RGBpredvar,vmask)
        return RGBfusion

    def Net_Gpart1(self,img,mask,netG,netGp):#所以你说这个netG和netGC有什么不一样#前一个是变化的，后一个是不变的
        #pred=torch.zeros_like(img)
        _,invmask=self.load_seginv(mask)#new
        imaskdilate= torch.nn.functional.max_pool2d(invmask, kernel_size=5, stride=1, padding=2)#dilate rate can not too big
        RGBinv=torch.mul(img,imaskdilate)#new
        maskinv=torch.mul(mask,imaskdilate)#new
        dst1=netGp(torch.cat([RGBinv,maskinv],1))
        _,vmask=self.load_segvar(mask)
        vmaskdilate= torch.nn.functional.max_pool2d(vmask, kernel_size=5, stride=1, padding=2)#avgpooling会不会好一些,这个padding 不宜太大，会造成干扰
        
        RGBv=torch.mul(img,vmaskdilate)#动的部分，也即要用gram约束的部分,就关键在于给他黑一下，然后,黑的部分是否影响
        maskv=torch.mul(mask,vmaskdilate)
        #RGBpredvar=netG(torch.cat([RGBv,maskv],1))
        RGBpredvar=netG(torch.cat([img,mask],1))
        RGBfusion=self.Fusion(dst1,RGBpredvar,vmask)
        return RGBfusion
    def Net_GC(self,img,mask,netG,netGC):#所以你说这个netG和netGC有什么不一样
        pred=torch.zeros_like(img)
        #print(img.shape)
        #print(mask.shape)
        dst1=netG(torch.cat([img,mask],1))
        #dst1=netG(img)
        #dst1=torch.ones_like(img)
        #onehot化第一个city部分
        cmask=self.load_segcity(mask)
        RGBcity=torch.mul(img,cmask)
        #去标准化及归一化
        #Hsicity=RGB2HSInormbatch(RGBcity)#以除以5000作为归一化，5000大了么，这里也要对应变化，不然不可能预测正确
        #Hsipredcity=netGC(torch.cat([Hsicity,img],1))#加到img效果可能不好
        #RGBpredcity=HSI2RGBnormbatch(Hsipredcity)#归一化的RGB
        RGBpredcity=netGC(torch.cat([RGBcity,img],1))
        #RGBpredcity=netGC(RGBcity)
        #替换
        inds=(cmask!=0)
        indsinv=(cmask==0)
        inds3=torch.cat([inds,inds,inds],axis=1)
        indsinv3=torch.cat([indsinv,indsinv,indsinv],axis=1)
        pred[inds3]=RGBpredcity[inds3]
        pred[indsinv3]=dst1[indsinv3]
        return pred#直接改return就好了
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.mask=input['A_mask'].to(self.device)
        #print(self.real.shape)#我们想要400×400
        #print(self.real)
        #print(self.mask)#我们想要400×400
        #print(self.mask.shape)
        self.sum=torch.cat([self.real,self.mask],1)
        
        # print(self.real)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        #self.fake = self.Net_GC(self.real,self.mask,self.netG,self.netGC)  # G(real)
        #self.fake = self.Net_Gpart(self.real,self.mask,self.netG,self.netGC)  # G(real)
        #self.fake = 1.2*self.netG_att(self.real)*self.Net_Gpart(self.real,self.mask,self.netG,self.netGC) + (1-1.2*self.netG_att(self.real))*self.real
        self.fake = (0.6+0.6*self.netG_att(self.real))*self.Net_Gpart(self.real,self.mask,self.netG,self.netGC) + (0.4-0.6*self.netG_att(self.real))*self.real
        #print(self.fake.shape)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass