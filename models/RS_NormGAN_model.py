import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn.functional as F
from .vgg import Vgg19
from models.WFusion import Weightfusion
from torch.optim import lr_scheduler
from .ssim import MSSSIM

class RSNormGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.task=opt.taskname
        if self.task=="GESD":
            self.subtask=opt.subtask
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #self.loss_names = ['G_AB', 'cycle_A', 'cycle_Av', 'idt_A', 'cycle_B','cycle_Bv', 'idt_B','D_A','D_B','D_Av','D_Bv','SSIM']
        self.loss_names = ['G_AB', 'cycle_A', 'idt_A', 'cycle_B', 'idt_B','D_A','D_B','SSIM']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        #visual_names_A = ['real_A', 'fake_B']
        #visual_names_B = ['real_B', 'fake_A']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            #self.model_names = ['G_A', 'G_B','G_AC', 'G_BC','G_att_A','G_att_B', 'D_A','D_B']#global-local attention module utlizes at the very first time
            #self.model_names = ['G_Av', 'G_Bv','G_A', 'G_B', 'D_A','D_B']
            self.model_names = ['G_A', 'G_B','G_Av', 'G_Bv', 'D_A','D_B', 'D_Av','D_Bv']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        #define spedific options for different tasks
        if self.task=="SHCD":
            self.RGBList = {
               'city':[255,0,0],#city
               'cloud':[255,255,255],#cloud
               'field':[0,255,0],#field
               'water':[0,0,255],#water
               'ground':[255,255,0],#field2
               'shadow':[0,0,0],#field2
              'tree':[0,139,0],#Tree
               'mountain':[139,90,0],#mountain
               'paddy':[255,0,255]}#paddy
            self.cW = 0
            self.CodetoWeight={
               'RED':0.05,
               'WHITE':0.5,
               'GREEN':0.3,
               'BLUE':0.5,
               'BLACK':0.5e-2,
               'YELLOW':0.3,
               'DGREEN':0.3,#0.3
               'BROWN':0.5e-1,
               'PINK':0.5e-1#
                }
            self.v_class = 5
            self.c_code = ['RED','BLUE','YELLOW','WHITE','BLACK','PINK','BROWN','DGREEN','GREEN']
            self.c_code_var = ['PINK','BROWN','DGREEN','GREEN']
            self.c_code_inv = ['RED','BLUE','YELLOW','WHITE','BLACK']
            self.AttWeight = 1.2
            self.AttBias = 0.0

        elif self.task=="GESD":
            self.RGBList = {
                'city':[255,0,0],#city 
                'field':[0,255,0],#field
                'water':[0,0,255],#water
                'tree':[0,139,0]#Tree
                }
            self.cW = 0.35
            self.CodetoWeight={
            'RED':0.065,
            'GREEN':0.4,
            'BLUE':0.065,
            'DGREEN':0.4#0.3
                }
            self.c_code = ['RED','BLUE','GREEN','DGREEN']
            self.AttWeight = 0.6
            self.AttBias = 0.6

            if self.subtask=="S2A":
                self.v_class = 2
                self.c_code_var = ['GREEN','DGREEN']
                self.c_code_inv = ['RED','BLUE']
            elif self.subtask=="S2W":
                self.v_class = 3
                self.c_code_var = ['DGREEN']
                self.c_code_inv = ['RED','BLUE','GREEN']
   
        # define networks (Generators and discriminators and other auxiliary structures)

        self.criterionSSIM=MSSSIM(window_size=11, size_average=True, channel=3)

        self.netG_Av = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_Bv = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_GAM_A=networks.define_Att(opt.input_nc, opt.output_nc,opt.ngf,opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)     
        self.netG_GAM_B=networks.define_Att(opt.input_nc, opt.output_nc,opt.ngf,opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) 

        self.Fusion=Weightfusion(radius=3)
        #logger = logging.getLogger()
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_Av = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_Bv = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.Dmaskloader=networks.DiscriminatorWeight(opt.ndf)
            self.vgg_extractor=Vgg19(vgg19_npy_path="./models/vgg19.npy",device=self.gpu_ids)
            self.vgg_extractor.load_dict()
        if self.isTrain:
            #if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                #assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionWeightGAN = networks.WeightGANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # self.styleCheck=networks.StyleGAN(opt).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_Av.parameters(),self.netG_Bv.parameters(),self.netG_A.parameters(),self.netG_B.parameters()),lr=opt.lr,betas=(opt.beta1, 0.999))#can set different lr for different networks
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),self.netD_B.parameters(),self.netD_Av.parameters(),self.netD_Bv.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def load_seg(self,content_seg, style_seg, c_code):
        color_codes=c_code
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
            
            elif color_str == "GREEN":
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = (seg[:,1,:, :] > 240)
                mask_b = (seg[:,2,:, :] < 15)
            elif color_str == "BLACK":
                mask_r = (seg[:,0,:, :] < 15)
                mask_g = (seg[:,1,:, :] < 15)
                mask_b = (seg[:,2,:, :] < 15)
            elif color_str == "YELLOW":
                mask_r = (seg[:,0,:, :] > 240)
                mask_g = (seg[:,1,:, :] > 240)
                mask_b = (seg[:,2,:, :]  < 15)
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
            return torch.mul(torch.mul(mask_r, mask_g), mask_b)

        color_content_masks = []
        color_style_masks = []
        weight_matrixc=torch.zeros([content_seg.shape[0],content_seg.shape[2],content_seg.shape[3]])
        weight_matrixs=torch.zeros([style_seg.shape[0],style_seg.shape[2],style_seg.shape[3]])
        for i in range(len(color_codes)):
            color_content_masks.append(torch.unsqueeze(_extract_mask(content_seg, color_codes[i]), 1).float())
            color_style_masks.append(torch.unsqueeze(_extract_mask(style_seg, color_codes[i]), 1).float())
            weight_matrixc[torch.all(color_content_masks[i]==True,axis=1)]=self.CodetoWeight[color_codes[i]]#c represent content 
            weight_matrixs[torch.all(color_style_masks[i]==True,axis=1)]=self.CodetoWeight[color_codes[i]]#s represent style 
        
        onehot_cmask=torch.cat(color_content_masks,dim=1)
        onehot_smask=torch.cat(color_style_masks,dim=1)
        return color_content_masks, color_style_masks, torch.unsqueeze(weight_matrixc,axis=1).to(self.device),  torch.unsqueeze(weight_matrixs,axis=1).to(self.device),onehot_cmask,onehot_smask

    def load_seginv(self,content_seg,c_code):
        color_codes=c_code
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
        color_codes=c_code
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
            return torch.mul(torch.mul(mask_r, mask_g), mask_b)
        cmaskv=[]
        smaskv=[]#calculated by addtion operator of corresponding area in onehot label
        for i in range(len(color_codes)):
            cmaskv.append(torch.unsqueeze(_extract_mask(content_seg, color_codes[i]), 1).float())
            #smaskv.append(torch.unsqueeze(_extract_mask(style_seg, color_codes[i]), 1).float())
        #b,c,h,w,
        cmaskv_onehot=torch.zeros_like(cmaskv[0])
        for i in range(len(color_codes)):
            #print("part",cmaskv[i])
            cmaskv_onehot+=cmaskv[i]
        #print(cmaskv_onehot)
        #print(cmaskv_onehot.shape)
        return cmaskv,cmaskv_onehot#,smaskv

    def Net_Gpart(self,img,mask,netGv,netG):#using mask as the supplement
        #pred=torch.zeros_like(img)
        _,invmask=self.load_seginv(mask,self.c_code_inv)#new
        imaskdilate= torch.nn.functional.max_pool2d(invmask, kernel_size=5, stride=1, padding=2)
        RGBinv=torch.mul(img,imaskdilate)#new
        maskinv=torch.mul(mask,imaskdilate)#new
        #dst1=netGp(torch.cat([img,mask],1))
        dst1=netG(torch.cat([RGBinv,maskinv],1))
        _,vmask=self.load_segvar(mask,self.c_code_var)
        vmaskdilate= torch.nn.functional.max_pool2d(vmask, kernel_size=5, stride=1, padding=2)
        RGBv=torch.mul(img,vmaskdilate)
        maskv=torch.mul(mask,vmaskdilate)
        RGBpredvar=netGv(torch.cat([RGBv,maskv],1))
        #RGBpredvar=netGv(torch.cat([img,mask],1))
        RGBfusion=self.Fusion(dst1,RGBpredvar,vmask)
        return RGBfusion
    
    def Net_Gpart1(self,img,mask,netGv,netG):#
        #pred=torch.zeros_like(img)
        _,invmask=self.load_seginv(mask,self.c_code_inv)#new
        imaskdilate= torch.nn.functional.max_pool2d(invmask, kernel_size=5, stride=1, padding=2)
        RGBinv=torch.mul(img,imaskdilate)#new
        maskinv=torch.mul(mask,imaskdilate)#new
        #dst1=netGp(torch.cat([img,mask],1))
        dst1=netG(RGBinv)
        _,vmask=self.load_segvar(mask,self.c_code_var)
        vmaskdilate= torch.nn.functional.max_pool2d(vmask, kernel_size=5, stride=1, padding=2)
        RGBv=torch.mul(img,vmaskdilate)
        maskv=torch.mul(mask,vmaskdilate)
        RGBpredvar=netGv(RGBv)
        #RGBpredvar=netGv(torch.cat([img,mask],1))
        RGBfusion=self.Fusion(dst1,RGBpredvar,vmask)
        return RGBfusion

    def gram_matrix(self,activations):#this gram matrix is not the enhanced one
        #print("a",activations.device)
        batchsize = activations.shape[0]
        num_channels = activations.shape[1]
        height = activations.shape[2]
        width = activations.shape[3]
        gram_matrix=activations
        gram_matrix = torch.reshape(gram_matrix, [batchsize,num_channels, width * height])
        gram_matrix_T=torch.transpose(gram_matrix,2,1)

        Gram=torch.empty(batchsize,num_channels,num_channels).to(self.device)
        #print('g',Gram.device)
        for i in range(batchsize):
            Gramn=torch.matmul(gram_matrix[i],gram_matrix_T[i])
            Gram[i]=Gramn
        return Gram


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.mask_A = input['A_mask' if AtoB else 'B'].to(self.device)
        self.mask_B = input['B_mask' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        #print(onehot_maskA.shape)
        #print(onehot_maskA)
        self.fake_B=self.Net_Gpart(self.real_A,self.mask_A,self.netG_Av,self.netG_A)#double sub-generators in a generative process
        self.rec_A=self.Net_Gpart(self.fake_B,self.mask_A,self.netG_Bv,self.netG_B)#
        self.fake_A=self.Net_Gpart(self.real_B,self.mask_B,self.netG_Bv,self.netG_B)#
        self.rec_B=self.Net_Gpart(self.fake_A,self.mask_B,self.netG_Av,self.netG_A)#

    def forward_att(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B=(1.2*self.netG_GAM_A(self.real_A))*self.Net_Gpart(self.real_A,self.mask_A,self.netG_Av,self.netG_A)+(1-1.2*self.netG_GAM_A(self.real_A))*self.real_A
        # self.rec_A=(1.2*self.netG_GAM_B(self.fake_B))*self.Net_Gpart(self.fake_B,self.mask_A,self.netG_Bv,self.netG_B)+(1-1.2*self.netG_GAM_B(self.fake_B))*self.fake_B
        # self.fake_A=(1.2*self.netG_GAM_B(self.real_B))*self.Net_Gpart(self.real_B,self.mask_B,self.netG_Bv,self.netG_B)+(1-1.2*self.netG_GAM_B(self.real_B))*self.real_B
        # self.rec_B=(1.2*self.netG_GAM_A(self.fake_A))*self.Net_Gpart(self.fake_A,self.mask_B,self.netG_Av,self.netG_A)+(1-1.2*self.netG_GAM_A(self.fake_A))*self.fake_A
        # self.fake_B=(0.6*self.netG_GAM_A(self.real_A)+0.6)*self.Net_Gpart(self.real_A,self.mask_A,self.netG_Av,self.netG_A)+(0.4-0.6*self.netG_GAM_A(self.real_A))*self.real_A#make the output value to 0.7~1.3
        # self.rec_A=(0.6*self.netG_GAM_B(self.fake_B)+0.6)*self.Net_Gpart(self.fake_B,self.mask_A,self.netG_Bv,self.netG_B)+(0.4-0.6*self.netG_GAM_B(self.fake_B))*self.fake_B
        # self.fake_A=(0.6*self.netG_GAM_B(self.real_B)+0.6)*self.Net_Gpart(self.real_B,self.mask_B,self.netG_Bv,self.netG_B)+(0.4-0.6*self.netG_GAM_B(self.real_B))*self.real_B
        # self.rec_B=(0.6*self.netG_GAM_A(self.fake_A)+0.6)*self.Net_Gpart(self.fake_A,self.mask_B,self.netG_Av,self.netG_A)+(0.4-0.6*self.netG_GAM_A(self.fake_A))*self.fake_A
        self.fake_B=(self.AttWeight*self.netG_GAM_A(self.real_A)+self.AttBias)*self.Net_Gpart(self.real_A,self.mask_A,self.netG_Av,self.netG_A)+(1-self.AttBias-self.AttWeight*self.netG_GAM_A(self.real_A))*self.real_A
        self.rec_A=(self.AttWeight*self.netG_GAM_B(self.fake_B)+self.AttBias)*self.Net_Gpart(self.fake_B,self.mask_A,self.netG_Bv,self.netG_B)+(1-self.AttBias-self.AttWeight*self.netG_GAM_B(self.fake_B))*self.fake_B
        self.fake_A=(self.AttWeight*self.netG_GAM_B(self.real_B)+self.AttBias)*self.Net_Gpart(self.real_B,self.mask_B,self.netG_Bv,self.netG_B)+(1-self.AttBias-self.AttWeight*self.netG_GAM_B(self.real_B))*self.real_B
        self.rec_B=(self.AttWeight*self.netG_GAM_A(self.fake_A)+self.AttBias)*self.Net_Gpart(self.fake_A,self.mask_B,self.netG_Av,self.netG_A)+(1-self.AttBias-self.AttWeight*self.netG_GAM_A(self.fake_A))*self.fake_A

    def stylelosscal(self,styleb,stylev,rmask,fmask):
        cmask,smask,cweight,sweight,_,_=self.load_seg(fmask,rmask,self.c_code)
        StyleLoss=0.0
        for i in range(0,5):#five different layers from feature maps
            #print('i',i)
            batch=styleb[i].shape[0]
            height=styleb[i].shape[2]
            width=styleb[i].shape[3]
            sweight=F.interpolate(sweight,scale_factor=height/sweight.shape[2],mode="bilinear",align_corners=False)
            cweight=F.interpolate(cweight,scale_factor=height/cweight.shape[2],mode="bilinear",align_corners=False)
            styleb[i]=torch.mul(styleb[i],sweight)
            stylev[i]=torch.mul(stylev[i],cweight)
            for j in range(self.v_class,len(cmask)):#??5 for SHCD
                # print('j',j)
                # print(cmask[j].shape)
                cmask[j]=F.interpolate(cmask[j],scale_factor=height/cmask[j].shape[2],mode="bilinear",align_corners=False)
                smask[j]=F.interpolate(smask[j],scale_factor=height/smask[j].shape[2],mode="bilinear",align_corners=False)
                styleB=torch.mul(styleb[i],smask[j])
                styleV=torch.mul(stylev[i],cmask[j])
                #smaskmean=torch.mean(smask[j],dim=(1,2,3))
                #cmaskmean=torch.mean(cmask[j],dim=(1,2,3))
                smasksum=torch.sum(smask[j],dim=(1,2,3))
                cmasksum=torch.sum(cmask[j],dim=(1,2,3))
                gm_const=self.gram_matrix(styleB)
                gm_var=self.gram_matrix(styleV)
                for k in range(smasksum.shape[0]):#batch
                    if smasksum[k]>0:
                        gm_const[k]=gm_const[k]/smasksum[k]#liang
                    if cmasksum[k]>0:
                        gm_var[k]=gm_var[k]/cmasksum[k]
                StyleLoss+=torch.mean(torch.mul(gm_const-gm_var,gm_const-gm_var))
        return StyleLoss

    def backward_D_inv(self, netD, real, fake,maskr,maskf):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        _,realinvm=self.load_seginv(maskr,self.c_code_inv)
        _,fakeinvm=self.load_seginv(maskf,self.c_code_inv)
        realc=torch.mul(real,realinvm)
        fakec=torch.mul(fake,fakeinvm)
        _realinvm=self.Dmaskloader(realinvm)
        _fakeinvm = self.Dmaskloader(fakeinvm)
        pred_real = netD(realc.detach())
        #loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_real=self.criterionWeightGAN(pred_real,True,_realinvm)
        # Fake
        pred_fake = netD(fakec.detach())
        #loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_fake = self.criterionWeightGAN(pred_fake, False,_fakeinvm)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_v(self, netD, real, fake,maskr,maskf):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        _,realvm=self.load_segvar(maskr,self.c_code_var)
        _,fakevm=self.load_segvar(maskf,self.c_code_var)
        realc=torch.mul(real,realvm)
        fakec=torch.mul(fake,fakevm)
        _realvm=self.Dmaskloader(realvm)#Dmaskloader has no parameters
        _fakevm = self.Dmaskloader(fakevm)
        pred_real = netD(realc.detach())
        #loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_real=self.criterionWeightGAN(pred_real,True,_realvm)
        # Fake
        pred_fake = netD(fakec.detach())#detach: to ignore the calculation of fakec
        #loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_fake = self.criterionWeightGAN(pred_fake, False,_fakevm)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B)# we should discuss whether it is better to use pool
        self.loss_D_A = self.backward_D_inv(self.netD_A, self.real_B, self.fake_B, self.mask_B, self.mask_A)
        self.loss_D_Av = self.backward_D_v(self.netD_Av, self.real_B, self.fake_B, self.mask_B, self.mask_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        #fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_inv(self.netD_B,self.real_A, self.fake_A, self.mask_A, self.mask_B)
        self.loss_D_Bv = self.backward_D_v(self.netD_Bv,self.real_A, self.fake_A, self.mask_A, self.mask_B)#there is something wrong，caculate self.realA,fakeA,maskAmaskB,without mask
        #self.loss_D_B = self.loss_D_Binv + self.cW*self.loss_D_Bvar

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        #with torch.autograd.detect_anomaly():
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        alenmask,a1hotmask=self.load_seginv(self.mask_A,self.c_code_inv)#a1hot represents invariant
        blenmask,b1hotmask=self.load_seginv(self.mask_B,self.c_code_inv)
        _,a1hotmaskvar=self.load_segvar(self.mask_A,self.c_code_var)
        _,b1hotmaskvar=self.load_segvar(self.mask_B,self.c_code_var)
        invrealA=torch.mul(self.real_A,a1hotmask)
        invrealB=torch.mul(self.real_B,b1hotmask)
        varrealA=torch.mul(self.real_A,a1hotmaskvar)
        varrealB=torch.mul(self.real_B,b1hotmaskvar)
        invpredA=torch.mul(self.fake_B,a1hotmask)#fakeB part
        invpredB=torch.mul(self.fake_A,b1hotmask)#fakeA part
        varpredA=torch.mul(self.fake_B,a1hotmaskvar)#fakeB part
        varpredB=torch.mul(self.fake_A,b1hotmaskvar)#fakeA part
        invrecA=torch.mul(self.rec_A,a1hotmask)
        invrecB=torch.mul(self.rec_B,b1hotmask)
        varrecA=torch.mul(self.rec_A,a1hotmaskvar)
        varrecB=torch.mul(self.rec_B,b1hotmaskvar)
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.Net_Gpart(self.real_B,self.mask_B,self.netG_Av,self.netG_A)
            invidtA=torch.mul(self.idt_A,b1hotmask)
            #self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.loss_idt_A = self.criterionIdt(invidtA, invrealB) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.Net_Gpart(self.real_A,self.mask_A,self.netG_Bv,self.netG_B)
            invidtB=torch.mul(self.idt_B,a1hotmask)
            #self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            self.loss_idt_B = self.criterionIdt(invidtB, invrealA) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        stylevA1,stylevA2,stylevA3,stylevA4,stylevA5,contentVA=self.vgg_extractor(self.fake_B)
        stylevB1,stylevB2,stylevB3,stylevB4,stylevB5,contentVB=self.vgg_extractor(self.fake_A)
        stylebB1,stylebB2,stylebB3,stylebB4,stylebB5,contentBA=self.vgg_extractor(self.real_A)
        stylebA1,stylebA2,stylebA3,stylebA4,stylebA5,contentBB=self.vgg_extractor(self.real_B)
        #_,_,_,_,_,contentVA1=self.vgg_extractor(self.mid_B)
        #_,_,_,_,_,contentVB1=self.vgg_extractor(self.mid_A)
        StyleAb=[]
        StyleBb=[]
        StyleAv=[]
        StyleBv=[]
        StyleAb.append(stylebA1)
        StyleAb.append(stylebA2)
        StyleAb.append(stylebA3)
        StyleAb.append(stylebA4)
        StyleAb.append(stylebA5)
        StyleAv.append(stylevA1)
        StyleAv.append(stylevA2)
        StyleAv.append(stylevA3)
        StyleAv.append(stylevA4)
        StyleAv.append(stylevA5)
        StyleBb.append(stylebB1)
        StyleBb.append(stylebB2)
        StyleBb.append(stylebB3)
        StyleBb.append(stylebB4)
        StyleBb.append(stylebB5)
        StyleBv.append(stylevB1)
        StyleBv.append(stylevB2)
        StyleBv.append(stylevB3)
        StyleBv.append(stylevB4)
        StyleBv.append(stylevB5)
        StyleLossA=self.stylelosscal(StyleAb,StyleAv,self.mask_B,self.mask_A)
        StyleLossB=self.stylelosscal(StyleBb,StyleBv,self.mask_A,self.mask_B)
        #div=1/(contentVA.shape[1]*contentVA.shape[2]*contentVA.shape[3])
        #ContentLossA=torch.mean(torch.mul(contentVA-contentBA,contentVA-contentBA))+torch.mean(torch.mul(contentVA1-contentBA,contentVA1-contentBA))
        #ContentLossB=torch.mean(torch.mul(contentVB-contentBB,contentVB-contentBB))+torch.mean(torch.mul(contentVB1-contentBB,contentVB1-contentBB))
        ContentLossA=torch.mean(torch.mul(contentVA-contentBA,contentVA-contentBA))
        ContentLossB=torch.mean(torch.mul(contentVB-contentBB,contentVB-contentBB))
        self.loss_G_AB=(StyleLossA+StyleLossB+5*ContentLossA+5*ContentLossB)*0.01
        # GAN loss D_A(G_A(A)) D_B(G_B(B))
        _amask=self.Dmaskloader(a1hotmask)
        _bmask=self.Dmaskloader(b1hotmask)
        _amaskv=self.Dmaskloader(a1hotmaskvar)
        _bmaskv=self.Dmaskloader(b1hotmaskvar)
        # self.basegan1=self.criterionGAN(self.netD_B(invpredB), True)#this bmask represents invariant area
        # self.basegan2=self.criterionGAN(self.netD_A(invpredA), True)#
        self.basegan1=self.criterionWeightGAN(self.netD_B(invpredB), True,_bmask)#this bmask represents invariant area
        self.basegan2=self.criterionWeightGAN(self.netD_A(invpredA), True,_amask)#
        self.baseganv1=self.criterionWeightGAN(self.netD_Bv(varpredB), True,_bmaskv)#
        self.baseganv2=self.criterionWeightGAN(self.netD_Av(varpredA), True,_amaskv)#
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(invrecA, invrealA) * lambda_A
        self.loss_cycle_Av = self.criterionCycle(varrecA, varrealA) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(invrecB, invrealB) * lambda_B
        self.loss_cycle_Bv = self.criterionCycle(varrecB, varrealB) * lambda_B
        # combined loss and calculate gradients

        amzeros=torch.zeros_like(a1hotmask)
        bmzeros=torch.zeros_like(b1hotmask)

        if torch.equal(b1hotmask,bmzeros):
            self.basegan1=0
        if torch.equal(a1hotmask,amzeros):
            self.basegan2=0
        # structural similarity
        self.loss_SSIM=2-self.criterionSSIM(self.real_A,self.fake_B)-self.criterionSSIM(self.real_B,self.fake_A)
        self.loss_G = (1-self.cW)*self.loss_G_AB +self.basegan1+self.basegan2+self.cW*(self.baseganv1+self.baseganv2+self.loss_cycle_Av + self.loss_cycle_Bv)+self.loss_cycle_A + self.loss_cycle_B+self.loss_SSIM
        #self.loss_G = self.loss_G_AB +self.basegan1+self.basegan2+self.loss_cycle_A + self.loss_cycle_B+self.loss_SSIM
        #self.loss_G = self.loss_G_AB +self.basegan1+self.basegan2+self.loss_SSIM+self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()


    def backward_G_att(self):
        """Calculate the loss for generators G_A and G_B to optimize weight net"""
        #with torch.autograd.detect_anomaly():
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        alenmask,a1hotmask=self.load_seginv(self.mask_A,self.c_code_inv)
        blenmask,b1hotmask=self.load_seginv(self.mask_B,self.c_code_inv)
        _,a1hotmaskvar=self.load_segvar(self.mask_A,self.c_code_var)
        _,b1hotmaskvar=self.load_segvar(self.mask_B,self.c_code_var)

        invrealA=torch.mul(self.real_A,a1hotmask)
        invrealB=torch.mul(self.real_B,b1hotmask)
        #varrealA=torch.mul(self.real_A,a1hotmaskvar)
        #varrealB=torch.mul(self.real_B,b1hotmaskvar)
        invpredA=torch.mul(self.fake_B,a1hotmask)#fakeB part
        invpredB=torch.mul(self.fake_A,b1hotmask)#fakeA part
        varpredA=torch.mul(self.fake_B,a1hotmaskvar)#fakeB part
        varpredB=torch.mul(self.fake_A,b1hotmaskvar)#fakeA part
        invrecA=torch.mul(self.rec_A,a1hotmask)
        invrecB=torch.mul(self.rec_B,b1hotmask)
        #varrecA=torch.mul(self.rec_A,a1hotmaskvar)
        #varrecB=torch.mul(self.rec_B,b1hotmaskvar)

        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            #self.idt_A=self.netG_GAM_A(self.real_B)*self.Net_Gpart(self.real_B,self.mask_B,self.netG_A,self.netG_AC)+(1-self.netG_GAM_A(self.real_B))*self.real_B
            #self.idt_A=(1.2*self.netG_GAM_A(self.real_B))*self.Net_Gpart(self.real_B,self.mask_B,self.netG_Av,self.netG_A)+(1-1.2*self.netG_GAM_A(self.real_B))*self.real_B
            #self.idt_A=(0.6*self.netG_GAM_A(self.real_B)+0.6)*self.Net_Gpart(self.real_B,self.mask_B,self.netG_Av,self.netG_A)+(0.4-0.6*self.netG_GAM_A(self.real_B))*self.real_B#??
            self.idt_A=(self.AttWeight*self.netG_GAM_A(self.real_B)+self.AttBias)*self.Net_Gpart(self.real_B,self.mask_B,self.netG_Av,self.netG_A)+(1-self.AttBias-self.AttWeight*self.netG_GAM_A(self.real_B))*self.real_B
            #self.idt_A = self.Net_Gpart(self.real_B,self.mask_B,self.netG_A,self.netG_AC)
            invidtA=torch.mul(self.idt_A,b1hotmask)
            #self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.loss_idt_A = self.criterionIdt(invidtA, invrealB) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            #self.idt_B=self.netG_GAM_B(self.real_A)*self.Net_Gpart(self.real_A,self.mask_A,self.netG_B,self.netG_BC)+(1-self.netG_GAM_B(self.real_A))*self.real_A
            #self.idt_B=(1.2*self.netG_GAM_B(self.real_A))*self.Net_Gpart(self.real_A,self.mask_A,self.netG_Bv,self.netG_B)+(1-1.2*self.netG_GAM_B(self.real_A))*self.real_A
            #self.idt_B=(0.6*self.netG_GAM_B(self.real_A)+0.6)*self.Net_Gpart(self.real_A,self.mask_A,self.netG_Bv,self.netG_B)+(0.4-0.6*self.netG_GAM_B(self.real_A))*self.real_A#??
            self.idt_B=(self.AttWeight*self.netG_GAM_B(self.real_A)+self.AttBias)*self.Net_Gpart(self.real_A,self.mask_A,self.netG_Bv,self.netG_B)+(1-self.AttBias-self.AttWeight*self.netG_GAM_B(self.real_A))*self.real_A
            #self.idt_B = self.Net_Gpart(self.real_A,self.mask_A,self.netG_B,self.netG_BC)
            invidtB=torch.mul(self.idt_B,a1hotmask)
            #self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            self.loss_idt_B = self.criterionIdt(invidtB, invrealA) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        # stylevA1,stylevA2,stylevA3,stylevA4,stylevA5,contentVA=self.vgg_extractor(torch.mul(self.fake_B,a1hotmaskvar))#the version of delete invariant part
        # stylevB1,stylevB2,stylevB3,stylevB4,stylevB5,contentVB=self.vgg_extractor(torch.mul(self.fake_A,b1hotmaskvar))
        # stylebB1,stylebB2,stylebB3,stylebB4,stylebB5,contentBA=self.vgg_extractor(torch.mul(self.real_A,a1hotmaskvar))
        # stylebA1,stylebA2,stylebA3,stylebA4,stylebA5,contentBB=self.vgg_extractor(torch.mul(self.real_B,b1hotmaskvar))
        stylevA1,stylevA2,stylevA3,stylevA4,stylevA5,contentVA=self.vgg_extractor(self.fake_B)
        stylevB1,stylevB2,stylevB3,stylevB4,stylevB5,contentVB=self.vgg_extractor(self.fake_A)
        stylebB1,stylebB2,stylebB3,stylebB4,stylebB5,contentBA=self.vgg_extractor(self.real_A)
        stylebA1,stylebA2,stylebA3,stylebA4,stylebA5,contentBB=self.vgg_extractor(self.real_B)
        StyleAb=[]
        StyleBb=[]
        StyleAv=[]
        StyleBv=[]
        StyleAb.append(stylebA1)
        StyleAb.append(stylebA2)
        StyleAb.append(stylebA3)
        StyleAb.append(stylebA4)
        StyleAb.append(stylebA5)
        StyleAv.append(stylevA1)
        StyleAv.append(stylevA2)
        StyleAv.append(stylevA3)
        StyleAv.append(stylevA4)
        StyleAv.append(stylevA5)
        StyleBb.append(stylebB1)
        StyleBb.append(stylebB2)
        StyleBb.append(stylebB3)
        StyleBb.append(stylebB4)
        StyleBb.append(stylebB5)
        StyleBv.append(stylevB1)
        StyleBv.append(stylevB2)
        StyleBv.append(stylevB3)
        StyleBv.append(stylevB4)
        StyleBv.append(stylevB5)
        StyleLossA=self.stylelosscal(StyleAb,StyleAv,self.mask_B,self.mask_A)
        StyleLossB=self.stylelosscal(StyleBb,StyleBv,self.mask_A,self.mask_B)
        #div=1/(contentVA.shape[1]*contentVA.shape[2]*contentVA.shape[3])
        #ContentLossA=torch.mean(torch.mul(contentVA-contentBA,contentVA-contentBA))+torch.mean(torch.mul(contentVA1-contentBA,contentVA1-contentBA))#mean就不需要div了吧感觉
        #ContentLossB=torch.mean(torch.mul(contentVB-contentBB,contentVB-contentBB))+torch.mean(torch.mul(contentVB1-contentBB,contentVB1-contentBB))
        ContentLossA=torch.mean(torch.mul(contentVA-contentBA,contentVA-contentBA))
        ContentLossB=torch.mean(torch.mul(contentVB-contentBB,contentVB-contentBB))
        self.loss_G_AB=(StyleLossA+StyleLossB+5*ContentLossA+5*ContentLossB)*0.01
        #self.loss_G_AB=(StyleLossA+StyleLossB)*0.01
        #self.loss_G_AB=(5*ContentLossA+5*ContentLossB)*0.01
        # GAN loss D_A(G_A(A)) D_B(G_B(B))
        _amask=self.Dmaskloader(a1hotmask)
        _bmask=self.Dmaskloader(b1hotmask)
        _amaskv=self.Dmaskloader(a1hotmaskvar)
        _bmaskv=self.Dmaskloader(b1hotmaskvar)
        self.basegan1=self.criterionWeightGAN(self.netD_B(invpredB), True,_bmask)#
        self.basegan2=self.criterionWeightGAN(self.netD_A(invpredA), True,_amask)#
        self.baseganv1=self.criterionWeightGAN(self.netD_Bv(varpredB), True,_bmaskv)#
        self.baseganv2=self.criterionWeightGAN(self.netD_Av(varpredA), True,_amaskv)#
        # Forward cycle loss || G_B(G_A(A)) - A||
        #self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_A = self.criterionCycle(invrecA, invrealA) * lambda_A
        #self.loss_cycle_Av = self.criterionCycle(varrecA, varrealA) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        #self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_cycle_B = self.criterionCycle(invrecB, invrealB) * lambda_B
        #self.loss_cycle_Bv = self.criterionCycle(varrecB, varrealB) * lambda_B
        # combined loss and calculate gradients


        amzeros=torch.zeros_like(a1hotmask)
        bmzeros=torch.zeros_like(b1hotmask)
        if torch.equal(b1hotmask,bmzeros):
            self.basegan1=0
        if torch.equal(a1hotmask,amzeros):
            self.basegan2=0
        #self.loss_SSIM=2-self.criterionSSIM(self.real_A,self.fake_B)-self.criterionSSIM(self.real_B,self.fake_A)
        #self.loss_SSIM=2-self.criterionSSIM(self.real_A,self.mid_B)-self.criterionSSIM(self.real_B,self.mid_A)+2-self.criterionSSIM(self.real_A,self.fake_B)-self.criterionSSIM(self.real_B,self.fake_A)
        self.loss_SSIM=2-self.criterionSSIM(self.real_A,self.fake_B)-self.criterionSSIM(self.real_B,self.fake_A)
        self.loss_G = (1-self.cW)*self.loss_G_AB +self.basegan1+self.basegan2+self.cW*(self.baseganv1+self.baseganv2)+self.loss_SSIM
        #self.loss_G = self.loss_G_AB +self.basegan1+self.basegan2+self.loss_SSIM
        #self.loss_G = self.loss_G_AB +self.basegan1+self.basegan2+self.loss_SSIM+self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        #self.loss_G = self.loss_G_AB +self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B+self.basegan1+self.basegan2+2*self.loss_SSIM
        self.loss_G.backward()

    def set_new_optimnet(self):
        """change the optimized networks to realize two-step strategy"""
        print("optlr",self.opt.lr)
        self.optimizer_G_att = torch.optim.Adam(itertools.chain(self.netG_GAM_A.parameters(),self.netG_GAM_B.parameters()),lr=self.opt.lr,betas=(self.opt.beta1, 0.999))#set different learning rates for different networks
        self.model_names = ['G_GAM_A', 'G_GAM_B', 'D_A','D_B', 'D_Av','D_Bv']#???
        #self.model_names = ['G_GAM_A', 'G_GAM_B', 'D_A','D_B']
        self.optimizers.append(self.optimizer_G_att)
        def lambda_rule_att(epoch):
            lr_l = 1.0 - max(0, epoch + self.opt.epoch_count - self.opt.n_epochs-self.opt.n_epochs_decay) / float(self.opt.n_epochs_att + 1)
            return lr_l
        self.schedulers = [lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule_att) for optimizer in self.optimizers]#start new lr policy

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A,self.netD_B,self.netD_Av,self.netD_Bv], False)
        #self.set_requires_grad([self.netD_A,self.netD_B], False)#??
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B 
        self.optimizer_G.step()       # update G_A and G_B's weights
        # # D_A and D_B
        self.set_requires_grad([self.netD_A,self.netD_B,self.netD_Av,self.netD_Bv], True)#optimize basic discriminator
        #self.set_requires_grad([self.netD_A,self.netD_B], True)#optimize basic discriminator
        self.optimizer_D.zero_grad()
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def optimize_Att(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward_att()      # compute fake images and reconstruction images.
        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A,self.netD_B,self.netD_Av,self.netD_Bv], False)
        #self.set_requires_grad([self.netD_A,self.netD_B], False)
        self.optimizer_G_att.zero_grad()  # set gradients to zero
        self.backward_G_att()             # calculate gradients for the attention network
        self.optimizer_G_att.step()       # update the weight of attention network
        # # D_A and D_B
        self.set_requires_grad([self.netD_A,self.netD_B,self.netD_Av,self.netD_Bv], True)#optimize basic discriminator
        #self.set_requires_grad([self.netD_A,self.netD_B], True)#optimize basic discriminator
        self.optimizer_D.zero_grad()
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
