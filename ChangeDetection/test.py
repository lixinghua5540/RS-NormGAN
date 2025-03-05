import sys

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import datasetsentinel as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
from PIL import Image
from collections import Counter
import os, time
import numpy as np
from argparse import ArgumentParser
from Evaluate_operation import excel_confu_matrix, metrics
import cv2
from models.A2Net import A2Net
from models.DMINet import DMINet
from models.FCEF import FC_siam_diff, FC_siam_diff2
from models.SEIFNet import SEIFNet
from models.DsferNet import MyNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

color_codes_BGR=[[0,0,0],[0,0,255],[0,255,255],[255,0,0],[128,0,128],[255,255,255]]#BGR for SHCD
#color_codes_BGR=[[0,0,0],[255,255,255]]#BGR for GESD binary change detection

def cal_confu_matrix(label,predict,class_num): 
    confu_list=[]
    for i in range(class_num):
        c=Counter(label[np.where(predict==i)])   
        single_row=[]
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)

def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice


def BCE(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


def MSEDiceLoss(inputs, targets):
    #mse = F.binary_cross_entropy(inputs, targets)
    mse = F.cross_entropy(inputs, targets.squeeze().to(torch.long))#the target should be batches of number and has one less dimension than inputs
    #inter = (inputs * targets).sum()#dice can not calculate in this way because we use several classes
    eps = 1e-5
    #dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    #return mse + 1 - dice
    return mse

@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=6)

    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    #h=1000 #config for GESD
    #w=1160
    #w=740
    #w=950
    #wp=768
    #wp=1024
    h=1792 #config for SHCD
    w=1536
    hp=1792
    wp=1536

    predwhole=torch.zeros((4,hp,wp)).cuda().to(dtype=torch.float)
    pred_mode=torch.zeros((hp,wp)).cuda().to(dtype=torch.uint8)
    pred_color=torch.zeros((3,h,w)).cuda().to(dtype=torch.uint8)
    patchsize=256
    for iter, batched_inputs in enumerate(val_loader):

        img, target, position = batched_inputs
        img_name = val_loader.sampler.data_source.file_list[iter]#？
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        #output, output2, output3, output4 = model(pre_img_var, post_img_var)
        #output, temp1, temp2 = model(pre_img_var, post_img_var)
        output = model(pre_img_var, post_img_var)
        # loss = BCEDiceLoss(output, target_var) + BCEDiceLoss(output2, target_var) + BCEDiceLoss(output3, target_var) + \
        #        BCEDiceLoss(output4, target_var)
        # loss = MSEDiceLoss(output, target_var) + MSEDiceLoss(output2, target_var) + MSEDiceLoss(output3, target_var) + \
        #        MSEDiceLoss(output4, target_var)#BCEDiceLoss is not appropriate for multi-class change detection

        #pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()
        pred = F.log_softmax(output,dim=1)
        Pred = torch.squeeze(torch.argmax(pred,dim=1))#send to
        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        # epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
             output = gather(Pred, 0, dim=0)

        position_x=int(position[0].split("_")[0])#获得位置直接拼接成完整结果
        position_y=int(position[0].split("_")[1])

        if (position_x%2==0 and position_y%2==0):
            predwhole[0,0+position_x*128:256+position_x*128,0+position_y*128:256+position_y*128]=Pred
            #predwhole[0,0+position_x*128:256+position_x*128,0+position_y*128:256+position_y*128]=pred
        elif (position_x%2==0 and position_y%2!=0):
            predwhole[1,0+position_x*128:256+position_x*128,0+position_y*128:256+position_y*128]=Pred
            #predwhole[1,0+position_x*128:256+position_x*128,0+position_y*128:256+position_y*128]=pred
        elif (position_x%2!=0 and position_y%2==0):
            predwhole[2,0+position_x*128:256+position_x*128,0+position_y*128:256+position_y*128]=Pred
            #predwhole[2,0+position_x*128:256+position_x*128,0+position_y*128:256+position_y*128]=pred
        else:
            predwhole[3,0+position_x*128:256+position_x*128,0+position_y*128:256+position_y*128]=Pred
            #predwhole[3,0+position_x*128:256+position_x*128,0+position_y*128:256+position_y*128]=pred
                
        # save change maps
        #pr = pred[0, 0].cpu().numpy()
        #gt = target_var[0, 0].cpu().numpy()
        # index_tp = np.where(np.logical_and(pr == 1, gt == 1))
        # index_fp = np.where(np.logical_and(pr == 1, gt == 0))
        # index_tn = np.where(np.logical_and(pr == 0, gt == 0))
        # index_fn = np.where(np.logical_and(pr == 0, gt == 1))
        # #
        # map = np.zeros([gt.shape[0], gt.shape[1], 3])
        # map[index_tp] = [255, 255, 255]  # white
        # map[index_fp] = [255, 0, 0]  # red
        # map[index_tn] = [0, 0, 0]  # black
        # map[index_fn] = [0, 255, 255]  # Cyan

        #change_map = Image.fromarray(np.array(map, dtype=np.uint8))
        #change_map.save(args.vis_dir + img_name)
        #if iter % 5 == 0:
        #    print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' % (iter, total_batches, f1, loss.data.item(), time_taken),
        #          end='')
    pred_mode=predwhole[0,:,:]
    no_edge=predwhole[:,int(patchsize/2):hp-int(patchsize/2),int(patchsize/2):wp-int(patchsize/2)]
    no_edge_mode,_=torch.mode(no_edge,dim=0)

    pred_mode[int(patchsize/2):hp-int(patchsize/2),int(patchsize/2):wp-int(patchsize/2)]=no_edge_mode
    pred_mode_part=pred_mode[0:h,0:w]
    for i in range(len(color_codes_BGR)):
        i_ind=(pred_mode_part==i)
        pred_color[0][i_ind]=color_codes_BGR[i][0]
        pred_color[1][i_ind]=color_codes_BGR[i][1]
        pred_color[2][i_ind]=color_codes_BGR[i][2]
    #pred_color_part=pred_color[:,0:h,0:w]
    cv2.imwrite("./results/net1_31_final.tif",pred_color.permute(1,2,0).to("cpu").numpy())
    real_label=cv2.imread("./results/CDmask1.tif",0)
    print(real_label)
    print(pred_mode_part.cpu().numpy())
    print(real_label.shape)
    print(pred_mode_part.cpu().numpy().shape)
    confu_list=cal_confu_matrix(real_label,pred_mode_part.cpu().numpy(),class_num=6)#
    excel_confu_matrix(confu_list,save_path="./results/",save_name="net1_31_final")
    F1, IoU,_=metrics(confu_list,save_path="./results/",save_name="net1_31_final")

        #f1 = salEvalVal.update_cm(pr, gt)#pr gt are new variables



    #average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    #scores = salEvalVal.get_scores()

    return 1


def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    #model = A2Net(3, 6)
    #model = DMINet(6)
    model = FC_siam_diff(3,6)
    #model = SEIFNet(args,input_nc=3, output_nc=6)
    #model = MyNet(n_classes=6, beta=512, dim = 512, numhead = 1)#default setting
    method="net1_31"
    print("predict result of:",method)
    args.savedir = args.savedir + '_' + method+ args.file_root +'_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/'
    args.vis_dir = './Predict/' + args.file_root + '/'

    if args.file_root == 'LEVIR':
        args.file_root = '../samples'
        # args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/LEVIR-CD_256_patches'
    elif args.file_root == 'BCDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/BCDD'
    elif args.file_root == 'SYSU':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/SYSU'
    elif args.file_root == 'CDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/CDD'
    elif args.file_root == 'testLEVIR':
        args.file_root = '../samples'
    elif args.file_root == 'SentinelCD':
        args.file_root = './Datasets/SentinelCD'
    elif args.file_root == 'S2W':
        args.file_root = './Datasets/S2W'
    elif args.file_root == 'S2W2':
        args.file_root = './Datasets/S2W2'
    elif args.file_root == 'S2A':
        args.file_root = './Datasets/S2A'
    elif args.file_root == 'S2A2':
        args.file_root = './Datasets/S2A2'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    valDataset = myTransforms.Compose([
        myTransforms.NormalizeSentinel(mean=mean, std=std),
        #myTransforms.Normalize(mean=mean, std=std),
        #myTransforms.Scale(args.inWidth, args.inHeight),#
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.DatasetSentinel("test","net1_3", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=1, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # load the model
    model_file_name = args.savedir + 'best_model.pth'
    #model_file_name = args.savedir + 'best_model_kappa.pth'
    #model_file_name = args.savedir + '2500.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    _ = val(args, testLoader, model, 0)
    #print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
    #      % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    #logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Test', score_test['Kappa'], score_test['IoU'],
    #                                                               score_test['F1'], score_test['recall'],
    #                                                               score_test['precision']))
    logger.flush()
    logger.close()

    import scipy.io as scio

    #scio.savemat(args.vis_dir + 'results.mat', score_test)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="SentinelCD", help='Data directory | LEVIR | BCDD | SYSU ')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=3, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training | '
                                                       './results_ep100/checkpoint.pth.tar')
    parser.add_argument('--logFile', default='testLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    ValidateSegmentation(args)
