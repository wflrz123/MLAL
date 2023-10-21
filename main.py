from __future__ import division, print_function, absolute_import

import os
import pdb
import copy
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchvision import utils
from tqdm import tqdm
import matplotlib.pyplot as plt                                  
from learner import Learner
from metalearner import MetaLearner
from dataloader import prepare_data
from utils import *
from PIL import Image
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

# add by rzliao
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', choices=['train', 'test'])
# Hyper-parameters
FLAGS.add_argument('--n-shot', type=int,
                   help="How many examples per class for training (k, n_support)")
FLAGS.add_argument('--n-eval', type=int,
                   help="How many examples per class for evaluation (n_query)")
FLAGS.add_argument('--n-class', type=int,
                   help="How many classes (N, n_way)")
FLAGS.add_argument('--input-size', type=int,
                   help="Input size for the first LSTM")
FLAGS.add_argument('--hidden-size', type=int,
                   help="Hidden size for the first LSTM")
FLAGS.add_argument('--lr', type=float,
                   help="Learning rate")
FLAGS.add_argument('--episode', type=int,
                   help="Episodes to train")
FLAGS.add_argument('--episode-val', type=int,
                   help="Episodes to eval")
FLAGS.add_argument('--epoch', type=int,
                   help="Epoch to train for an episode")
FLAGS.add_argument('--batch-size', type=int,
                   help="Batch size when training an episode")
FLAGS.add_argument('--image-size', type=int,
                   help="Resize image to this size")
FLAGS.add_argument('--grad-clip', type=float,
                   help="Clip gradients larger than this number")
FLAGS.add_argument('--bn-momentum', type=float,
                   help="Momentum parameter in BatchNorm2d")
FLAGS.add_argument('--bn-eps', type=float,
                   help="Eps parameter in BatchNorm2d")

# Paths
FLAGS.add_argument('--data', choices=['miniimagenet'],
                   help="Name of dataset")
FLAGS.add_argument('--data-root', type=str,
                   help="Location of data")
FLAGS.add_argument('--resume', type=str,
                   help="Location to pth.tar")
FLAGS.add_argument('--save', type=str, default='logs',
                   help="Location to logs and ckpts")
# Others
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")
FLAGS.add_argument('--n-workers', type=int, default=4,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', type=bool, default=False,
                   help="DataLoader pin_memory")
FLAGS.add_argument('--log-freq', type=int, default=100,
                   help="Logging frequency")
FLAGS.add_argument('--val-freq', type=int, default=1000,
                   help="Validation frequency")
FLAGS.add_argument('--seed', type=int,
                   help="Random seed")

fmap_block = dict()  # 装feature map

def show_cam(cam, imgs):
    cam = cv2.resize(cam * 255., (84, 84)).astype('uint8') # 调整热图尺寸与图片
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET) # 将热图转化为“伪彩热图”显示模式
    superimposed_img = cv2.addWeighted(cam, .3, imgs, .7, 1.) # 将特图叠加到原图片上
    cv2.imwrite('cam.jpg', superimposed_img)
    
def show_gradcam(cam, imgs, c):
    cam = cv2.resize(cam[c] * 255., (84, 84)).astype('uint8') # 调整热图尺寸与图片
    
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET) # 将热图转化为“伪彩热图”显示模式
    cv2.imwrite('heatmap'+str(c)+'.jpg', cam)
    superimposed_img = cv2.addWeighted(cam, .3, imgs, .7, 0.) # 将特图叠加到原图片上
    cv2.imwrite('cam'+str(c)+'.jpg', superimposed_img)

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (84, 84)
    bz, nc, h, w = feature_conv.shape # 获取feature_conv特征的尺寸
    output_cam = []
    #lass_idx为预测分值较大的类别的数字表示的数组，一张图片中有N类物体则数组中N个元素
    for idx in class_idx:
    # weight_softmax中预测为第idx类的参数w乘以feature_map(为了相乘，故reshape了map的形状)

        cam = weight_softmax[idx]*(feature_conv.reshape((nc, h*w))) # 把原来的相乘再相加转化为矩阵
                                                                    # w1*c1 + w2*c2+ .. -> (w1,w2..) * (c1,c2..)^T -> (w1,w2...)*((c11,c12,c13..),(c21,c22,c23..))

        # 将feature_map的形状reshape回去
        cam = cam.reshape(h, w)
        # 归一化操作（最小的值为0，最大的为1）
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        # 转换为图片的255的数据
        cam_img = np.uint8(255 * cam_img)
        # resize 图片尺寸与输入图片一致
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
    
def farward_hook(module, inp, outp):
    fmap_block['input'] = inp
    fmap_block['output'] = outp
    
def imageSavePLT(images,fileName,normalization=True,mean=0,std=1):
    image = utils.make_grid(images)
    image = image.permute(1,2,0)
    if normalization:
        image = (image*torch.tensor(std)+torch.tensor(mean)).numpy()
    plt.imsave(fileName,image)
    return image
def hook_fn(grad):
    print(grad.shape)
def meta_test(eps, eval_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger):
    c=0
    sum_test=0
    for subeps, (episode_x, episode_y) in enumerate(tqdm(eval_loader, ascii=True)):
        train_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev) # [n_class * n_shot]
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.eval()
        
        
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, test_input, args)

        learner_wo_grad.transfer_params(learner_w_grad, cI)
        learner_wo_grad.model.features2.register_forward_hook(farward_hook)
        output, x_features = learner_wo_grad(test_input, test_input)

        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
        sum_test+=len(test_input)

        for name, param in learner_w_grad.named_parameters():
            if name=='model.features2.conv4.weight':
                conv=param.grad

        if c<75:
            params = {}
            for name, param in learner_wo_grad.named_parameters():
                params[name]=param
            weight_softmax = np.squeeze(params['model.cls.weight'].data.cpu().numpy()) 

            # grad_cam
            img =  imageSavePLT(test_input[c].cpu(),'img'+str(c)+'.png',std=[0.229, 0.224, 0.225],mean=[0.485, 0.456, 0.406])
            img = cv2.imread('img'+str(c)+'.png')
            grad_map = fmap_block['output']

            grad = torch.mean(grad_map,(2,3),keepdim=True)
            gradcam = torch.sum(grad * grad_map,dim=1)
            gradcam = torch.maximum(gradcam.cpu(), torch.zeros(75,10,10).cpu())
            for j in range(gradcam.shape[0]):
                gradcam[j] =gradcam[j] / torch.max(gradcam[j])
            
            show_gradcam(gradcam.detach().numpy(), img, c)
            c=c+1
        
        logger.batch_info(loss=loss.item(), acc=acc, phase='eval')

    return logger.batch_info(eps=eps, totaleps=args.episode_val, phase='evaldone')


def train_learner(learner_w_grad, metalearner, train_input, train_target, test_input, args):
    cI = metalearner.metalstm.cI.data
    hs = [None]

    for c in range(args.epoch):
        for i in range(0, len(train_input), args.batch_size):

            x = train_input[i:i+args.batch_size]
#             print(train_input.size(),x.size())
            if args.n_shot==5:
                r_t = random.randint(0,10)
                x_t0 = test_input[0+r_t:5+r_t]
                x_t1=test_input[args.n_eval+r_t:args.n_eval+5+r_t]
                x_t2=test_input[2*args.n_eval+r_t:2*args.n_eval+5+r_t]
                x_t3=test_input[3*args.n_eval+r_t:3*args.n_eval+5+r_t]
                x_t4=test_input[4*args.n_eval+r_t:4*args.n_eval+5+r_t]
            elif args.n_shot==1:
                r_t = random.randint(0,14)
                x_t0 = test_input[r_t:1+r_t]
                x_t1=test_input[args.n_eval+r_t:args.n_eval+r_t+1]
                x_t2=test_input[2*args.n_eval+r_t:2*args.n_eval+r_t+1]
                x_t3=test_input[3*args.n_eval+r_t:3*args.n_eval+r_t+1]
                x_t4=test_input[4*args.n_eval+r_t:4*args.n_eval+r_t+1]

            x_t=torch.cat((x_t0,x_t1,x_t2,x_t3,x_t4),0)
            y = train_target[i:i+args.batch_size]
            
            learner_w_grad.copy_flat_params(cI)
            output, x_features = learner_w_grad(x, x_t)
            loss = learner_w_grad.criterion(output, y)

            acc = accuracy(output, y)

            learner_w_grad.zero_grad()
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) / args.batch_size for p in learner_w_grad.parameters()], 0)
            sum=0

            grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
            #将损失，梯度和上一个状态的元学习参数提供给元学习器
            cI, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

    return cI


def main():

    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    if args.seed is None:
        args.seed = random.randint(0, 1e3)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cpu:
        args.dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.dev = torch.device('cuda')

    logger = GOATLogger(args)

    # Get data
    train_loader, val_loader, test_loader = prepare_data(args)
    
    # Set up learner, meta-learner
    learner_w_grad = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_class).to(args.dev)
    learner_wo_grad = copy.deepcopy(learner_w_grad)
    
    metalearner = MetaLearner(args.input_size, args.hidden_size, learner_w_grad.get_flat_params().size(0)).to("cuda")

    metalearner.metalstm.init_cI(learner_w_grad.get_flat_params())
    
    # Set up loss, optimizer, learning rate scheduler
    optim = torch.optim.Adam(metalearner.parameters(), args.lr)

    if args.resume:
        logger.loginfo("Initialized from: {}".format(args.resume))
        last_eps, metalearner, optim = resume_ckpt(metalearner, optim, args.resume, args.dev)

    if args.mode == 'test':
        print("测试模式")

        _ = meta_test(last_eps, test_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
        return

    best_acc = 0.0
    logger.loginfo("Start training")
    loss_list = []
    acc_list = []
    # Meta-training
    for eps, (episode_x, episode_y) in enumerate(train_loader):
        # episode_x.shape = [n_class, n_shot + n_eval, c, h, w]
        # episode_y.shape = [n_class, n_shot + n_eval] --> NEVER USED

        train_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev) # [n_class * n_shot]
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.train()

        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, test_input, args)
        # Train meta-learner with validation loss
        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output, x_features = learner_wo_grad(test_input, test_input)

        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(metalearner.parameters(), args.grad_clip)
        optim.step()

        logger.batch_info(eps=eps, totaleps=args.episode, loss=loss.item(), acc=acc, phase='train')

        # Meta-validation
        if eps % args.val_freq == 0 and eps != 0:


            save_ckpt(eps, metalearner, optim, args.save)
            loss, acc = meta_test(eps, val_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
            loss_list.append(loss)
            acc_list.append(acc)
            if acc > best_acc:
                best_acc = acc
                logger.loginfo("* Best accuracy so far *\n")
    logger.loginfo("Done")


if __name__ == '__main__':
    main()
