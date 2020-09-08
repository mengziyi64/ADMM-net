from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import ADMM_net
from utils import generate_masks, time2file_name, A, At
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

#data_path = "../Data/B_8_DAVIS2017/train/480p"  # traning data from DAVIS2017
data_path = "/home/jma/Ziyi/CS_CACTI/Data/B_8_DAVIS2017_expand/train/480p"  # B_8_DAVIS2017_expand  B_8_DAVIS2016

test_path1 = "/home/jma/Ziyi/CS_CACTI/Data/test/simulation"  # simulation data for comparison


mask, mask_s = generate_masks(data_path)

last_train = 0
model_save_filename = ''
max_iter = 200
batch_size = 5
learning_rate = 0.001
stage_num = 9
mode = 'train'  # train or test

dataset = Imgdataset(data_path)
train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#first_frame_net = cnn1().cuda()
network = ADMM_net().cuda()

if last_train != 0:
    network = torch.load(
        './model/' + model_save_filename + "/model_epoch_{}.pth".format(last_train))

criterion  = nn.MSELoss()
criterion.cuda()


def test(test_path, epoch, result_path, psnr_epoch):
    test_list = os.listdir(test_path)
    psnr_sample = torch.zeros(len(test_list))
    pred = []
    for i in range(len(test_list)):
        pic = scio.loadmat(test_path + '/' + test_list[i])

        if "orig" in pic:
            pic = pic['orig']
            sign = 1
        elif "patch_save" in pic:
            pic = pic['patch_save']
            sign = 0
        elif "p1" in pic:
            pic = pic['p1']
            sign = 0
        elif "p2" in pic:
            pic = pic['p2']
            sign = 0
        elif "p3" in pic:
            pic = pic['p3']
            sign = 0
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // 8, 8, 256, 256])
        for jj in range(pic.shape[2]):
            if jj % 8 == 0:
                meas_t = np.zeros([256, 256])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // 8, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % 8 == 0 and jj != 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        meas = torch.from_numpy(meas)
        pic_gt = torch.from_numpy(pic_gt)
        meas = meas.cuda()
        pic_gt = pic_gt.cuda()
        meas = meas.float()
        pic_gt = pic_gt.float()
        batch_size1 = pic_gt.shape[0]
            
        y = meas            # [batch,256 256]
        Phi = mask.expand([batch_size1, 8, 256, 256])
        Phi_s = mask_s.expand([batch_size1, 256, 256])
        with torch.no_grad():
            out_pic_list = network(y, Phi, Phi_s)
            out_pic = out_pic_list[-1]

            psnr_1 = 0
            for ii in range(meas.shape[0] * 8):
                out_pic_p = out_pic[ii // 8, ii % 8, :, :]
                gt_t = pic_gt[ii // 8, ii % 8, :, :]
                rmse = torch.sqrt(criterion(out_pic_p, gt_t))
                rmse = rmse.data
                psnr_1 += 10 * torch.log10(1 / criterion(out_pic_p, gt_t))
            psnr_1 = psnr_1 / (meas.shape[0] * 8)
            psnr_sample[i] = psnr_1
        
        pred.append(out_pic.cpu().numpy())
        
    psnr_epoch.append(psnr_sample)
    return pred, psnr_epoch



def train(epoch, learning_rate):
    epoch_loss = 0
    begin = time.time()

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    if __name__ == '__main__':
        for iteration, batch in enumerate(train_data_loader):
            gt, meas = Variable(batch[0]), Variable(batch[1])
            gt = gt.cuda()  # [batch,8,256,256]
            gt = gt.float()
            meas = meas.cuda()  # [batch,256 256]
            meas = meas.float()
            batch_size1 = gt.shape[0]
            y = meas            # [batch,256 256]
            Phi = mask.expand([batch_size1, 8, 256, 256])
            Phi_s = mask_s.expand([batch_size1, 256, 256])

            optimizer.zero_grad()
            #time_start=time.time()
            model_out = network(y, Phi, Phi_s)
            #time_end=time.time()
            #print('time cost',time_end-time_start,'s')

            Loss = torch.sqrt(criterion(model_out[-1], gt)) + 0.5*torch.sqrt(criterion(model_out[-2], gt)) + 0.5*torch.sqrt(criterion(model_out[-3], gt))

            epoch_loss += Loss.data

            Loss.backward()
            optimizer.step()


    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))

    

def checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/S{}'.format(stage_num) + "_model_epoch_{}.pth".format(epoch)
    torch.save(network, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))



def main(learning_rate):
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    psnr_epoch = []
    psnr_max = 0

    for epoch in range(last_train + 1, last_train + max_iter + 1):
        train(epoch, learning_rate)
        pred, psnr_epoch = test(test_path1, epoch, result_path, psnr_epoch)
        psnr_mean = torch.mean(psnr_epoch[-1])
        print("Test result: {:.4f}".format(psnr_mean))
        
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 31:
                name = result_path + '/S{}'.format(stage_num) + '_pred_' + '{}_{:.4f}'.format(epoch, psnr_mean) + '.mat'
                scio.savemat(name, {'pred': pred})
                checkpoint(epoch, model_path)
        
        if (epoch % 6 == 0) and (epoch < 200):
            learning_rate = learning_rate * 0.9
            print(learning_rate)

if __name__ == '__main__':
    main(learning_rate)
