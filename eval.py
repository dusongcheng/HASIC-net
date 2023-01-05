import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from model import DoubleFlow
from scipy.io import loadmat
import glob
from utils import mrae_loss, rmse_loss, record_loss, reconstruction_patch_image_gpu, reconstruction_whole_image_gpu, save_matv73

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
patch = 128
stride = 128
model_path = 'model.pth'
result_path = './Result'

img_path = './NTIRE2020_Validation_Clean'
ground_path = '/NTIRE2020/NTIRE2020_Validation_Spectral'
var_name = 'cube'
record_time = []
record_mrae = []
record_rmse = []

# save results
if not os.path.exists(result_path):
    os.makedirs(result_path)
loss_csv = open(os.path.join(result_path, 'loss.csv'), 'a+')
model = DoubleFlow()
save_point = torch.load(model_path)
model_param = save_point['state_dict']
model_dict = {}
for k1, k2 in zip(model.state_dict(), model_param):
    model_dict[k1] = model_param[k2]
model.load_state_dict(model_dict)
model = model.cuda()

img_path_name = glob.glob(os.path.join(img_path, '*.png'))
hs_path_name = glob.glob(os.path.join(ground_path, '*.mat'))
img_path_name.sort()
hs_path_name.sort()
for i in range(len(img_path_name)):
    # load rgb images
    rgb = cv2.imread(img_path_name[i])  # imread -> BGR model
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb) / 255.0
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()  # 1,3,482,512
    # load hyper images
    mat = loadmat(hs_path_name[i])['cube']
    ground = np.float32(np.array(mat))  # 482,512,31

    print(img_path_name[i].split('/')[-1], hs_path_name[i].split('/')[-1])

    per_time, img_res = reconstruction_patch_image_gpu(rgb, model, patch, stride)
    per_time_overlap, img_res_overlap = reconstruction_patch_image_gpu(rgb[:, :, patch//2:, patch//2:], model, patch, stride)
    img_res[patch//2:, patch//2:, :] = (img_res[patch//2:, patch//2:, :] + img_res_overlap) / 2.0
    per_time += per_time_overlap

    mrae = mrae_loss(img_res, ground)
    rmse = rmse_loss(img_res, ground)
    record_mrae.append(mrae)
    record_rmse.append(rmse)
    record_time.append(per_time)
    print(img_path_name[i].split('/')[-1], "reconstruct time: ", per_time, "mrae_loss: ", mrae, "rmse_loss: ", rmse)
    # save loss
    record_loss(loss_csv, img_path_name[i].split('/')[-1], "reconstruct time: ", per_time, "mrae_loss: ", mrae, "rmse_loss: ", rmse)

    mat_name = img_path_name[i].split('/')[-1][:-10] + '.mat'
    mat_dir = os.path.join(result_path, mat_name)

    save_matv73(mat_dir, var_name, img_res)

print("[%d] images spend all time [%f], each spend time [%f], average mrae_loss [%f], average rmse_loss [%f]"
      % (len(record_time), sum(record_time), sum(record_time)/len(record_time), sum(record_mrae)/len(record_mrae),
         sum(record_rmse)/len(record_rmse)))
record_loss(loss_csv, len(record_time), sum(record_time), sum(record_time)/len(record_time), sum(record_mrae)/len(record_mrae), sum(record_rmse)/len(record_rmse))
print(torch.__version__)





