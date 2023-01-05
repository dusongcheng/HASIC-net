import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os


class HyperDatasetValid(udata.Dataset):
    def __init__(self, mode='valid'):
        if mode != 'valid':
            raise Exception("Invalid mode!", mode)
        data_path = '../02Dataset/CleanP64S32/Valid'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain1(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = '../02Dataset/CleanP64S32/Train1'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))
        data_path1 = '../02Dataset/CleanP64S32/PreValid1'
        data_names1 = glob.glob(os.path.join(data_path1, '*.mat'))

        self.keys = data_names + data_names1
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain2(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = '../02Dataset/CleanP64S32/Train2'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))
        data_path2 = '../02Dataset/CleanP64S32/PreValid2'
        data_names2 = glob.glob(os.path.join(data_path2, '*.mat'))

        self.keys = data_names + data_names2
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain3(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = '../02Dataset/CleanP64S32/Train3'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))
        data_path3 = '../02Dataset/CleanP64S32/PreValid3'
        data_names3 = glob.glob(os.path.join(data_path3, '*.mat'))

        self.keys = data_names + data_names3
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper


class HyperDatasetTrain4(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_path = '../02Dataset/CleanP64S32/Train4'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        data_path4 = '../02Dataset/CleanP64S32/PreValid4'
        data_names4 = glob.glob(os.path.join(data_path4, '*.mat'))

        self.keys = data_names + data_names4
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper