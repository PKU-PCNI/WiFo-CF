import os
import numpy as np
import scipy.io as sio
import hdf5storage
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
import matplotlib.pyplot as plt
from einops import rearrange

__all__ = ['QuadrigaDataLoader_multi', 'PreFetcher']
np.random.seed(1234)


class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


def add_noise(H, snr_dB):
    signal_power = torch.mean(H ** 2)
    snr_linear = 10 ** (snr_dB / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(H) * torch.sqrt(noise_power)
    H_noisy = H + noise
    return H_noisy


def augment_and_shuffle(data_train, data_raw=None, n_joint_user=4, factor=2):
    B, Nc, Nt, Nk = data_train.shape
    if n_joint_user == 1: factor = 1
    data_train = data_train.unsqueeze(2).repeat(1, 1, n_joint_user * factor, 1, 1)
    shuffle_indices = torch.stack([torch.randperm(B) for _ in range(n_joint_user * factor)], dim=0)  # [n_joint_user, B]
    data_train = data_train[shuffle_indices, :, torch.arange(n_joint_user * factor).view(-1, 1), :, :]

    data_train = rearrange(data_train, '(u f) b o n k -> (b f) o u n k', f=factor)

    if data_raw is not None:
        data_raw = data_raw.unsqueeze(2).repeat(1, 1, n_joint_user * factor, 1, 1)
        data_raw = data_raw[shuffle_indices, :, torch.arange(n_joint_user * factor).view(-1, 1), :, :]
        data_raw = rearrange(data_raw, '(u f) b o n k -> (b f) o u n k', f=factor)
        return data_train, data_raw
    else:
        return data_train


class QuadrigaDataLoader_single(object):
    r""" PyTorch DataLoader for LH-CDF dataloader.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory, is_norm_like_cost2100=True,
                 is_UL_instead=False, n_joint_user=1, SNR=15, augmentation_factor=2, itr=30000):
        assert os.path.isdir(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        dir_train = os.path.join(root, f"X_DL_train.mat")
        dir_val = os.path.join(root, f"X_DL_val.mat")
        dir_test = os.path.join(root, f"X_DL_test.mat")

        # Training data loading
        data_train = hdf5storage.loadmat(dir_train)['X_DL_train']  # Nu, Nt, Nc
        data_train = np.stack((np.real(data_train), np.imag(data_train)), axis=1)  # Nu, 2, Nt, Nc
        data_train = torch.tensor(data_train, dtype=torch.float32)[:itr, ...]  # Nu, 2, Nt, Nc
        data_train = augment_and_shuffle(data_train, n_joint_user=n_joint_user, factor=augmentation_factor)
        data_train_noisy = add_noise(data_train, SNR)
        self.train_dataset = TensorDataset(data_train, data_train_noisy)

        # Validation data loading
        data_val = hdf5storage.loadmat(dir_val)['X_DL_val']
        data_val = np.stack((np.real(data_val), np.imag(data_val)), axis=1)  # Nu, 2, Nt, Nc
        data_val = torch.tensor(data_val, dtype=torch.float32)  # Nu, 2, Nt, Nc
        data_val = augment_and_shuffle(data_val, n_joint_user=n_joint_user, factor=augmentation_factor)
        data_val_noisy = add_noise(data_val, SNR)
        self.val_dataset = TensorDataset(data_val, data_val_noisy)

        # Test data loading, including the sparse data and the raw data
        data_test = hdf5storage.loadmat(dir_test)['X_DL_test']
        data_test = np.stack((np.real(data_test), np.imag(data_test)), axis=1)  # Nu, 2, Nt, Nc
        data_test = torch.tensor(data_test, dtype=torch.float32)  # Nu, 2, Nt, Nc

        data_test = augment_and_shuffle(data_test, n_joint_user=n_joint_user, factor=1)
        data_test_noisy = add_noise(data_test, SNR)
        self.test_dataset = TensorDataset(data_test, data_test_noisy, data_test)

        print(f"Loading from {root}")
        print(f'X_train size: {data_train.shape} || X_val size: {data_val.shape} '
              f'X_test size: {data_test.shape} ')

    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)
            val_loader = PreFetcher(val_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, val_loader, test_loader


def QuadrigaDataLoader_single2multi(root, scenario, batch_size, num_workers, pin_memory, SNR=15, NUM_UE_MIN=3,
                                    NUM_UE_MAX=6,
                                    is_UL_instead=True, augmentation_factor=2, itr=1):
    assert os.path.isdir(root)

    train_data = []
    val_data = []
    test_data = []
    for dataset_name in scenario.split('*'):
        root_single = os.path.join(root, dataset_name)
        n_joint_user = np.random.randint(NUM_UE_MIN, NUM_UE_MAX + 1)  # [ )
        train_loader, val_loader, test_loader = QuadrigaDataLoader_single(
            root_single, batch_size, num_workers, pin_memory, is_UL_instead=is_UL_instead, n_joint_user=n_joint_user,
            SNR=SNR, augmentation_factor=augmentation_factor, itr=itr
        )()
        print(dataset_name, n_joint_user)
        for [data, data_noisy] in train_loader:
            train_data.append((dataset_name, data, data_noisy))
        for [data, data_noisy] in val_loader:
            val_data.append((dataset_name, data, data_noisy))
        for [data, data_noisy, raw_data] in test_loader:
            test_data.append((dataset_name, data, data_noisy, raw_data))
    return train_data, val_data, test_data


class QuadrigaDataLoader_multi(data.Dataset):
    def __init__(self, data):
        super(QuadrigaDataLoader_multi, self).__init__()
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]  # (dataset_name, data)

    def __len__(self):
        return self.length


def load_data_LH_CDF(args):
    pin_memory = False
    if not hasattr(args, 'itr'):
        args.itr = 10000000
    train_data, val_data, test_data = QuadrigaDataLoader_single2multi(args.data_dir, args.scenario, args.batch_size,
                                                                      args.workers, pin_memory=pin_memory,
                                                                      NUM_UE_MIN=args.NUM_UE_MIN,
                                                                      NUM_UE_MAX=args.NUM_UE_MAX,
                                                                      is_UL_instead=args.is_UL_instead,
                                                                      SNR=args.SNR,
                                                                      augmentation_factor=args.augmentation_factor,
                                                                      itr=args.itr)

    train_dataset = QuadrigaDataLoader_multi(train_data)
    train_data_loader = DataLoader(train_dataset, batch_size=1,
                                   num_workers=args.workers,
                                   pin_memory=pin_memory,
                                   shuffle=True)
    val_dataset = QuadrigaDataLoader_multi(val_data)
    val_data_loader = DataLoader(val_dataset, batch_size=1,
                                 num_workers=args.workers,
                                 pin_memory=pin_memory,
                                 shuffle=True)
    test_dataset = QuadrigaDataLoader_multi(test_data)
    test_data_loader = DataLoader(test_dataset, batch_size=1,
                                  num_workers=args.workers,
                                  pin_memory=pin_memory,
                                  shuffle=True)

    return train_data_loader, val_data_loader, test_data_loader


if __name__ == '__main__':
    class EmptyClass:
        pass


    args = EmptyClass()
    args.data_dir = '/data1/PCNI1_data/FMMF/dataset/Mixed_Dataset/'
    args.scenario = 'Q1*Q3*Q5*C1*Boston5G_28'
    args.batch_size = 64
    args.workers = 0
    args.pin_memory = 0
    args.NUM_UE_MIN = 2
    args.NUM_UE_MAX = 2
    args.is_UL_instead = True
    args.SNR = 10
    args.augmentation_factor = 2

    train_data_loader, val_data_loader, test_data_loader = load_data(args)

    for (name, data, data_noisy) in train_data_loader:
        print(name, data[0].shape, data_noisy[0].shape)

    for (name, data, data_noisy, data_raw) in test_data_loader:
        print(name, data[0].shape, data_raw[0].shape, data_noisy[0].shape)
