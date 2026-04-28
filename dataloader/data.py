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


def augment_and_shuffle(data_input, data_raw=None, n_joint_user=4, factor=2):
    B, Nc, Nt, Nk = data_input.shape
    if n_joint_user == 1: factor = 1
    data_input = data_input.unsqueeze(2).repeat(1, 1, n_joint_user * factor, 1, 1)
    shuffle_indices = torch.stack([torch.randperm(B) for _ in range(n_joint_user * factor)], dim=0)  # [n_joint_user, B]
    data_input = data_input[shuffle_indices, :, torch.arange(n_joint_user * factor).view(-1, 1), :, :]

    data_input = rearrange(data_input, '(u f) b o n k -> (b f) o u n k', f=factor)

    if data_raw is not None:
        data_raw = data_raw.unsqueeze(2).repeat(1, 1, n_joint_user * factor, 1, 1)
        data_raw = data_raw[shuffle_indices, :, torch.arange(n_joint_user * factor).view(-1, 1), :, :]
        data_raw = rearrange(data_raw, '(u f) b o n k -> (b f) o u n k', f=factor)
        return data_input, data_raw
    else:
        return data_input
 
class QuadrigaDataLoader_single_test(object):
    r""" PyTorch DataLoader only for test data.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory,
                 is_UL_instead=False, n_joint_user=1, SNR=15):
        assert os.path.isdir(root)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dir_test = os.path.join(root, "X_DL_test.mat")

        # Only load test data
        data_test = hdf5storage.loadmat(dir_test)['X_DL_test']
        data_test = np.stack((np.real(data_test), np.imag(data_test)), axis=1)
        data_test = torch.tensor(data_test, dtype=torch.float32)

        # Test data uses factor=1
        data_test = augment_and_shuffle(
            data_test,
            n_joint_user=n_joint_user,
            factor=1
        )

        data_test_noisy = add_noise(data_test, SNR)

        # Keep the same output format as before:
        # data_test, noisy data, raw clean data
        self.test_dataset = TensorDataset(
            data_test,
            data_test_noisy,
            data_test
        )

        print(f"Loading test data from {root}")
        print(f"X_test size: {data_test.shape}")

    def __call__(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

        if self.pin_memory is True and torch.cuda.is_available():
            test_loader = PreFetcher(test_loader)

        return test_loader

def QuadrigaDataLoader_single2multi_test(root, scenario, batch_size, num_workers,
                                         pin_memory, SNR=15,
                                         NUM_UE_MIN=3, NUM_UE_MAX=6,
                                         is_UL_instead=True):
    assert os.path.isdir(root)

    test_data = []

    for dataset_name in scenario.split('*'):
        root_single = os.path.join(root, dataset_name)

        n_joint_user = np.random.randint(NUM_UE_MIN, NUM_UE_MAX + 1)

        test_loader = QuadrigaDataLoader_single_test(
            root_single,
            batch_size,
            num_workers,
            pin_memory,
            is_UL_instead=is_UL_instead,
            n_joint_user=n_joint_user,
            SNR=SNR
        )()

        print(dataset_name, n_joint_user)

        for data, data_noisy, raw_data in test_loader:
            test_data.append((dataset_name, data, data_noisy, raw_data))

    return test_data
 
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

    test_data = QuadrigaDataLoader_single2multi_test(
        args.data_dir,
        args.scenario,
        args.batch_size,
        args.workers,
        pin_memory=pin_memory,
        NUM_UE_MIN=args.NUM_UE_MIN,
        NUM_UE_MAX=args.NUM_UE_MAX,
        is_UL_instead=args.is_UL_instead,
        SNR=args.SNR
    )

    test_dataset = QuadrigaDataLoader_multi(test_data)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return test_data_loader

