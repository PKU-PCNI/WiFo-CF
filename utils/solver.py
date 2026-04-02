import time
import os
import torch
from collections import namedtuple
from tensorboardX import SummaryWriter
from utils import logger
from utils.statics import AverageMeter, evaluator, AverageMeter_multi_rate
import numpy as np

__all__ = ['Tester']

field = ('nmse', 'rho', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))
vision_test = SummaryWriter(log_dir="data_vision/test")
vision_best = SummaryWriter(log_dir="data_vision/best")
vision_every = SummaryWriter(log_dir="data_vision/every")


class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=20, dataset_type='quadriga'
                 , feedback_type='random', num_bit=4, is_output_mat=False):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq
        self.dataset_type = dataset_type
        self.feedback_type = feedback_type
        self.num_bit = num_bit
        if self.feedback_type == 'uniform':
            self.bit_range = [self.num_bit]
        else:
            self.bit_range = list(range(self.num_bit - 2, self.num_bit + 2 + 1))
        self.is_output_mat = is_output_mat

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            if self.is_output_mat:
                loss, rho, nmse, se, loss_list, rho_list, nmse_list, se_list, se_max, H_label, H_output = self._iteration(test_data)
            else:
                loss, rho, nmse, se, loss_list, rho_list, nmse_list, se_list, se_max = self._iteration(test_data)
        if verbose:
            print(f'\n=> Test result: \nloss: {loss:.3e}'
                  f'    rho: {rho:.3e}    NMSE: {nmse:.3e} SE: {se:.3e}\n')
            print(f'\n=> Test result [Detailed] : \nloss: {loss_list}'
                  f'    rho: {rho_list}    NMSE: {nmse_list} SE: {se_list}\n')
        if self.is_output_mat:
            return loss, rho, nmse, se, se_max, H_label, H_output
        else:
            return loss, rho, nmse, se, se_max,

    def _iteration(self, data_loader):
        r""" protected function which test the model on given data loader for one epoch.
        """
        iter_loss = AverageMeter_multi_rate('Iter loss', self.bit_range)
        iter_rho = AverageMeter_multi_rate('Iter rho', self.bit_range)
        iter_se = AverageMeter_multi_rate('Iter se', self.bit_range)
        iter_nmse = AverageMeter_multi_rate('Iter nmse', self.bit_range)
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()
        if self.is_output_mat:
            H_label = []
            H_output = []
        for batch_idx, data in enumerate(data_loader):
            (name, sparse_gt, sparse_gt_noisy, raw_gt) = data
            sparse_gt = sparse_gt[0].to(self.device)
            raw_gt = raw_gt[0].to(self.device)
            sparse_gt_noisy = sparse_gt_noisy[0].to(self.device)
            name = name[0]
            nu = sparse_gt_noisy.shape[2]
            num_bit_users = self.allocate_feedback_bits(nu, self.num_bit, mode=self.feedback_type,
                                                        max_b=self.num_bit + 2, min_b=self.num_bit - 2)
            sparse_pred, q_loss, e_loss, d_loss = self.model(sparse_gt_noisy, bit_user=num_bit_users,
                                                             dataset_type=1)
            if self.is_output_mat:
                H_label.append(sparse_gt.detach().cpu().numpy())
                H_output.append(sparse_pred.detach().cpu().numpy())

            loss_per_user = [0 for i in range(nu)]
            se_per_uesr, rho_per_user, nmse_per_user = [0 for i in range(nu)], [0 for i in range(nu)], [0 for i in range(nu)],
            for i in range(nu):
                loss_per_user[i] = self.criterion(sparse_pred[:, :, [i], ...], sparse_gt[:, :, [i], ...]).item()
                rho_per_user[i], nmse_per_user[i], se_per_uesr[i], se_max = evaluator(sparse_pred[:, :, [i], ...], sparse_gt[:, :, [i], ...],
                                                              raw_gt[:, :, [i], ...])

            # Log and visdom update
            iter_loss.update(num_bit_users, loss_per_user)
            iter_rho.update(num_bit_users, rho_per_user)
            iter_se.update(num_bit_users, se_per_uesr)
            iter_nmse.update(num_bit_users, nmse_per_user)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}]  '
                            f'loss: {iter_loss.avgs} | rho: {iter_rho.avgs} | '
                            f'NMSE: {iter_nmse.avgs} | SE: {iter_se.avgs} | time: {iter_time.avg:.3f}')

        logger.info(f'=> Test rho:{iter_rho.avgs}  NMSE: {iter_nmse.avgs}  SE:  {iter_se.avgs} / {-se_max}\n')
        if self.is_output_mat:
            H_label = np.concatenate(H_label, axis=0)
            H_output = np.concatenate(H_output, axis=0)
            return iter_loss.avg, iter_rho.avg, iter_nmse.avg, iter_se.avg, \
                   iter_loss.avgs, iter_rho.avgs, iter_nmse.avgs, iter_se.avgs, se_max.item(), \
                   H_label, H_output
        else:
            return iter_loss.avg, iter_rho.avg, iter_nmse.avg, iter_se.avg, \
                   iter_loss.avgs, iter_rho.avgs, iter_nmse.avgs, iter_se.avgs, se_max.item()

    def allocate_feedback_bits(self, nu, b, mode='uniform', min_b=None, max_b=None):
        total_bits = nu * b
        if mode == 'uniform':
            bit_allocation = torch.full((nu, 1), b, dtype=torch.int)
        elif mode == 'random':
            if min_b is None:
                min_b = max(1, b // 2)
            if max_b is None:
                max_b = min(total_bits, b * 2)

            allocation = torch.randint(min_b, max_b + 1, (nu,))
            bit_allocation = allocation.view(nu, 1)
        elif mode == 'random-group':
            bit_allocation = torch.randint(min_b, max_b + 1, (1,)).repeat(nu)
        else:
            raise ValueError("Mode must be 'uniform' or 'random'")

        return bit_allocation
