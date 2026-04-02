import torch
from einops import rearrange
from utils.metrics import SE_Loss
__all__ = ['AverageMeter', 'evaluator']


class AverageMeter_multi_rate(object):
    def __init__(self, name, bit_range):
        self.reset()
        self.nums = len(bit_range)
        self.bit_range = bit_range
        self.AverageMeters = [AverageMeter(f"{name} with {bit} bit") for bit in bit_range]
        self.name = name
        self.avgs = [0 for bit in bit_range]
        self.sums = [0 for bit in bit_range]
        self.counts = [0 for bit in bit_range]
        self.avg = 0

    def reset(self):
        self.nums = 0
        self.avg = 0
        self.AverageMeters = []
        self.avgs = []
        self.sums = []
        self.counts = []

    def fresh(self):
        for ob, i in zip(self.AverageMeters, range(self.nums)):
            self.avgs[i] = ob.avg.item() if isinstance(ob.avg, torch.Tensor) else ob.avg
            self.sums[i] = ob.sum.item() if isinstance(ob.sum, torch.Tensor) else ob.sum
            self.avgs[i] = round(self.avgs[i], 3)
            self.sums[i] = round(self.sums[i], 3)
            self.counts[i] = ob.count
        self.avg = sum(self.sums) / sum(self.counts)

    def update(self, indexs, vals, n=1):
        for bit, val in zip(indexs, vals):
            loc = self.bit_range.index(bit)
            self.AverageMeters[loc].update(val)
        self.fresh()

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sums}; avg={self.avgs}"

class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def evaluator(sparse_pred, sparse_gt, raw_gt, is_quadriga=True, is_norm_like_cost2100=True,):
    r""" Evaluation of decoding implemented in PyTorch Tensor
         Computes normalized mean square error (NMSE) and rho.
    """
    SE_func = SE_Loss(snr=10)
    with torch.no_grad():
        # Basic params
        nt = 32
        nc = 32
        nc_expand = 257

        if is_quadriga:
            nt, nc_expand = raw_gt.shape[-2:]
            nc = sparse_gt.shape[-1]
        # # De-centralize
        # if is_norm_like_cost2100:
        #     sparse_gt = sparse_gt + 0.5
        #     sparse_pred = sparse_pred + 0.5

        if len(sparse_pred.shape) == 5:
            sparse_pred = rearrange(sparse_pred, 'b o u n k -> (b u) o n k')
            sparse_gt = rearrange(sparse_gt, 'b o u n k -> (b u) o n k')
            raw_gt = rearrange(raw_gt, 'b o u n k -> (b u) o n k')

        # Calculate the NMSE
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        # print(((sparse_gt - sparse_pred)**2).mean(), mse.sum(dim=[1, 2]).mean(), power_gt.sum(dim=[1, 2]).mean())
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

        # Calculate the Rho
        sparse_gt = sparse_gt.permute(0, 2, 3, 1)
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)
        # raw_pred = torch.fft.fft(sparse_pred, signal_ndim=1)[:, :, :125, :]
        norm_pred = sparse_pred[..., 0] ** 2 + sparse_pred[..., 1] ** 2
        norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        norm_gt = sparse_gt[..., 0] ** 2 + sparse_gt[..., 1] ** 2
        norm_gt = torch.sqrt(norm_gt.sum(dim=1))

        real_cross = sparse_pred[..., 0] * sparse_gt[..., 0] + sparse_pred[..., 1] * sparse_gt[..., 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = sparse_pred[..., 0] * sparse_gt[..., 1] - sparse_pred[..., 1] * sparse_gt[..., 0]
        imag_cross = imag_cross.sum(dim=1)
        norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)

        rho = (norm_cross / (norm_pred * norm_gt)).mean()

        # get se
        pred_complex = torch.complex(sparse_pred[..., 0], sparse_pred[..., 1])
        pred_complex = rearrange(pred_complex, 'b (n o) k -> (b k) n o', o=1)
        gt_complex = torch.complex(sparse_gt[..., 0], sparse_gt[..., 1])
        gt_complex = rearrange(gt_complex, 'b (n o) k -> (b k) n o', o=1)
        se, se_max = SE_func(pred_complex, gt_complex)

        return rho, nmse, se, se_max
