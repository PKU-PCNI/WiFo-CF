import argparse

parser = argparse.ArgumentParser(description='CRNet PyTorch Training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--data-dir', type=str, required=True,
                    help='the path of dataloader.')
parser.add_argument('--scenario', type=str, required=True,
                    help="the channel scenario")
parser.add_argument('-b', '--batch-size', type=int, required=True, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-init', type=float, required=True,
                    help='learning rate')
parser.add_argument('-j', '--workers', type=int, metavar='N', required=True,
                    help='number of data loading workers')
parser.add_argument('--save-path', type=str, required=True,
                    help='The save path of checkpoints')
parser.add_argument('--num-encoder-layers', type=int, required=True, metavar='N',
                    help='num_encoder_layers')
parser.add_argument('--num-decoder-layers', type=int, required=True, metavar='N',
                    help='num_decoder_layers')

# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')

parser.add_argument('--dim_patch', type=int, default=64, metavar= 'N', help= 'number of Transformer feature dimension.' )

parser.add_argument('-d', '--d_model', type=int, default=64, metavar= 'N', help= 'number of Transformer feature dimension.' )

parser.add_argument('--d_model_decoder', type=int, default=64, metavar= 'N', help= 'number of Transformer feature dimension.' )

parser.add_argument('--d_ff', type=int, default=128, metavar='N', help= 'number of Transformer feature dimension for feedback.' )
parser.add_argument('--d_ff_d', type=int, default=128, metavar='N', help= 'number of Transformer feature dimension for feedback.' )
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')
parser.add_argument('--save_name', type=str, default='null',
                    help='name of test result')
parser.add_argument('--is_UL_instead', default=True, type=bool,
                    help='Train use UL data ? ')
parser.add_argument('--NUM_UE_MIN', default=3, type=int,
                    help='Num of UE (min) ')
parser.add_argument('--NUM_UE_MAX', default=6, type=int,
                    help='Num of UE (max) ')
parser.add_argument('--is_single_user', default=False, type=bool,
                    help='whether or not to train with single users data ')
parser.add_argument('--transnet', default=1, type=int,
                    help='type of transnet')
parser.add_argument('--n_routed_experts', default=16, type=int,
                    help='num of expert in moe')
parser.add_argument('--n_activated_experts', default=2, type=int,
                    help='topk')
parser.add_argument('--n_shared_expert', default=1, type=int,
                    help='n_shared_expert')
parser.add_argument('--SNR', default=15, type=float,
                    help='snr')
parser.add_argument('--feedback_type', type=str, required=True, choices=["uniform", "random", "random-group"],
                    help='feedback_type')
parser.add_argument('--num_bit', default=4, type=int,
                    help='num_bit(avg)')
parser.add_argument('--augmentation_factor', default=2, type=int,
                    help='augmentation_factor')
parser.add_argument('--model_name', type=str, default='small',
                    help='model_name')
parser.add_argument('--if_moe_activate', default=True, type=bool,
                    help='whether or not to use deepseek moe')
parser.add_argument('--beta1', type=float, default=0.1,
                    help='beta1')
parser.add_argument('--beta2', type=float, default=0.01,
                    help='beta2')
parser.add_argument('--beta3', type=float, default=0.01,
                    help='beta3')
args = parser.parse_args()
