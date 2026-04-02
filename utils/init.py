import os
import random
import thop
import torch

from models.WiFo_CF import WiFo_CF
from utils import logger, line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading
    model = WiFo_CF(args)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained, weights_only=False,
                                map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict, strict=False)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    H_a = torch.randn([1, 2, 8, 32, 32])
    flops, params = thop.profile(model, inputs=(H_a,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    total_params_encoder = sum(
        p.numel() for name, module in model.named_children() if name in model.encoder_names for p in
        module.parameters())
    total_params_decoder = sum(
        p.numel() for name, module in model.named_children() if name in model.decoder_names for p in
        module.parameters())
    print(total_params_encoder / 1e6, total_params_decoder / 1e6,
          total_params_encoder / 1e6 + total_params_decoder / 1e6)

    # Model info logging
    logger.info(f'=> Model Name: TransNet [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n')
    # logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
