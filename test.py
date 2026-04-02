import os.path
import re
import torch
import torch.nn as nn
from utils.parser import args
from utils import logger, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataloader import load_data_LH_CDF
from tensorboardX import SummaryWriter
from torchviz import make_dot


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)
    # Define loss function
    criterion = nn.MSELoss().to(device)
    pretrained_dict = {
        'Standard': 'xxx',
    }
    # Inference mode
    Result = []
    if args.evaluate:
        for pretrained_name in pretrained_dict.keys():
            args.pretrained = os.path.join('./checkpoints', pretrained_dict[pretrained_name], 'last.pth')
            model = init_model(args)
            model.to(device)

            for scenario in ['Q1.1', 'Q1.2', 'Q1.3', 'Q1.4', 'Q2.1', 'Q2.2', 'Q2.3', 'Q2.4', 'Q3.1', 'Q3.2', 'Q3.3',
                             'Q3.4',
                             'Q4.1', 'Q4.2', 'Q4.3', 'Q4.4', 'Q5.1', 'Q5.2', 'Q5.3', 'Q5.4', 'Q6.1', 'Q6.2', 'Q6.3',
                             'Q6.4',
                             'Q7.1', 'Q7.2', 'Q7.3', 'Q7.4', 'Q8.1', 'Q8.2', 'Q8.3', 'Q8.4']:  # test dataset

                args.scenario = scenario
                args.NUM_UE_MAX = 1
                args.NUM_UE_MIN = 1
                args.SNR = 20
                train_loader, val_loader, test_loader = load_data_LH_CDF(args)
                cr_range = [32]
                for cr in cr_range:
                    model.reduction = cr
                    for num_bit in [5]:
                        print(f'Test scenario: {scenario} || Test Cr: {cr} || num bit: {num_bit}\n')
                        loss, rho, nmse, se, se_max = Tester(model, device, criterion, dataset_type=args.dataset_type,
                                                             feedback_type='uniform', num_bit=num_bit)(test_loader)

                        Result.append(
                            {'model': pretrained_name, 'dataset': scenario, 'cr': cr, 'num_bit': num_bit, 'nmse': nmse,
                             'se': se, 'se_max': se_max})

        import pandas as pd
        pd.DataFrame(Result).to_csv(f'{args.model_name}_test_Q1-Q8.csv', index=False)
        print(f"Saved {args.model_name}_test_Q1-Q8.csv")

if __name__ == "__main__":
    main()
