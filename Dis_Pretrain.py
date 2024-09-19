# import os
# import argparse
# import numpy as np
# import torch
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from recbole.utils import init_seed, init_logger
# from logging import getLogger
#
#
# from config import Config
# from fms import WTMSRec
# from Data.dataset import PretrainWTMSRecDataset
# from Data.dataloader import CustomizedTrainDataLoader
# from trainer import DisFMSPreTrainer
#
#
# def pretrain(GPU_ranking,world_size,dataset,**kwargs):
#     # configurations initialization
#     props = ['props/WTMSRec.yaml', 'props/pretrain.yaml']
#     if GPU_ranking == 0:
#         print('Distributed Pre-training on:', dataset)
#         print(props)
#
#     # configurations initialization
#     kwargs.update({'ddp': True, 'rank': GPU_ranking, 'world_size': world_size})
#     config = Config(model=WTMSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
#
#     init_seed(config['seed'], config['reproducibility'])
#     # logger initialization
#     if config['rank'] not in [-1, 0]:
#         config['state'] = 'warning'
#     init_logger(config)
#     logger = getLogger()
#     logger.info(config)
#
#     # dataset filtering
#     dataset = PretrainWTMSRecDataset(config)
#     logger.info(dataset)
#
#     pretrain_dataset = dataset.build()[0]
#     pretrain_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)
#
#     # model loading and initialization
#     model = WTMSRec(config, pretrain_data.dataset)
#     logger.info(model)
#
#     # trainer loading and initialization
#     trainer = DisFMSPreTrainer(config, model)
#
#     # model pre-training
#     trainer.pretrain(config,pretrain_data, show_progress=(GPU_ranking == 0))
#
#     dist.destroy_process_group()
#
#     return config['model'], config['dataset']
#
#
# if __name__=='__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', type=str, default='FHCKM_mm_full', help='dataset name')
#     parser.add_argument('-p', type=str, default='12355', help='port for distributed training')
#     args, unparsed = parser.parse_known_args()
#
#     # At least two Gpus are required
#     n_gpus = torch.cuda.device_count()
#     assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}."
#     world_size = n_gpus
#
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = args.p
#
#     mp.spawn(pretrain,
#              args=(world_size, args.d,),
#              nprocs=world_size,
#              join=True)
#
import os
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from recbole.utils import init_seed, init_logger
from logging import getLogger


from config import Config
from WTMSRec import WTMSRec
from Data.dataset import PretrainWTMSRecDataset
from Data.dataloader import CustomizedTrainDataLoader
from trainer import DisFMSPreTrainer


def pretrain(GPU_ranking,world_size,dataset,**kwargs):
    # configurations initialization
    props = ['props/WTMSRec.yaml', 'props/pretrain.yaml']
    if GPU_ranking == 0:
        print('Distributed Pre-training on:', dataset)
        print(props)

    # configurations initialization
    kwargs.update({'ddp': True, 'rank': GPU_ranking, 'world_size': world_size})
    config = Config(model=WTMSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    if config['rank'] not in [-1, 0]:
        config['state'] = 'warning'
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = PretrainWTMSRecDataset(config)
    logger.info(dataset)

    pretrain_dataset = dataset.build()[0]
    pretrain_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)

    # model loading and initialization
    model = WTMSRec(config, pretrain_data.dataset)
    logger.info(model)

    # trainer loading and initialization
    trainer = DisFMSPreTrainer(config, model)

    # model pre-training
    trainer.pretrain(config,pretrain_data, show_progress=(GPU_ranking == 0))

    dist.destroy_process_group()

    return config['model'], config['dataset']


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='FHCKM_mm_full', help='dataset name')
    parser.add_argument('-p', type=str, default='12355', help='port for distributed training')
    args, unparsed = parser.parse_known_args()

    # At least two Gpus are required
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}."
    world_size = n_gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.p

    mp.spawn(pretrain,
             args=(world_size, args.d,),
             nprocs=world_size,
             join=True)

