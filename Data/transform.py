from logging import getLogger
import random
import torch
import torch.nn as nn
from recbole.data.interaction import Interaction

def construct_transform(config):
    if config['transform'] is None:
        logger = getLogger()
        logger.warning('Equal transform')
        return Equal(config)
    else:
        str2transform = {
            'plm_emb': PLMEmb,
            'plm_img_emb': PLMImgEmb
        }
        return str2transform[config['transform']](config)

class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataloader, interaction):
        return interaction

class PLMEmb:
    def __init__(self, config):
        self.logger = getLogger()
        self.logger.info('PLM Embedding Transform in DataLoader.')
        self.item_drop_ratio = config['item_drop_ratio']
        self.item_drop_coefficient = config['item_drop_coefficient']

    def __call__(self, dataloader, interaction):
        '''Sequence augmentation and PLM embedding fetching
        '''
        item_seq_len = interaction['item_length']
        item_seq = interaction['item_id_list']

        plm_embedding = dataloader.dataset.plm_embedding
        item_emb_seq = plm_embedding(item_seq)
        pos_item_id = interaction['item_id']
        pos_item_emb = plm_embedding(pos_item_id)

        mask_p = torch.full_like(item_seq, 1 - self.item_drop_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)

        # Augmentation(物品丢弃或者词丢弃达到增强数据的目的，防止过拟合)
        rd = random.random()
        if rd < self.item_drop_coefficient:
            # Item drop
            seq_mask = item_seq.eq(0).to(torch.bool)
            mask = torch.logical_or(mask, seq_mask)
            mask[:, 0] = True
            drop_index = torch.cumsum(mask, dim=1) - 1

            item_seq_aug = torch.zeros_like(item_seq).scatter(dim=-1, index=drop_index, src=item_seq)
            item_seq_len_aug = torch.gather(drop_index, 1, (item_seq_len - 1).unsqueeze(1)).squeeze() + 1
            item_emb_seq_aug = plm_embedding(item_seq_aug)
        else:
            # Word drop
            plm_embedding_aug = dataloader.dataset.plm_embedding_aug
            full_item_emb_seq_aug = plm_embedding_aug(item_seq)

            item_seq_aug = item_seq
            item_seq_len_aug = item_seq_len
            item_emb_seq_aug = torch.where(mask.unsqueeze(-1), item_emb_seq, full_item_emb_seq_aug)

        interaction.update(Interaction({
            'item_emb_list': item_emb_seq,
            'pos_item_emb': pos_item_emb,
            'item_id_list_aug': item_seq_aug,
            'item_length_aug': item_seq_len_aug,
            'item_emb_list_aug': item_emb_seq_aug,
        }))

        return interaction

class PLMImgEmb:
    def __init__(self, config):
        self.logger = getLogger()
        self.logger.info('PLM Embedding Transform in DataLoader.')
        self.item_drop_ratio = config['item_drop_ratio']
        self.item_drop_coefficient = config['item_drop_coefficient']


    def __call__(self, dataloader, interaction):
        '''Sequence augmentation and PLM embedding fetching
        '''
        item_seq_len = interaction['item_length']
        item_seq = interaction['item_id_list']

        plm_embedding = dataloader.dataset.plm_embedding
        img_embedding = dataloader.dataset.img_embedding
        plm_embedding_empty_mask = dataloader.dataset.plm_embedding_empty_mask
        
        img_embedding_empty_mask = dataloader.dataset.img_embedding_empty_mask
       
        

        text_emb_seq = plm_embedding(item_seq)
        img_emb_seq = img_embedding(item_seq)
        text_emb_empty_mask_seq = plm_embedding_empty_mask(item_seq)
        img_emb_empty_mask_seq = img_embedding_empty_mask(item_seq)
        
        
        pos_item_id = interaction['item_id']
        pos_text_emb = plm_embedding(pos_item_id)
        pos_img_emb = img_embedding(pos_item_id)

        mask_p = torch.full_like(item_seq, 1 - self.item_drop_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)

        # Augmentation
        rd = random.random()
        if rd < self.item_drop_coefficient:
            # Item drop
            seq_mask = item_seq.eq(0).to(torch.bool)
            mask = torch.logical_or(mask, seq_mask)
            mask[:, 0] = True
            drop_index = torch.cumsum(mask, dim=1) - 1

            item_seq_aug = torch.zeros_like(item_seq).scatter(dim=-1, index=drop_index, src=item_seq)
            item_seq_len_aug = torch.gather(drop_index, 1, (item_seq_len - 1).unsqueeze(1)).squeeze() + 1
            text_emb_seq_aug = plm_embedding(item_seq_aug)
            text_emb_empty_mask_seq_aug = plm_embedding_empty_mask(item_seq_aug)

        else:
            # Word drop
            plm_embedding_aug = dataloader.dataset.plm_embedding_aug
            full_text_emb_seq_aug = plm_embedding_aug(item_seq)

            item_seq_aug = item_seq
            item_seq_len_aug = item_seq_len
            text_emb_seq_aug = torch.where(mask.unsqueeze(-1), text_emb_seq, full_text_emb_seq_aug)
            text_emb_empty_mask_seq_aug = text_emb_empty_mask_seq
           

        interaction.update(Interaction({
            'text_emb_list': text_emb_seq,
            'text_emb_empty_mask_list': text_emb_empty_mask_seq,
            # 'text_interest_list': text_interest_seq,
            'img_emb_list': img_emb_seq,
            'img_emb_empty_mask_list': img_emb_empty_mask_seq,
            # 'img_interest_list': img_interest_seq,
            'pos_text_emb': pos_text_emb,
            'pos_img_emb': pos_img_emb,
            'item_id_list_aug': item_seq_aug,
            'item_length_aug': item_seq_len_aug,
            'text_emb_list_aug': text_emb_seq_aug,
            'text_emb_empty_mask_list_aug': text_emb_empty_mask_seq_aug,        
        }))

        return interaction
