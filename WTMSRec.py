import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import  WaveletMMEncoder,LearnableFilter


class WTMSRec(WaveletMMEncoder):
    def __init__(self,config,dataset):
        super().__init__(config,dataset)
        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.modal_type = config['modal_type']
        self.id_type = config['id_type']
        self.beta = config['beta']
        self.have_T = config['have_T']
        self.t_start = config['t_start']
        self.t_end = config['t_end']
        self.filter = LearnableFilter(config,self.max_seq_length)

        self.mse_fct = nn.MSELoss()

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None

        if 'text' in self.modal_type:
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
            self.register_buffer('plm_embedding_empty_mask', (~self.plm_embedding.weight.data.sum(-1).bool()))
            self.text_adaptor = nn.Linear(config['plm_size'], config['hidden_size'])  # 可删

        if 'img' in self.modal_type:
            self.img_embedding = copy.deepcopy(dataset.img_embedding)
            self.register_buffer('img_embedding_empty_mask', (~self.img_embedding.weight.data.sum(-1).bool()))
            self.img_adaptor = nn.Linear(config['img_size'], config['hidden_size'])  # 可删

    # def make_graph_data(self,fields):
    #     # 提取用户数量和物品数量
    #     user_num = fields['user_num']
    #     item_num = fields['item_num']
    #
    #     # 调用 inter_matrix 函数获取用户-物品交互系数矩阵
    #     inter_matrix_func = fields['inter_matrix']
    #     sparse_matrix = inter_matrix_func(form='coo')
    #
    #     # 构建用户-物品交互图
    #     graph = np.vstack((sparse_matrix.row, sparse_matrix.col)).T
    #
    #     # 构建用户-物品交互字典
    #     user_item_dict = {}
    #     for user, item in zip(sparse_matrix.row, sparse_matrix.col):
    #         if user not in user_item_dict:
    #             user_item_dict[user] = []
    #         user_item_dict[user].append(item)

        # # 增加用户数量偏移
        # train_edge[:, 1] += user_num
        # user_item_dict = {i: [j + user_num for j in user_item_dict[i]] for i in user_item_dict.keys()}
        #
        # return user_num, item_num, graph, user_item_dict
    def get_attention_mask(self, input_tensor, is_causal=True):
        B, L, D = input_tensor.shape
        device = input_tensor.device

        # 计算填充掩码，并将其转换到 input_tensor 的设备
        padding_mask = (input_tensor.sum(-1) == 0).to(device)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

        # 计算因果掩码，并将其转换到 input_tensor 的设备
        if is_causal:
            causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)  # [L, L]
        else:
            causal_mask = None

        # 将因果掩码扩展到合适的形状
        if causal_mask is not None:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, L, L]

        # 结合填充掩码和因果掩码
        if causal_mask is not None:
            attention_mask = padding_mask + causal_mask  # [B, 1, L, L]
        else:
            attention_mask = padding_mask  # [B, 1, 1, L]

        return attention_mask

    def get_encoder_attention_mask(self, dec_input_seq=None, is_casual=True):
        """memory_mask: [BxL], dec_input_seq: [BxNq]"""
        key_padding_mask = (dec_input_seq == 0)  # binary, [BxNq], Nq=L
        dec_seq_len = dec_input_seq.size(-1)
        attn_mask = torch.triu(torch.full((dec_seq_len, dec_seq_len), float('-inf'), device=dec_input_seq.device),
                               diagonal=1) if is_casual else None
        return attn_mask, key_padding_mask

    def get_decoder_attention_mask(self, enc_input_seq, item_modal_empty_mask, is_casual=True):
        # enc_input_seq: [BxL]j
        # item_modal_empty_mask: [BxMxL]
        assert enc_input_seq.size(0) == item_modal_empty_mask.size(0)
        assert enc_input_seq.size(-1) == item_modal_empty_mask.size(-1)
        batch_size, num_modality, seq_len = item_modal_empty_mask.shape  # M
        if self.seq_mm_fusion == 'add':
            key_padding_mask = (enc_input_seq == 0)  # binary, [BxL]
        else:
            # binary, [Bx1xL] | [BxMxL] => [BxMxL]
            key_padding_mask = torch.logical_or((enc_input_seq == 0).unsqueeze(1), item_modal_empty_mask)
            key_padding_mask = key_padding_mask.flatten(1)  # [BxMxL] => [Bx(M*L)]
        if is_casual:
            attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=enc_input_seq.device),
                                   diagonal=1)  # [LxL]
            if self.seq_mm_fusion != 'add':
                attn_mask = torch.tile(attn_mask, (num_modality, num_modality))  # [(M*L)x(M*L)]
        else:
            attn_mask = None
        cross_attn_mask = None  # Full mask
        return attn_mask, cross_attn_mask, key_padding_mask

    def forward(self, item_seq, item_seq_len,text_emb,img_emb):


        # encoder input
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_embedding = self.position_embedding(position_ids)  # [LxD]
        text_emb = text_emb + position_embedding  # [BxMxLxD] or [BxLxD]

        # img_emb = img_emb + position_embedding


        text_emb = text_emb.view(text_emb.size(0), -1,
                                           text_emb.size(-1))  # [BxMxLxD] => [Bx(M*L)xD]

        img_emb = img_emb.view(img_emb.size(0), -1, img_emb.size(-1))  # [BxMxLxD] => [Bx(M*L)xD]

        # text_mask = self.get_attention_mask(text_emb,is_causal=True)
        # img_mask = self.get_attention_mask(img_emb,is_causal=True)

        if self.train_stage == 'transductive_ft':
            if self.id_type != 'none':
                item_id_embeddings = self.item_embedding(item_seq)
                img_emb = img_emb + item_id_embeddings
                text_emb = text_emb + item_id_embeddings

        multi_feature = self.filter(text_emb, img_emb)
        multi_mask = self.get_attention_mask(multi_feature)

        mul = self.encoder[0](multi_feature,multi_mask)
        for i in range(1,self.layers):
            mul = self.encoder[i](mul,multi_mask)



        mul = self.gather_indexes(mul,item_seq_len-1)
        return mul

    def _compute_dynamic_fused_logits(self, seq_output, text_emb, img_emb):
        text_emb = F.normalize(text_emb, dim=1)
        img_emb = F.normalize(img_emb, dim=1)
        text_logits = torch.matmul(seq_output, text_emb.transpose(0, 1)) # [BxB]


        img_logits = torch.matmul(seq_output, img_emb.transpose(0, 1)) # [BxB]

        modality_logits = torch.stack([text_logits, img_logits], dim=-1) # [BxBx2]
        # if self.item_mm_fusion in ['dynamic_shared', 'dynamic_instance']:
        #     agg_logits = (modality_logits * F.softmax(modality_logits * self.fusion_factor.unsqueeze(-1), dim=-1)).sum(dim=-1) # [BxBx2] => [BxB]
        # else: # 'static'
        agg_logits = modality_logits.mean(dim=-1) # [BxBx2] => [BxB]

        if self.train_stage == 'transductive_ft':
            if self.id_type != 'none':
                test_id_emb = F.normalize(self.item_embedding.weight, dim=1)
                id_logits = torch.matmul(seq_output, test_id_emb.transpose(0, 1))
                agg_logits = (id_logits + agg_logits * 2) / 3
        return agg_logits

    def seq_item_contrastive_task(self, seq_output, interaction, batch_labels,temp,criterion_cross):


        if 'text' in self.modal_type:
            pos_text_emb = self.text_adaptor(interaction['pos_text_emb'])

        if 'img' in self.modal_type:
            pos_img_emb = self.img_adaptor(interaction['pos_img_emb'])

        if 'text' in self.modal_type and 'img' in self.modal_type: # weighted fusion
            logits = self._compute_dynamic_fused_logits(seq_output, pos_text_emb, pos_img_emb)

        else: # single modality or no modality
            if 'text' in self.modal_type:
                pos_item_emb = pos_text_emb
            if 'img' in self.modal_type:
                pos_item_emb = pos_img_emb
            pos_item_emb = F.normalize(pos_item_emb, dim=1)
            logits = torch.matmul(seq_output, pos_item_emb.transpose(0, 1))
        loss = criterion_cross(logits, batch_labels,temp)
        return loss

    def _compute_seq_embeddings_pretrain(
            self, item_seq, item_seq_len,
            text_emb, img_emb,
        ):

        seq_output= self.forward(
            item_seq=item_seq, item_seq_len=item_seq_len,text_emb=text_emb,img_emb=img_emb
        )
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output

    def _compute_seq_embeddings(self, item_seq, item_seq_len):
        if 'text' in self.modal_type:
            text_emb = self.text_adaptor(self.plm_embedding(item_seq))

        if 'img' in self.modal_type:
            img_emb = self.img_adaptor(self.img_embedding(item_seq))


        seq_output  = self.forward(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            text_emb=text_emb,
            img_emb=img_emb,
        )
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output


    def pretrain(self, interaction,temp,criterion_cross):
        img_emb=self.img_adaptor(interaction['img_emb_list'])
        seq_output = self.forward(
            item_seq=interaction[self.ITEM_SEQ],
            item_seq_len=interaction[self.ITEM_SEQ_LEN],
            text_emb=self.text_adaptor(interaction['text_emb_list']),
            img_emb=img_emb,
        )
        batch_size = seq_output.shape[0]
        device = seq_output.device
        batch_labels = torch.arange(batch_size, device=device, dtype=torch.long)


        # mse_loss = self.mse_fct(decoder_0,filter_output)
        loss_seq_item = self.seq_item_contrastive_task(seq_output, interaction, batch_labels,temp,criterion_cross)
        torch.cuda.empty_cache()
        loss_seq_seq  = self.seq_seq_contrastive_task(
            seq_output, interaction, img_emb, batch_labels,temp,criterion_cross)
        loss = loss_seq_item + self.lam * loss_seq_seq   #+self.beta * mse_loss
        return loss

    def _compute_test_item_embeddings(self):
        test_item_emb = 0
        if 'text' in self.modal_type:
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_item_emb = test_item_emb + test_text_emb
        if 'img' in self.modal_type:
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            test_item_emb = test_item_emb + test_img_emb

        if self.train_stage == 'transductive_ft':
            if self.id_type != 'none':
                test_item_emb = test_item_emb + self.item_embedding.weight

        test_item_emb = F.normalize(test_item_emb, dim=1)
        return test_item_emb

    def seq_seq_contrastive_task(self, seq_output, interaction, img_emb, batch_labels,temp,criterion_cross):
        seq_output_aug = self._compute_seq_embeddings_pretrain(
            item_seq=interaction[self.ITEM_SEQ + '_aug'],
            item_seq_len=interaction[self.ITEM_SEQ_LEN + '_aug'],
            text_emb=self.text_adaptor(interaction['text_emb_list_aug']),
            img_emb=img_emb,
        )
        logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1))
        loss = criterion_cross(logits, batch_labels,temp)
        return loss

    def calculate_loss(self,interaction,decay_value,T_mlp,criterion_cross):
        if self.have_T:
            temp = T_mlp(decay_value)
            temp = self.t_start + self.t_end * torch.sigmoid(temp)
            T_mlp.train()
        else:
            temp = (self.temperature * torch.ones(1))

        if self.train_stage == 'pretrain':
            return self.pretrain(interaction,temp,criterion_cross)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self._compute_seq_embeddings(item_seq, item_seq_len)
        if 'text' in self.modal_type and 'img' in self.modal_type: # weighted fusion
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            logits = self._compute_dynamic_fused_logits(seq_output, test_text_emb, test_img_emb)
        else: # single modality or no modality
            test_item_emb = self._compute_test_item_embeddings()
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        pos_items = interaction[self.POS_ITEM_ID]
        loss = criterion_cross(logits, pos_items,temp)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self._compute_seq_embeddings(item_seq, item_seq_len)
        if 'text' in self.modal_type and 'img' in self.modal_type:  # weighted fusion
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            scores = self._compute_dynamic_fused_logits(seq_output, test_text_emb, test_img_emb) / self.temperature
        else:  # single modality or no modality
            test_item_emb = self._compute_test_item_embeddings()
            scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        return scores







