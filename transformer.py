import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import math
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()

        hidden_size = config['c_hidden_dim']
        inner_size = 4 * config['c_hidden_dim']

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(config['hidden_act'])

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        # self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm = LayerNorm(config['c_hidden_dim'], eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout'])

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        if config['f_hidden_size'] % config['num_attention_heads'] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config['f_hidden_size'], config['num_attention_heads']))
        self.config = config
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = int(config['f_hidden_size'] / config['num_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(config['f_hidden_size'] , self.all_head_size)
        self.key = nn.Linear(config['f_hidden_size'] , self.all_head_size)
        self.value = nn.Linear(config['f_hidden_size'] , self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(config['attention_dropout'])

        self.dense = nn.Linear(config['f_hidden_size'], config['f_hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['f_hidden_size'], eps=1e-12) # TODO
        self.out_dropout = nn.Dropout(config['hidden_dropout'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SimpleCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(SimpleCrossAttentionLayer, self).__init__()
        self.num_heads = config['c_num_heads']
        self.c_hidden_dim = config['c_hidden_dim']
        self.head_dim = self.c_hidden_dim // self.num_heads

        assert self.head_dim * self.num_heads == self.c_hidden_dim, "c_hidden_dim must be divisible by num_heads"

        self.W_query = nn.Linear(config['plm_size'], self.c_hidden_dim)
        self.W_key = nn.Linear(config['img_size'], self.c_hidden_dim)
        self.W_value = nn.Linear(config['img_size'], self.c_hidden_dim)
        self.fc = nn.Linear(self.c_hidden_dim, config['plm_size'])
        self.dropout = nn.Dropout(config['hidden_dropout'])

    def forward(self, text, image):
        batch_size = text.size(0)

        # Linear projections
        query = self.W_query(text)  # [batch_size, seq_len_text, c_hidden_dim]
        key = self.W_key(image)  # [batch_size, seq_len_image, c_hidden_dim]
        value = self.W_value(image)  # [batch_size, seq_len_image, c_hidden_dim]

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.c_hidden_dim)

        # Final linear projection
        attention_output = self.fc(attention_output)

        return attention_output


class FourierLayer(nn.Module):
    def __init__(self, config):
        super(FourierLayer, self).__init__()
        self.out_dropout = nn.Dropout(config['f_out_dropout'])
        self.LayerNorm = LayerNorm(config['f_hidden_size'], eps=1e-12)
        self.c = config['c']//2+1  #频率截断位置
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, config['f_hidden_size']))

    def forward(self,input_tensor):
        batch_size,seq_len,hidden=input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1,norm='ortho')

        low_pass = x[:]
        low_pass[:,self.c:,:] = 0
        low_pass = torch.fft.irfft(low_pass,n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft =low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) #残差连接

        return hidden_states

class WaveletLayer(nn.Module):
    def __init__(self, config):
        super(WaveletLayer, self).__init__()
        self.out_dropout = nn.Dropout(config['f_out_dropout'])
        self.LayerNorm = LayerNorm(config['f_hidden_size'], eps=1e-12)
        self.low_freq_transform = nn.Linear(22, 22)

    def forward(self, input_tensor):
        # 执行DWT
        yl, yh = self.dwt(input_tensor)
        # print("yl_shape:", yl.shape)
        # 低频部分的可学习变换
        yl_transformed = self.low_freq_transform(yl)

        # 执行逆DWT
        hidden_states = self.idwt((yl_transformed, yh))

        # 后续处理
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FourierMMLayer(nn.Module):
    def __init__(self, config):
        super(FourierMMLayer, self).__init__()
        self.attention_module = MultiHeadAttention(config)
        self.fourier_module = FourierLayer(config)
        self.cross_attention_module = SimpleCrossAttentionLayer(config)
        self.alpha = config['alpha']

    def forward(self,multi_feature,multi_mask):

        # vision_feat_f = self.fourier_module(input_tensor=image)  # fourier feature of vision
        # vision_feat_a = self.attention_module(input_tensor=image,
        #                                       attention_mask=img_attention_mask)  # self-attention feature of vision
        # vision_hidden_states = self.alpha * vision_feat_f + (1 - self.alpha) * vision_feat_a
        #
        # text_feat_f = self.fourier_module(input_tensor=text)
        # text_feat_a = self.attention_module(input_tensor=text, attention_mask=text_attention_mask)
        # text_hidden_states = self.alpha * text_feat_f + (1 - self.alpha) * text_feat_a
        #
        # vision_cross = self.cross_attention_module(vision_hidden_states,text_hidden_states)
        # text_cross = self.cross_attention_module(text_hidden_states,vision_hidden_states)
        multi_feat_f = self.fourier_module(input_tensor=multi_feature)  # fourier feature of vision
        multi_feat_a = self.attention_module(input_tensor=multi_feature,
                                              attention_mask=multi_mask)  # self-attention feature of vision
        multi_hidden_states = self.alpha * multi_feat_f + (1 - self.alpha) * multi_feat_a

        return multi_hidden_states

class WaveletMMLayer(nn.Module):
    def __init__(self, config):
        super(WaveletMMLayer, self).__init__()
        self.attention_module = MultiHeadAttention(config)
        self.wavelet_module = WaveletLayer(config)
        self.cross_attention_module = SimpleCrossAttentionLayer(config)
        self.alpha = config['alpha']
    def forward(self,multi_feature,multi_mask):

        multi_feat_w = self.wavelet_module(input_tensor=multi_feature)  # wavelet feature of vision
        multi_feat_a = self.attention_module(input_tensor=multi_feature,
                                              attention_mask=multi_mask)  # self-attention feature of vision
        multi_hidden_states = self.alpha * multi_feat_w + (1 - self.alpha) * multi_feat_a

        return multi_hidden_states


class FourierMMEncoderLayer(nn.Module):
    def __init__(self, config):
        super(FourierMMEncoderLayer, self).__init__()
        self.fourier_mm_layer = FourierMMLayer(config)
        self.feedforward = FeedForward(config)

    def forward(self, multi_feature,multi_mask):
        multi = self.fourier_mm_layer(multi_feature,multi_mask)
        multi = self.feedforward(multi)
        return multi

class WaveletMMEncoderLayer(nn.Module):
    def __init__(self, config):
        super(WaveletMMEncoderLayer, self).__init__()
        self.wavelet_mm_layer = WaveletMMLayer(config)
        self.feedforward = FeedForward(config)

    def forward(self, multi_feature,multi_mask):
        multi = self.wavelet_mm_layer(multi_feature,multi_mask)
        multi = self.feedforward(multi)
        return multi


class LearnableFilter(nn.Module):
    def __init__(self, config,max_seq_length):
        super(LearnableFilter, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.complex_weight = nn.Parameter(torch.randn(1, max_seq_length//2 + 1, config['plm_size'], 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(config['f_dropout'])
        self.LayerNorm = LayerNorm(config['plm_size'], eps=1e-12)
        self.layer = nn.Linear(config['plm_size'], config['plm_size'])
        self.beta = config['beta']
        self.cross_attention = SimpleCrossAttentionLayer(config)


    def forward(self, text_tensor,img_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)

        cross_text = self.cross_attention(text_tensor,img_tensor)
        cross_img = self.cross_attention(img_tensor, text_tensor)
        text_tensor = self.beta * cross_text + (1 - self.beta) * text_tensor
        img_tensor = self.beta * cross_img + (1 - self.beta) * img_tensor
        
        batch, seq_len, hidden = text_tensor.shape
        x = torch.fft.rfft(text_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        text_hidden_states = self.out_dropout(sequence_emb_fft)
        text_hidden_states = self.LayerNorm(text_hidden_states + text_tensor)
        text_hidden_states = self.layer(text_hidden_states)



        batch, seq_len, hidden = img_tensor.shape
        x = torch.fft.rfft(img_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        img_hidden_states = self.out_dropout(sequence_emb_fft)
        img_hidden_states = self.LayerNorm(img_hidden_states + img_tensor)
        img_hidden_states = self.layer(img_hidden_states)

        hidden_states = text_hidden_states + img_hidden_states
        return hidden_states


class FourierMMEncoder(SequentialRecommender):
    def __init__(self, config,model):
        super(FourierMMEncoder,self).__init__(config,model)
        self.initializer_range = config['initializer_range']  # 初始化权重的系数(defalut 0.02)
        self.loss_type = config['loss_type']  # defalut CE Loss
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.inner_size = config['inner_size']

        self.hidden_dropout = config['hidden_dropout']
        self.attention_dropout = config['attention_dropout']
        self.hidden_act = config['hidden_act']  # 激活函数(default gelu)
        self.layer_norm_eps = config['layer_norm_eps']  # Layer Norm层的epsilon参数
        self.n_layers = config['transformer_layers']
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.layers = config['transformer_layers']

        self.encoder = nn.ModuleList()
        for _ in range(self.layers):
            self.encoder.append(FourierMMEncoderLayer(config))
    def forward(self):
        pass



class WaveletMMEncoder(SequentialRecommender):
    def __init__(self, config,model):
        super(WaveletMMEncoder,self).__init__(config,model)
        self.initializer_range = config['initializer_range']  # 初始化权重的系数(defalut 0.02)
        self.loss_type = config['loss_type']  # defalut CE Loss
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.inner_size = config['inner_size']

        self.hidden_dropout = config['hidden_dropout']
        self.attention_dropout = config['attention_dropout']
        self.hidden_act = config['hidden_act']  # 激活函数(default gelu)
        self.layer_norm_eps = config['layer_norm_eps']  # Layer Norm层的epsilon参数
        self.n_layers = config['transformer_layers']
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.layers = config['transformer_layers']

        self.encoder = nn.ModuleList()
        for _ in range(self.layers):
            self.encoder.append(WaveletMMEncoderLayer(config))
    def forward(self):
        pass


















