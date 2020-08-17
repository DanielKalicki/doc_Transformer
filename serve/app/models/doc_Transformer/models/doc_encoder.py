import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn import Parameter

def _get_activation_fn(activation):
    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class CatPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=512):
        super(CatPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        pe = torch.cat(24*[pe], dim=1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.cat((x, self.pe[:x.size(0), :]), dim=2)
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RemovePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=512):
        super(RemovePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x - self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.attn_act = F.relu
        # self.attn_linear = nn.Linear(d_model, d_model)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.linear1 = nn.Conv1d(d_model, dim_feedforward, 3, padding=1)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        self.gate1 = nn.Linear(d_model*2, d_model)
        # self.gate1_act = F.relu
        # self.gate1_ = nn.Linear(d_model*4, d_model)

        # self.gate2 = nn.Linear(d_model*2, d_model)
        # self.gate2_act = F.relu
        # self.gate2_ = nn.Linear(d_model*4, d_model)

        # self.lstm1 = nn.GRU(input_size=d_model*2, hidden_size=d_model, num_layers=1, bidirectional=False)
        # self.lstm2 = nn.GRU(input_size=d_model*2, hidden_size=d_model, num_layers=1, bidirectional=False)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # src2 = self.attn_linear(self.attn_act(src2))
        src2 = torch.sigmoid(self.gate1(torch.cat((src, src2), dim=2))) * src2
        # src2 = torch.sigmoid(self.lstm1(torch.cat((src, src2), dim=2).permute(1,0,2))[0].permute(1,0,2)) * src2
        # src2 = torch.sigmoid(self.gate1_(self.gate1_act(self.gate1(torch.cat((src, src2), dim=2))))) * src2
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src2 = torch.sigmoid(self.gate2(torch.cat((src, src2), dim=2))) * src2
        # # src2 = torch.sigmoid(self.lstm2(torch.cat((src, src2), dim=2).permute(1,0,2))[0].permute(1,0,2)) * src2
        # # src2 = torch.sigmoid(self.gate2_(self.gate2_act(self.gate2(torch.cat((src, src2), dim=2))))) * src2
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

class DenseLayer(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, output_dim=4096,
                 drop=0.0, hidden_act='relu', conv=False, cat_dim=1):
        super(DenseLayer, self).__init__()
        self.hidden_act = hidden_act
        self.conv = conv
        self.cat_dim = cat_dim

        if not self.conv:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(input_dim+hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(input_dim+2*hidden_dim, output_dim)
        else:
            self.conv_1 = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
            self.conv_2 = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 3, padding=1)
            self.conv_3 = nn.Conv1d(input_dim+2*hidden_dim, output_dim, 3, padding=1)

        if self.hidden_act == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        self.drop1 = nn.Dropout(p=drop)
        self.drop2 = nn.Dropout(p=drop)

    def forward(self, x):
        # dense layer 1
        # x = F.normalize(x, dim=2)
        x1_in = x
        if not self.conv:
            x1 = self.fc1(x1_in)
        else:
            x1 = self.conv_1(x1_in.permute(0,2,1)).permute(0,2,1)
        # x1 = F.normalize(x1, dim=2)
        x1 = self.act1(x1)
        x1 = self.drop1(x1)

        # dense layer 2
        x2_in = torch.cat((x, x1), dim=self.cat_dim)
        if not self.conv:
            x2 = self.fc2(x2_in)
        else:
            x2 = self.conv_2(x2_in.permute(0,2,1)).permute(0,2,1)
        # x2 = F.normalize(x2, dim=2)
        x2 = self.act2(x2)
        x2 = self.drop2(x2)

        # dense layer 3
        x3_in = torch.cat((x, x1, x2), dim=self.cat_dim)
        if not self.conv:
            y = self.fc_out(x3_in)
        else:
            y = self.conv_3(x3_in.permute(0,2,1)).permute(0,2,1)
        # y = F.normalize(y, dim=2)
        # # dense layer 3
        # x3_in = torch.cat((x, x1, x2), dim=2)
        # # y = self.fc_out(x3_in)
        # y = self.conv_3(x3_in.permute(0,2,1)).permute(0,2,1)
        return y


class DocEncoder(nn.Module):
    def __init__(self, config):
        super(DocEncoder, self).__init__()
        self.config = config
        s2v_dim = config['s2v_dim']
        nhead = config['transformer']['nhead']
        num_layers = config['transformer']['num_layers']
        dim_feedforward = config['transformer']['ffn_dim']
        tr_drop = config['transformer']['dropout']

        tr_dim = 1024
        # self.pos_encoder = CatPositionalEncoding(4, dropout=0.0, max_len=512)
        # self.rem_pos_encoder = RemovePositionalEncoding(tr_dim, dropout=0.0, max_len=512)
        # self.input_drop = nn.Dropout(p=0.3)

        # self.self_attn = nn.MultiheadAttention(s2v_dim, nhead, dropout=0.1)

        # self.pos_encoder = PositionalEncoding(tr_dim, dropout=0.0, max_len=512)
        encoder_layer = TransformerEncoderLayer(d_model=tr_dim, nhead=nhead,
                                                dim_feedforward=dim_feedforward, dropout=tr_drop)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.conv_1 = nn.Conv1d(s2v_dim, tr_dim, 3, padding=1)
        # self.conv_2 = nn.Conv1d(tr_dim, s2v_dim, 3, padding=1)
        self.dense1 = DenseLayer(input_dim=s2v_dim, hidden_dim=tr_dim, output_dim=tr_dim, drop=0.5, conv=False, cat_dim=2)
        # self.gdense1 = DenseLayer(input_dim=s2v_dim, hidden_dim=tr_dim, output_dim=tr_dim, drop=0.5, conv=False, cat_dim=2)
        # self.dense2 = DenseLayer(input_dim=2*tr_dim, hidden_dim=2*tr_dim, output_dim=2*tr_dim, drop=0.5, conv=False, cat_dim=2)

        # self.dense_class1 = DenseLayer(input_dim=4*s2v_dim, hidden_dim=tr_dim, output_dim=s2v_dim, drop=0.5, conv=False)
        # self.dense_class2 = DenseLayer(input_dim=s2v_dim, hidden_dim=tr_dim, output_dim=2, drop=0.5, conv=False)
        # self.conv_2 = nn.Conv1d(tr_dim, tr_dim, 3, padding=1)
        # self.out_projection = nn.Linear(s2v_dim, 3*s2v_dim)

        # self.pos_encoder2 = PositionalEncoding(tr_dim, dropout=0.0, max_len=512)
        # encoder_layer2 = TransformerEncoderLayer(d_model=s2v_dim, nhead=nhead,
        #                                         dim_feedforward=dim_feedforward, dropout=tr_drop)
        # self.tr2 = nn.TransformerEncoder(encoder_layer2, num_layers=2)
        # self.conv_1_2 = nn.Conv1d(s2v_dim, tr_dim, 3, padding=1)
        # self.conv_2_2 = nn.Conv1d(tr_dim, s2v_dim, 3, padding=1)
        # self.res_conv_1 = nn.Conv1d(tr_dim, tr_dim, 3, padding=1)
        # self.res_conv_1_act = F.relu
        # self.res_lin_1 = nn.Linear(tr_dim, tr_dim)

        # self.res_conv_2 = nn.Conv1d(tr_dim, tr_dim, 3, padding=1)
        # self.res_conv_2_act = F.relu
        # self.res_lin_2 = nn.Linear(tr_dim, tr_dim)

        # self.res_conv_3 = nn.Conv1d(tr_dim, tr_dim, 3, padding=1)
        # self.res_conv_3_act = F.relu
        # self.res_lin_3 = nn.Linear(tr_dim, tr_dim)

        # self.res_conv_4 = nn.Conv1d(tr_dim, tr_dim, 3, padding=1)
        # self.res_conv_4_act = F.relu
        # self.res_lin_4 = nn.Linear(tr_dim, tr_dim)

        # self.lstm = nn.LSTM(input_size=tr_dim, hidden_size=tr_dim, num_layers=1, bidirectional=False)
        # self.fc1 = nn.Linear(4*s2v_dim, s2v_dim)
        # self.fc1 = nn.Linear(4*tr_dim, s2v_dim)
        # self.dense_class1 = DenseLayer(input_dim=4*tr_dim, hidden_dim=s2v_dim, output_dim=s2v_dim, drop=0.5, conv=False, cat_dim=2)
        self.fc1 = nn.Linear(4*tr_dim, s2v_dim)
        self.act1 = F.relu
        self.fc2 = nn.Linear(s2v_dim, 2)
        # self.dense_class2 = DenseLayer(input_dim=s2v_dim, hidden_dim=s2v_dim, output_dim=2, drop=0.5, conv=False, cat_dim=2)
        # self.act2 = torch.sigmoid

    def forward(self, x, test_s2v, mask):
        # x = x.permute((1, 0, 2))
        # x = self.tr2(x, mask=mask[0])
        # x = x.permute((1, 0, 2))

        # x = self.input_drop(x)

        x = self.dense1(x)
        # x = self.dense1(x) * torch.sigmoid(self.gdense1(x))
        test_s2v = self.dense1(test_s2v)
        # test_s2v = self.dense1(test_s2v) * torch.sigmoid(self.gdense1(test_s2v))

        x = x.permute((1, 0, 2))
        # x, _ = self.lstm(x)
        # x = self.pos_encoder(x)
        x = self.tr(x, mask=mask[0])
        # x = self.rem_pos_encoder(x)
        x = x.permute((1, 0, 2))

        # x = x.permute((0, 2, 1))
        # x = self.conv_2(x)
        # x = x.permute((0, 2, 1))

        # x = self.dense2(x)

        # x_pred = x[:, self.config['doc_len']-1]
        x_pred = x

        test_s2v = F.normalize(test_s2v, dim=2)
        x_pred = F.normalize(x, dim=2)

        x_class_in = torch.cat((x_pred, test_s2v, x_pred*test_s2v, torch.abs(x_pred-test_s2v)), dim=2)
        # x_class = self.dense_class1(x_class_in)
        x_class = self.fc1(x_class_in)
        x_class = self.act1(x_class)
        # x_class = self.dense_class2(x_class)
        x_class = self.fc2(x_class)
        # x_class = self.dense_class1(x_class_in)
        # x_class = self.dense_class2(x_class)

        return x, x_class
