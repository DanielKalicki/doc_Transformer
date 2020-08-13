import numpy as np
from typing import Optional


def get_batch(sentence_emb, config):
    sentence1 = np.zeros(shape=(config['batch_size'],
                                config['max_sent_len'],
                                config['word_edim']),
                         dtype=np.float32)
    sentence1_mask = np.zeros(shape=(config['batch_size'],
                                     config['max_sent_len']),
                              dtype=np.bool_)
    sentence2 = np.zeros(shape=(config['batch_size'],
                                config['max_sent_len'],
                                config['word_edim']),
                         dtype=np.float32)
    sentence2_mask = np.zeros(shape=(config['batch_size'],
                                     config['max_sent_len']),
                              dtype=np.bool_)
    label = np.zeros(shape=(config['batch_size'], 3), dtype=np.bool_)

    b_idx = 0
    sent1 = sentence_emb[0]
    if len(sentence_emb) > 1:
        sent2 = sentence_emb[1]

    sentence1[b_idx][0:min(len(sent1), config['max_sent_len'])] = \
        sent1[0:min(len(sent1), config['max_sent_len'])]
    sentence1_mask[b_idx][0:min(len(sent1),
                          config['max_sent_len'])] = True

    if len(sentence_emb) > 1:
        sentence2[b_idx][0:min(len(sent2), config['max_sent_len'])] = \
            sent2[0:min(len(sent2), config['max_sent_len'])]
        sentence2_mask[b_idx][0:min(len(sent2),
                              config['max_sent_len'])] = True

    x = {
        'sentence1': sentence1,
        'sentence1_mask': sentence1_mask,
        'sentence1_transformer_mask': create_attention_mask(
            sentence1_mask, is_causal=False, bert_attention=True),
        'sentence2': sentence2,
        'sentence2_mask': sentence2_mask,
        'sentence2_transformer_mask': create_attention_mask(
            sentence2_mask, is_causal=False, bert_attention=True),
    }
    y = {
        'nli_classifier_model': label,
    }
    return x, y


def create_attention_mask(pad_mask: Optional[np.array], is_causal: bool,
                          batch_size: Optional[int] = None,
                          length: Optional[int] = None,
                          bert_attention: bool = False) -> np.array:
    ndim = pad_mask.ndim
    pad_shape = pad_mask.shape
    if ndim == 3:
        pad_mask = np.reshape(pad_mask, (pad_shape[0]*pad_shape[1],
                                         pad_shape[2]))
    if pad_mask is not None:
        assert pad_mask.ndim == 2
        batch_size, length = pad_mask.shape
    if is_causal:
        b = np.cumsum(np.eye(length, dtype=np.float32), axis=0)
    else:
        b = np.ones((length, length), dtype=np.float32)
    b = np.reshape(b, [1, 1, length, length])
    b = np.repeat(b, batch_size, axis=0)  # B, 1, L, L
    if pad_mask is not None:
        _pad_mask = pad_mask[..., np.newaxis]
        _pad_mask = np.repeat(_pad_mask, length, 2)
        _pad_mask_t = np.transpose(_pad_mask, [0, 2, 1])
        if bert_attention:
            tmp = _pad_mask_t
        else:
            tmp = _pad_mask * _pad_mask_t
        tmp = tmp[:, np.newaxis, ...]
        if b is None:
            b = tmp.astype(np.float32)
        else:
            b = b * tmp
    if ndim == 3:
        b_shape = b.shape
        b = np.reshape(b, (pad_shape[0], pad_shape[1], b_shape[1],
                           b_shape[2], b_shape[3]))
    return b
