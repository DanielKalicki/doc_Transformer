import copy

default_config = {
    'batch_size': 64,
    's2v_dim': 4096,
    'doc_len': 7,
    'name': '',
    'transformer': {
        'nhead': 16,
        'num_layers': 4,
        'ffn_dim': 4096,
        'dropout': 0.1
    },
    'training': {
        'optimizer': 'Adam',
        'clipnorm': None,
        'lr': 1e-5,
        'epochs': 5000,
        'log': True
    }
}

configs = []
for i in range(1000):
    configs.append(copy.deepcopy(default_config))

# -----------------------------------------------------------------------------
i = 0
configs[i]['name'] = 'bL16_Tr2l16h1xhDr.1_Lr1e-2_'+str(i)
# -----------------------------------------------------------------------------
i = 1
configs[i]['name'] = 'bL16_Tr2l16h8dhDr.1_Lr1e-2_'+str(i)
# -----------------------------------------------------------------------------
i = 4
configs[i]['name'] = 'bL16_Tr2l16h8dhDr.1_Lr1e-2_'+str(i)
# -----------------------------------------------------------------------------
i = 5
configs[i]['name'] = 'bL16_Tr2l16h8dhDr.1noLastNorm_Lr1e-2_'+str(i)
# -----------------------------------------------------------------------------
i = 6
configs[i]['name'] = 'bL16_posEnc_Tr2l16h8dhDr.1_Lr1e-2_'+str(i)
# -----------------------------------------------------------------------------
i = 7
configs[i]['name'] = 'bL16_docl24_posEnc_Tr1l16h8dhDr.1_Lr5e-3@10.1_'+str(i)
# -----------------------------------------------------------------------------
i = 8
configs[i]['name'] = 'bL128_docl24_200bFiles_trRndSentStart_posEncRemEnd_Tr3l32h8dhDr.1nNorm_Lr5e-5_'+str(i)
# -----------------------------------------------------------------------------
i = 9
# -----------------------------------------------------------------------------
i = 10
configs[i]['name'] = '_bL16_docl7_PrClAllSnt_TrVSntSzDt10vD10Doc*SentFix_rndAll_Test.15OthDc_aftClosSentTr_SrcTgt_noPosEnc_1500bFilesRndStartUpd_inDense_Tr4l16h1xhDr.1HidDim1024NormGatedNoFfn_normClasIn_Lr1e-5_resave_'+str(i)
