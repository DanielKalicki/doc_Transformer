import json
from flask import Flask
from flask_cors import CORS, cross_origin

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from models.s2v_gammaTransformer.sentence_processing import embed_sentence
from models.s2v_gammaTransformer.s2v_gammaTransformer import generate_s2v
from models.doc_Transformer.models.doc_encoder import DocEncoder

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

s2v_dim = 4096

# load doc_Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
config = {
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
        'lr': 1e-4,
        'epochs': 5000,
        'log': True
    }
}
doc_Transformer_model = DocEncoder(config)
doc_Transformer_checkpoint_name = './models/doc_Transformer/save/_bL16_docl7_PrClAllSnt_TrVSntSzDt10vD10Doc*SentFix_rndAll_Test.15OthDc_aftClosSentTr_SrcTgt_noPosEnc_1500bFilesRndStartUpd_inDense_Tr4l16h1xhDr.1HidDim1024NormGatedNoFfn_normClasIn_Lr1e-5_resave_10'
checkpoint = torch.load(doc_Transformer_checkpoint_name, map_location=torch.device("cpu"))
doc_Transformer_model.load_state_dict(checkpoint['model_state_dict'])
doc_Transformer_model.to(device)
del checkpoint
doc_Transformer_model.eval()

def get_predictions_from_document(s2v, document):
    input_sentence = torch.zeros((len(document), 1, s2v_dim,), dtype=torch.float)
    encoder_mask = torch.full((len(document), 1, 1,), fill_value=float(0), dtype=torch.float) # TODO fix this -inf
    test_s2v = torch.zeros((len(document), 1, s2v_dim,), dtype=torch.float)

    for sent_idx in range(len(document)):
        input_sentence[sent_idx, -1] = torch.from_numpy(document[sent_idx].astype(np.float32))
    test_s2v[:, -1] = torch.from_numpy(s2v.astype(np.float32))

    data = input_sentence
    test_s2v = test_s2v
    tr_mask = encoder_mask

    data, test_s2v, tr_mask = data.to(device), test_s2v.to(device), tr_mask.to(device)
    data = torch.nn.functional.normalize(data, dim=2)
    test_s2v = torch.nn.functional.normalize(test_s2v, dim=2)
    _, pred_class = doc_Transformer_model(data, test_s2v, mask=tr_mask)
    pred_class = torch.sigmoid(pred_class)

    return pred_class[:, -1, 1]


def get_best_predictions(s2v):
    batch_dir = "../../train/datasets/"
    file_list = []
    for file in os.listdir(batch_dir):
        if (file.split(".")[-1] == 'pickle') and ('s2v' in file):
            file_list.append(file)

    for file in file_list:
        data = pickle.load(open(batch_dir+file, 'rb'))
        for title in data:
            doc_s2vs = []
            for sent in data[title]:
                doc_s2vs.append(sent['sentence_vect_gTr'])
            sent_pred = get_predictions_from_document(s2v, doc_s2vs)
            for idx, score in enumerate(sent_pred):
                if score > 0.9:
                    print(str(round(float(score), 2)) + "\t" + data[title][idx]['text'])

# @app.route("/<sentence>")
# @cross_origin()
def main2(sentence):
    sentence_embeddings = []
    sentence_emb, sentence_token = embed_sentence(sentence.split('|||'))
    for idx in range(0, len(sentence_emb[0])):
        emb = sentence_emb[0][idx]
        tok = sentence_token[0][idx]
        sentence_embeddings.append({'embedding': emb.tolist(), 'token': tok})
    s2v, prediction = generate_s2v(sentence_emb)
    get_best_predictions(s2v[0])
    return json.dumps({'sentence_embeddings': sentence_embeddings,
                       's2v': s2v[0].tolist(),
                       'prediction': prediction[0].tolist()})


if __name__ == "__main__":
    # app.run(port=5001)
    # main2("Electrons have spin 1/2")
    main2("Nitrogen is a chemical element.")
