import tensorflow as tf

from s2v_embedding_models.s2v_gammaTransformer.models.sentence_encoder_model import SentenceEncoderModel
from s2v_embedding_models.s2v_gammaTransformer.models.nli_matching_model import NliClassifierModel
from s2v_embedding_models.s2v_gammaTransformer.sentence_processing import get_batch
from s2v_embedding_models.s2v_gammaTransformer.configs import configs

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
model_number = 45
config = configs[model_number]
config['batch_size'] = 1

# -----------------------------------------------------------------------------
# Model inputs
# -----------------------------------------------------------------------------
input_sentence1 = tf.keras.layers.Input(
    shape=(config['max_sent_len'], config['word_edim'],),
    name='sentence1')
input_sentence1_mask = tf.keras.layers.Input(
    shape=(config['max_sent_len'],),
    name='sentence1_mask')
input_sentence1_transformer_mask = tf.keras.layers.Input(
    shape=(1, config['max_sent_len'], config['max_sent_len'],),
    name='sentence1_transformer_mask')
input_sentence2 = tf.keras.layers.Input(
    shape=(config['max_sent_len'], config['word_edim'],),
    name='sentence2')
input_sentence2_mask = tf.keras.layers.Input(
    shape=(config['max_sent_len'],), name='sentence2_mask')
input_sentence2_transformer_mask = tf.keras.layers.Input(
    shape=(1, config['max_sent_len'], config['max_sent_len'],),
    name='sentence2_transformer_mask')

# -----------------------------------------------------------------------------
# Sentence encoder
# -----------------------------------------------------------------------------
sentence_encoder_model = SentenceEncoderModel(config)
sent1_s2v = sentence_encoder_model(input_sentence1, input_sentence1_mask,
                                   input_sentence1_transformer_mask)
sent2_s2v = sentence_encoder_model(input_sentence2, input_sentence2_mask,
                                   input_sentence2_transformer_mask)

# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------
nli_matching_model = NliClassifierModel(config)
nli_predictions = nli_matching_model(sent1_s2v, sent2_s2v)

model = tf.keras.models.Model(inputs=[input_sentence1, input_sentence1_mask,
                                      input_sentence1_transformer_mask,
                                      input_sentence2, input_sentence2_mask,
                                      input_sentence2_transformer_mask],
                              outputs=[sent1_s2v, sent2_s2v, nli_predictions])
model.load_weights("./train/batchers/s2v_embedding_models/s2v_gammaTransformer/save/"+config['name']+"/model")


def generate_s2v(sentence_emb):
    x, _ = get_batch(sentence_emb, config)
    sent1_s2v, sent2_s2v, prediction = model.predict(x, batch_size=1)
    return (sent1_s2v, sent2_s2v, prediction)
