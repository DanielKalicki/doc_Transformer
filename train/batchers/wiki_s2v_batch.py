import os.path
import json
import bz2
import pickle
import random
import re
import copy
import math
from urllib.parse import unquote
import numpy as np
import torch
from torch.utils.data import Dataset
# from flair.embeddings import RoBERTaEmbeddings
# from flair.data import Sentence
# from s2v_embedding_models.s2v_gammaTransformer.s2v_gammaTransformer import generate_s2v
# import spacy
from tqdm import tqdm
from datetime import datetime

random.seed(0)
batch_train_data = []
batch_train_data_size = 0
batch_valid_data = []
batch_valid_data_size = 0
batch_full_data = {}
batch_file_list = []
init_cnt = 0


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def find_html_links_from_wikipage(text):
    links = []
    # search_results = re.findall('\shref.*?>', text)
    search_results = re.findall("href=\"view-source:https://en.wikipedia.org/wiki/.*?>", text)
    search_results = re.findall("<a href=\"https://en.wikipedia.org/wiki/.*?\s", text)
    for link in search_results:
        links.append(unquote(
            link.replace("<a href=\"https://en.wikipedia.org/wiki/", "")[:-2]))
    return links

class WikiS2vBatch(Dataset):
    def __init__(self, config, valid=False):
        self.config = config
        self.valid = valid

        self.datasets_dir = "/home/kalickid/Projects/github/s2v_linker/datasets/hotpotqa/"
        self.batch_dir = './train/datasets/'

        # articles = self._get_input_articles_list_from_vital_articles()
        # self._create_batch_words_emb(articles)
        # self._create_batch_s2v()
        # exit(1)

        self.train_batch_part = -1
        self._init_batch()
    
    def _get_input_articles_list_from_vital_articles(self):
        self.vital_articles_dump_dir = './datasets/wiki/'
        articles = set()
        for file in sorted(os.listdir(self.vital_articles_dump_dir)):
            if file.split(".")[-1] == 'html':
                cnt = 0
                if 'html' in file:
                    with open(self.vital_articles_dump_dir+file, "r") as f:
                        for link in find_html_links_from_wikipage(f.read()):
                            key_words_list = [":", "Main_Page"]
                            if all(word not in link for word in key_words_list):
                                article = link.replace("_", " ")
                                articles.update([article.lower()])
                                cnt += 1
        print('Number of input articles: '+ str(len(articles)))
        return articles
    

    def _embed_sentences(self, text):
        article = []
        for p_idx, paragraph in enumerate(text):
            for l_idx, line in enumerate(paragraph):
                sentence = line
                sentence = remove_html_tags(sentence)
                if len(sentence) > 0:
                    sentence_emb = self._process_sentences([sentence])
                    article.append({
                        'sentence_emb': sentence_emb,
                        'paragraph_idx': p_idx,
                        'line_idx': l_idx,
                        'text': sentence
                    })
        return article
    
    def _create_batch_words_emb(self, articles):
        self.embedding = RoBERTaEmbeddings(
            pretrained_model_name_or_path="roberta-large",
            layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20," +
                   "21,22,23,24",
            pooling_operation="mean", use_scalar_mix=True)
        all_articles_lc = [x.lower() for x in articles]
        all_sentences = 0
        for folder in sorted(os.listdir(self.datasets_dir)):
            for file in sorted(os.listdir(self.datasets_dir+folder)):
                if file.split(".")[-1] == 'bz2':
                    print(self.datasets_dir+folder+'/'+file)
                    if not os.path.isfile(self.batch_dir + folder + file.replace("wiki","").replace(".bz2", "") + "_" + 'articles_wEmb.pickle'):
                        article_dict = {}
                        with bz2.BZ2File(self.datasets_dir+folder+"/"+file, "r") as fp:
                            for line in fp:
                                data = json.loads(line)
                                title = data['title'].lower()
                                if (title in all_articles_lc) and (title != "oath of office") and (title != "punjabi language") and \
                                   (title != "surname") and (title != "florence la badie") and (title != "plasmodium") and (title != "anambra state") and \
                                   (title != "norodom sihamoni"):
                                    text = data['text'][1:-1] # skip first and last paragraph
                                    sent_cnt = 0
                                    for paragraph in text:
                                        sent_cnt += len(paragraph)
                                    if sent_cnt > 8:
                                        print(title, sent_cnt)
                                        try:
                                            article_dict[title] = self._embed_sentences(text)
                                        except IndexError:
                                            print('\tIndexError')
                        if len(article_dict) > 0:
                            pickle.dump(article_dict, open(self.batch_dir + folder + file.replace("wiki","").replace(".bz2", "") + "_" + 'articles_wEmb.pickle', 'wb'))
    
    def _process_sentences(self, sentences):
        sentences_emb = []
        for sentence in sentences:
            sentence = " ".join(sentence.split())
            sent = sentence
            if len(sent.strip()) == 0:
                sent = 'empty'
            if len(sent.split(" ")) > 220:
                print(len(sent.split(" ")))
                sent = sent[:512]
            try:
                sent = Sentence(sent)
                self.embedding.embed(sent)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
            except IndexError:
                print('IndexError')
                print(sentence)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
        sentences_emb_short = sentences_emb
        return sentences_emb_short

    def _create_batch_s2v(self):
        for file in sorted(os.listdir(self.batch_dir)):
            if (file.split(".")[-1] == 'pickle') and (not os.path.isfile(self.batch_dir + file.replace('_wEmb.', '_s2v.'))) and \
               ('_wEmb' in file):
                print(file)
                data = pickle.load(open(self.batch_dir+file, 'rb'))
                print(len(data))
                for title in data:
                    print(title)
                    pbar = tqdm(total=len(data[title]), dynamic_ncols=True)
                    for sentence in data[title]:
                        s2v_gammaTransformer = self._generate_s2v(sentence['sentence_emb'])
                        del sentence['sentence_emb']
                        sentence['sentence_vect_gTr'] = s2v_gammaTransformer[0]
                        pbar.update(1)
                    pbar.close()
                if len(data) > 0:
                    pickle.dump(data, open(self.batch_dir + file.replace('_wEmb.', '_s2v.'), 'wb'))

    def _generate_s2v(self, sentence):
        sent1_s2v, _, _ = generate_s2v(sentence)
        return sent1_s2v

    def _init_batch(self):
        global batch_train_data, batch_valid_data, batch_full_data, init_cnt, batch_train_data_size, batch_valid_data_size, batch_file_list
        if not self.valid:
            batch_full_data = {}
            batch_train_data = []
            batch_valid_data = []

            start_idx = init_cnt
            num_of_files_in_one_batch = 1500
            if len(batch_file_list) == 0:
                for file in os.listdir(self.batch_dir):
                    if (file.split(".")[-1] == 'pickle') and ('s2v' in file):
                        batch_file_list.append(file)
                random.shuffle(batch_file_list)
            if (start_idx+1)*num_of_files_in_one_batch > len(batch_file_list):
                init_cnt = 0
                start_idx = init_cnt
            for file in batch_file_list[start_idx*num_of_files_in_one_batch:(start_idx+1)*num_of_files_in_one_batch]:
                data = pickle.load(open(self.batch_dir+file, 'rb'))
                for title in data:
                    if len(data[title]) > self.config['doc_len']:
                        batch_full_data[title] = data[title]
            for title in batch_full_data:
                batch_train_data.append(title)
            random.seed(init_cnt)
            random.shuffle(batch_train_data)
            random.seed(datetime.now())
            batch_valid_data = batch_train_data[:int(len(batch_train_data)*0.1)]
            batch_train_data = batch_train_data[int(len(batch_train_data)*0.1):]
            print("init_cnt: "+str(init_cnt)+" ["+str(start_idx*num_of_files_in_one_batch)+":"+str((start_idx+1)*num_of_files_in_one_batch)+"]")
            init_cnt += 1
            print(len(batch_file_list))
            print(batch_valid_data[0:10])
            batch_train_data_size = 0
            for title in batch_train_data:
                batch_train_data_size += len(batch_full_data[title])
            batch_valid_data_size = 0
            for title in batch_valid_data:
                batch_valid_data_size += len(batch_full_data[title])

            print("Train dataset size: " + str(batch_train_data_size))
            print("Test dataset size: " + str(batch_valid_data_size))
    
    def _find_closest_s2v(self, s2v, article=None):
        sim_dict = {}
        a = s2v
        for art in batch_full_data:
            if article:
                art = article
            for sentence in batch_full_data[art]:
                b = sentence['sentence_vect_gTr']
                p = sentence['paragraph_idx']
                l = sentence['line_idx']
                line = sentence['text']
                sim = ((a - b)**2).mean(axis=0)
                if sim not in sim_dict:
                    sim_dict[sim] = []
                sim_dict[sim].append(art+" p:" + str(p) + " l:" + str(l)+" ---> "+remove_html_tags(line)[0:190])
                print(str(sim) + "\t" + str(sim_dict[sim]))
            if article:
                break
        for key in sorted(sim_dict)[0:11]:
            print('\t'+str(key)+'\t'+str(sim_dict[key]))

    def on_epoch_end(self):
        self._init_batch()

    def __len__(self):
        global batch_train_data, batch_valid_data
        if self.valid:
            return int(batch_valid_data_size/2)
        else:
            return int(batch_train_data_size/10)

    def get_title_from_idx(self, batch_dataset):
        title_list = []
        title_weight = []
        for title in batch_dataset:
            title_list.append(title)
            title_weight.append(len(batch_full_data[title]))
        rnd_title = random.choices(title_list, weights=title_weight)[0]
        return rnd_title

    def __getitem__(self, idx):
        global batch_train_data, batch_valid_data 
        batch_dataset = batch_valid_data if self.valid else batch_train_data
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_sentence = torch.zeros((self.config['doc_len'], self.config['s2v_dim'],),
                                        dtype=torch.float)
        output_sentence = torch.zeros((self.config['doc_len'], self.config['s2v_dim'],),
                                      dtype=torch.float)
        encoder_mask = torch.full((self.config['doc_len'], self.config['doc_len'],),
                                    fill_value=float("-inf"), dtype=torch.float)
        input_mask = torch.ones((self.config['doc_len'], self.config['s2v_dim'],),
                                 dtype=torch.float)
        output_mask = torch.zeros((self.config['doc_len'], self.config['s2v_dim'],),
                                  dtype=torch.float)
        test_s2v = torch.zeros((self.config['doc_len'], self.config['s2v_dim'], ), dtype=torch.float)
        test_label = torch.zeros((self.config['doc_len'], 2, ), dtype=torch.float)

        input_title = self.get_title_from_idx(batch_dataset)

        start_sent_idx = 0
        if not self.valid or True:
            if len(batch_full_data[input_title]) > (self.config['doc_len']+1):
                start_sent_idx = random.randint(0, len(batch_full_data[input_title])-self.config['doc_len']-2)
        for sent_idx, _ in enumerate(batch_full_data[input_title]):
            if sent_idx >= self.config['doc_len']:
                break
            if (sent_idx+start_sent_idx+1) >= len(batch_full_data[input_title]):
                break
            sentence = batch_full_data[input_title][sent_idx+start_sent_idx]
            sentence_next = batch_full_data[input_title][sent_idx+start_sent_idx+1]
            rnd_mask = random.random()
            if rnd_mask < 0.10:
                input_mask[sent_idx, :] = torch.tensor(0.0)
            input_sentence[sent_idx] = torch.from_numpy(sentence['sentence_vect_gTr'].astype(np.float32))
            output_sentence[sent_idx] = torch.from_numpy(sentence_next['sentence_vect_gTr'].astype(np.float32))
            output_mask[sent_idx, :] = torch.tensor(1.0)

        for sent_idx in range(self.config['doc_len']):
            encoder_mask[sent_idx, 0:sent_idx+1] = torch.tensor(0.0)
        
        for sent_idx in range(self.config['doc_len']):
            rnd_label = random.random()
            test_label[sent_idx][int(rnd_label < 0.5)] = torch.tensor(1.0)
            if (rnd_label < 0.5):
                test_s2v[sent_idx] = output_sentence[sent_idx]
            else:
                rnd_title = random.random()
                test_doc_title = input_title
                if (rnd_title < 0.15) and (not self.valid): # TODO weight choice
                    rnd_idx = random.randint(0, len(batch_dataset)-1)
                    test_doc_title = batch_dataset[rnd_idx]
                rnd_sent_idx = random.randint(0, len(batch_full_data[test_doc_title])-2)
                # if (sent_idx >= 3) and (rnd_title >= 0.15):
                #     rnd_prev_sent = random.random() # 30% chance to use previous sentence in transformer
                #     if rnd_prev_sent < 0.30:
                #         rnd_sent_idx = random.randint(0, sent_idx-1)+start_sent_idx
                #     elif rnd_prev_sent < 0.60: # 30% chance to use one of next 5 sentences
                #         max_rand = min(len(batch_full_data[test_doc_title])-1, sent_idx+start_sent_idx+2+5)
                #         if max_rand > sent_idx+start_sent_idx+2:
                #             rnd_sent_idx = random.randint(sent_idx+start_sent_idx+2, max_rand)
                if rnd_sent_idx == (sent_idx+start_sent_idx+1):
                    rnd_sent_idx = len(batch_full_data[test_doc_title])-1
                test_s2v[sent_idx] = torch.from_numpy(batch_full_data[test_doc_title][rnd_sent_idx]['sentence_vect_gTr'].astype(np.float32))

        return (input_sentence, output_sentence, encoder_mask, input_mask, output_mask, test_s2v, test_label)

    def get_test_data(self, art_idx, start_sent_idx, test_sent_idx, init=False):
        global batch_train_data, batch_valid_data
        batch_dataset = batch_valid_data if self.valid else batch_train_data

        input_sentence = torch.zeros((1, self.config['doc_len'], self.config['s2v_dim'],),
                                        dtype=torch.float)
        output_sentence = torch.zeros((1, self.config['doc_len'], self.config['s2v_dim'],),
                                      dtype=torch.float)
        encoder_mask = torch.full((1, self.config['doc_len'], self.config['doc_len'],),
                                    fill_value=float("-inf"), dtype=torch.float)
        input_mask = torch.ones((1, self.config['doc_len'], self.config['s2v_dim'],),
                                 dtype=torch.float)
        output_mask = torch.zeros((1, self.config['doc_len'], self.config['s2v_dim'],),
                                  dtype=torch.float)
        test_s2v = torch.zeros((1, self.config['doc_len'], self.config['s2v_dim'], ), dtype=torch.float)
        test_label = torch.zeros((1, self.config['doc_len'], 2, ), dtype=torch.float)

        idx = art_idx
        input_title = batch_dataset[idx]
        if init:
            print("-----"+input_title+"-----")
        for sent_idx, _ in enumerate(batch_full_data[input_title]):
            if sent_idx >= self.config['doc_len']:
                break
            if (sent_idx+start_sent_idx+1) >= len(batch_full_data[input_title]):
                break
            sentence = batch_full_data[input_title][sent_idx+start_sent_idx]
            input_sentence[0, sent_idx] = torch.from_numpy(sentence['sentence_vect_gTr'].astype(np.float32))
            if init:
                print(sentence['text'])

        for sent_idx in range(self.config['doc_len']):
            encoder_mask[0, sent_idx, 0:sent_idx+1] = torch.tensor(0.0)

        test_sent = batch_full_data[input_title][test_sent_idx]
        if (start_sent_idx + self.config['doc_len']) == test_sent_idx:
            print(".................")
        print('\t'+test_sent['text'], end='')
        test_s2v[0, -1] = torch.from_numpy(test_sent['sentence_vect_gTr'].astype(np.float32))

        return (input_sentence, output_sentence, encoder_mask, input_mask, output_mask, test_s2v, test_label)
    
    def check_accuracy(self, s2v):
        self._find_closest_s2v(s2v.to("cpu").numpy())


def test():
    batcher = WikiS2vBatch({
        's2v_dim': 4096,
        'doc_len': 400
    })
    for i in range(100):
        batcher.__getitem__(i)


# test()
