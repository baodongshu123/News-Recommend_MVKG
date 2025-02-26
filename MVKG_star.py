import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.mvkg import MVKGModel
from reco_utils.recommender.newsrec.io.mind_all_iterator526 import MINDAllIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set
print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


epochs = 15
seed = 42
batch_size = 32

# Options: demo, small, large
MIND_type = 'demo'


train_news_file = 'data/train/news.tsv'
train_behaviors_file = 'data/train/behaviors.tsv'
test_news_file = 'data/test/news.tsv'
test_behaviors_file = 'data/test/behaviors.tsv'
valid_news_file = 'data/valide/news.tsv'
valid_behaviors_file = 'data/valide/behaviors.tsv'
wordEmb_file = 'data/utils/embedding.npy'
userDict_file = 'data/utils/uid2index.pkl'
wordDict_file = 'data/utils/word_dict.pkl'
yaml_file = 'data/utils/mvkg.yaml'
vertDict_file = 'data/utils/vert_dict.pkl'
subvertDict_file = 'data/utils/subvert_dict.pkl'
entityEmb_file = 'recommenders/dataset/demo_dkn/entity_embeddings_100.npy'
hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file,
                          subvertDict_file=subvertDict_file,
                          vertDict_file=vertDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                         )
print(hparams)
hparams.lstm_unit=400
hparams.entityDict_file = "data\\utils\\entity_dict.pkl"
hparams.entity_size = 10
hparams.neighbor_size =50
hparams.entity_emb_dim = 100
hparams.entityEmb_file = "data\\utils\\entity_embeddings_100.npy"
hparams.entity_neighbors_file = "data\\utils\\entity_with_neighbors.pkl"
hparams.vert_num = 20
hparams.subvert_num = 249
hparams.cnn_activation = tf.nn.relu
hparams.dense_activation = tf.nn.relu
hparams.user_emb_dim = 400
hparams.neighbor_size = 50
hparams.npratio = 4
hparams.body_size = 50
hparams.vert_size = 1
hparams.learning_rate = 0.000182
hparams.his_size = 40


iterator = MINDAllIterator
print("(((((((((((", iterator)
model = MVKGModel(hparams, iterator, seed=seed)
print("*********")
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

