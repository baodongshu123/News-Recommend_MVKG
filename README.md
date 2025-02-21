# MVKG：Multi-View news recommendation based on Knowledge Graph
MVKG integrates scaled attention mechanisms with knowledge graphs to enhance performance. During the news encoder phase, the algorithm jointly learns
news representations by combining multiple features such as title, abstract, and category. Furthermore, it interacts knowledge entities with news 
content to further enrich the content of news modeling. In the user encoder, a scalable attention mechanism is used to assign different scores to 
users’ varying degrees of interest. Finally, the proposed algorithm in this paper and some baseline algorithm are trained and validated on the 
same public dataset. The results show that the proposed algorithm demonstrates higher performance.

![image](https://github.com/user-attachments/assets/39cf4542-c815-4d6b-bc4f-aea40b5ec8dd)


```python
# 1.Required libraries and functions to import
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
from reco_utils.recommender.newsrec.models.nrms_526 import NRMSModel
from reco_utils.recommender.newsrec.io.mind_all_iterator526 import MINDAllIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set

```
```python
# 2.Parameter configuration
train_news_file = 'train/news.tsv'
train_behaviors_file = 'train/behaviors.tsv'
valid_news_file = 'valid/news.tsv'
valid_behaviors_file = 'valid/behaviors.tsv'
wordEmb_file = 'utils/embedding.npy'
userDict_file = 'utils/uid2index.pkl'
wordDict_file = 'utils/word_dict.pkl'
yaml_file = 'utils/mvkg.yaml'
vertDict_file = 'utils/vert_dict.pkl'
subvertDict_file = 'utils/subvert_dict.pkl'
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
hparams.entityDict_file = "entity_dict.pkl"
hparams.entity_size = 10
hparams.neighbor_size =50  
hparams.entity_emb_dim = 100
hparams.entityEmb_file = "entity_embeddings_100.npy"
hparams.entity_neighbors_file = "entity_with_neighbors.pkl"
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
```
```python
# 3.model training
iterator = MINDAllIterator
model = NRMSModel(hparams, iterator, seed=seed)
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
```
