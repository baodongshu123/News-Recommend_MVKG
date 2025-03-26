# MVKG：Multi-View Personalized News Recommendation Method Integrating Scalable Attention and Knowledge Graphs
MVKG integrates scaled attention mechanisms with knowledge graphs to enhance performance. During the news encoder phase, the algorithm jointly learns
news representations by combining multiple features such as title, abstract, and category. Furthermore, it interacts knowledge entities with news 
content to further enrich the content of news modeling. In the user encoder, a scalable attention mechanism is used to assign different scores to 
users’ varying degrees of interest. Finally, the proposed algorithm in this paper and some baseline algorithm are trained and validated on the 
same public dataset. The results show that the proposed algorithm demonstrates higher performance.


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
from reco_utils.recommender.newsrec.models.mvkg import MVKGModel
from reco_utils.recommender.newsrec.io.mind_all_iterator526 import MINDAllIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set

```
```python
# 2.Parameter configuration
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
```
```python
# 3.model training
iterator = MINDAllIterator
model = MVKGModel(hparams, iterator, seed=seed)
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
```
![image](https://github.com/user-attachments/assets/3060939b-860f-4ea4-a583-aa75eb53183d)



Comment: Due to GitHub's resource upload size limit, the current dataset used is the MIND-small dataset. If you require the MIND-Large dataset, please download it from the official website.
