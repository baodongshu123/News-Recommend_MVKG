# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import AttLayer2

__all__ = ["NAMLModel"]


class TANRModel(BaseModel):
    """NAML model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie,
    Neural News Recommendation with Attentive Multi-View Learning, IJCAI 2019

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization steps for NAML.
        Compared with the BaseModel, NAML need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        
        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train (object): NAML data loader class for train data.
            iterator_creator_test (object): NAML data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams

        super().__init__(hparams, iterator_creator, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["clicked_title_batch"],


            batch_data["candidate_title_batch"],

        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """ get input of user encoder 
        Args:
            batch_data: input batch data from user iterator
        
        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        input_feature = [
            batch_data["clicked_title_batch"]
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _get_news_feature_from_iter(self, batch_data):
        """ get input of news encoder
        Args:
            batch_data: input batch data from news iterator
        
        Returns:
            numpy.ndarray: input news feature (candidate titl_get_user_feature_from_itere batch)
        """
        input_feature = [
            batch_data["candidate_title_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _build_graph(self):
        """Build NAML model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_tanr()
        return model, scorer

    def _build_userencoder(self, newsencoder):
        """The main function to create user encoder of NAML.

        Args:
            newsencoder (object): the news encoder of NAML. 

        Return:
            object: the user encoder of NAML.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        click_title_presents = layers.TimeDistributed(newsencoder)(his_input_title)
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(click_title_presents)

        model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NAML.
        news encoder in composed of title encoder, body encoder, vert encoder and subvert encoder

        Args:
            embedding_layer (object): a word embedding layer.
        
        Return:
            object: the news encoder of NAML.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")

        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_tanr(self):
        """The main function to create NAML's logic. The core of NAML
        is a user encoder and a news encoder.
        
        Returns:
            object: a model used to train.
            object: a model used to evaluate and predict.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(1, hparams.title_size,), dtype="int32"
        )
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )
        # word embedding_layer
        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        user_present = self.userencoder(his_input_title)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([his_input_title, pred_input_title], preds)
        scorer = keras.Model([his_input_title, pred_input_title_one], pred_one)

        return model, scorer