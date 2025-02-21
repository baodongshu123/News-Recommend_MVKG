# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]


class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference 
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference 
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(
        self, hparams, iterator_creator, seed=None,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        
        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train (object): NRMS data loader class for train data.
            iterator_creator_test (object): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)

        super().__init__(
            hparams, iterator_creator, seed=seed,
        )

    def _get_input_label_from_iter(self, batch_data):
        """ get input and labels for trainning from iterator

        Args: 
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"]
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
            batch_data["clicked_title_batch"],
            batch_data["clicked_ab_batch"]
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _get_news_feature_from_iter(self, batch_data):
        """ get input of news encoder
        Args:
            batch_data: input batch data from news iterator
        
        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        input_feature = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"]
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS. 

        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title_body_verts = keras.Input(
            shape=(hparams.his_size, hparams.title_size + hparams.body_size),
            dtype="int32",
        )
        #(?,50,82)-->(?,50,400)
        click_news_presents = layers.TimeDistributed(titleencoder)(
            his_input_title_body_verts
        )
        #(?,?,400)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_news_presents] * 3
        )
        y_user = layers.Dropout(hparams.dropout)(y)
        #(?,400)
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            y_user
        )


        model = keras.Model(
            his_input_title_body_verts, user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.
        
        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        input_title_body_verts = keras.Input(
            shape=(hparams.title_size + hparams.body_size,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            input_title_body_verts
        )
        sequences_input_body = layers.Lambda(
            lambda x: x[:, hparams.title_size: hparams.title_size + hparams.body_size]
        )(input_title_body_verts)
        #(?,1,400)
        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr = self._build_bodyencoder(embedding_layer)(sequences_input_body)

        drop_title_repr = layers.Dropout(hparams.dropout)(title_repr)
        drop_body_repr = layers.Dropout(hparams.dropout)(body_repr)

        y_drop_title_repr = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([drop_title_repr,drop_title_repr,drop_title_repr])
        y_drop_body_repr = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([drop_body_repr, drop_body_repr, drop_body_repr])

        y_drop2_title_repr = layers.Dropout(hparams.dropout)(y_drop_title_repr)
        y_drop2_body_repr = layers.Dropout(hparams.dropout)(y_drop_body_repr)

        concate_repr = layers.Concatenate(axis=-2)(
            [y_drop2_title_repr, y_drop2_body_repr]
        )

        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(concate_repr)

        model = keras.Model(input_title_body_verts, pred_title, name="news_encoder")
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.
        
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        his_input_body = keras.Input(
            shape=(hparams.his_size, hparams.body_size), dtype="int32"
        )

        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_body = keras.Input(
            shape=(hparams.npratio + 1, hparams.body_size), dtype="int32"
        )

        pred_input_title_one = keras.Input(
            shape=(hparams.title_size,), dtype="int32"
        )
        pred_input_body_one = keras.Input(shape=(hparams.body_size,), dtype="int32")


        his_title_body_verts = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_body]
        )

        pred_title_body_verts = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_body]
        )

        pred_input_title_one_reshape = layers.Reshape((1, hparams.title_size))(pred_input_title_one)
        pred_input_body_one_reshape = layers.Reshape((1, hparams.body_size))(pred_input_body_one)


        pred_title_body_verts_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one_reshape,
                pred_input_body_one_reshape,
            ]
        )



        pred_title_one_reshape = layers.Reshape((-1,))(pred_title_body_verts_one)
        #word embedding_layer
        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        user_present = self.userencoder(his_title_body_verts)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_verts)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                his_input_title,
                his_input_body,

                pred_input_title,
                pred_input_body,

            ],
            preds,
        )

        scorer = keras.Model(
            [
                his_input_title,
                his_input_body,
                pred_input_title_one,
                pred_input_body_one,

            ],
            pred_one,
        )

        return model, scorer

    def _build_titleencoder(self, embedding_layer):
        """build title encoder of NAML news encoder.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the title encoder of NAML.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, hparams.filter_num))(pred_title)

        model = keras.Model(sequences_input_title, pred_title, name="title_encoder")
        return model

    def _build_bodyencoder(self, embedding_layer):
        """build body encoder of NAML news encoder.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the body encoder of NAML.
        """
        hparams = self.hparams
        sequences_input_body = keras.Input(shape=(hparams.body_size,), dtype="int32")
        embedded_sequences_body = embedding_layer(sequences_input_body)#(?,50,300)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_body)
        # y1 = layers.Conv1D(
        #     hparams.filter_num,
        #     hparams.window_size,
        #     activation=hparams.cnn_activation,
        #     padding="same",
        #
        #     bias_initializer=keras.initializers.Zeros(),
        #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        # )(y)
        y = layers.Dropout(hparams.dropout)(y)

        y_body = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [y, y, y])

        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y_body)
        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)

        model = keras.Model(sequences_input_body, pred_body, name="body_encoder")
        return model
