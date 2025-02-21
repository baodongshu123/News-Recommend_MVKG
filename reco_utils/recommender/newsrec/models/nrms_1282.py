# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from tensorflow.keras.layers import Attention
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import layers

from tensorflow.keras import backend as K
from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import AttLayer2, SelfAttention,PersonalizedAttentivePooling
# LSMV
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
        self, hparams, iterator_creator, seed=None,):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train (object): NRMS data loader class for train data.
            iterator_creator_test (object): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.entitty2vec_embedding = self._init_embedding(hparams.entityEmb_file)
        super().__init__(hparams, iterator_creator, seed=seed,)

    def _get_input_label_from_iter(self, batch_data):
        """ get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data['user_index_batch'],
            batch_data["clicked_title_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
            batch_data["clicked_entity_batch"],
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
            batch_data["candidate_entity_batch"],
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
            batch_data["clicked_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
            batch_data["clicked_entity_batch"],

        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return [input_feature,batch_data['user_index_batch']]

    def _get_news_feature_from_iter(self, batch_data):
        """ get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        input_feature = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
            batch_data["candidate_entity_batch"],
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

    def _build_userencoder(self, titleencoder,type="con"):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title_body_verts = keras.Input(
            shape=(hparams.his_size, hparams.title_size + hparams.body_size + 2 + hparams.entity_size),
            dtype="int32",
        )



        # 长短期注意力网络
        # 加的
        user_indexes = keras.Input(shape=(1,), dtype="int32")
        user_embedding_layer = layers.Embedding(
            len(self.train_iterator.uid2index),
            hparams.gru_unit,
            trainable=True,
            embeddings_initializer="zeros",
        )
        long_u_emb = layers.Reshape((hparams.gru_unit,))(
            user_embedding_layer(user_indexes)
        )
        # 不是加的
        print(his_input_title_body_verts)  # (?,50,92)
        click_news_presents = keras.layers.TimeDistributed(titleencoder)(
            his_input_title_body_verts
        )
        print(click_news_presents)  # shape=(?, 50, 400)


        # 自己加的
        user_present = layers.GRU(
            hparams.gru_unit,
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            bias_initializer=keras.initializers.Zeros(),
        )(
            layers.Masking(mask_value=0.0)(click_news_presents),
            initial_state=[long_u_emb],
        )
        # 加的
        # if type == "ini":
        #     user_present = layers.GRU(
        #         hparams.gru_unit,
        #         kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #         recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #         bias_initializer=keras.initializers.Zeros(),
        #     )(
        #         layers.Masking(mask_value=0.0)(click_news_presents),
        #         initial_state=[long_u_emb],
        #     )
        #     print("aaaaaaaaaaaaaaaaaaaa")


        # elif type == "con":
        #     short_uemb = layers.GRU(
        #         hparams.gru_unit,
        #         kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #         recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #         bias_initializer=keras.initializers.Zeros(),
        #     )(layers.Masking(mask_value=0.0)(click_news_presents))
            # 改为使用LSTM来学习短期用户表示
            # short_uemb = layers.LSTM(
            #     hparams.lstm_unit,
            #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            #     recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            #     bias_initializer=keras.initializers.Zeros(),
            # )(layers.Masking(mask_value=0.0)(click_news_presents))


            # user_present = layers.Concatenate()([short_uemb, long_u_emb])
            # user_present = layers.Dense(
            #     hparams.gru_unit,
            #     bias_initializer=keras.initializers.Zeros(),
            #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            # )(user_present)


        #########################################################################################################
        #这是师兄的代码，Lstm是我自己找的
        #click_news_presents = tf.reduce_mean(click_news_presents,axis=1)  #这要保证输入的代码是#shape=(?, 50, 400)
        #print(click_news_presents)  #shape=(?, 400)

        # short_uemb = layers.GRU(
        #     hparams.gru_unit,
        #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #     recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #     bias_initializer=keras.initializers.Zeros(),
        # )(layers.Masking(mask_value=0.0)(click_news_presents))
        #

        # short_uemb = layers.LSTM(
        #     hparams.lstm_unit,
        #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #     recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #     bias_initializer=keras.initializers.Zeros(),
        # )(layers.Masking(mask_value=0.0)(click_news_presents))
        # print(short_uemb)     #shape=(?, 400)
        # user_present = PersonalizedAttentivePooling(
        #     hparams.his_size,
        #     hparams.filter_num,
        #     hparams.attention_hidden_dim,
        #     seed=self.seed,
        # )([click_news_presents, layers.Dense(hparams.attention_hidden_dim)(short_uemb)])
        # print(user_present)  #shape=(?, 400)
        #############################################################################################################


        # user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
        #     click_news_presents
        # )
        # user_present = tf.reduce_mean(click_news_presents,axis=1)  #这要保证输入的代码是#shape=(?, 50, 400)

        print("wwwwwwwwwwwwwwwwwwwwwwwwwww")
        print(user_present)   #shape=(?, 400)
        # 加了user_indexes
        model = keras.Model(
            [his_input_title_body_verts,user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer,entity_embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        input_title_body_verts = keras.Input(
            shape=(hparams.title_size + hparams.body_size + 2+ hparams.entity_size,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            input_title_body_verts
        )
        sequences_input_body = layers.Lambda(
            lambda x: x[:, hparams.title_size: hparams.title_size + hparams.body_size]
        )(input_title_body_verts)
        input_vert = layers.Lambda(
            lambda x: x[
                      :,
                      hparams.title_size
                      + hparams.body_size: hparams.title_size
                                           + hparams.body_size
                                           + 1,
                      ]
        )(input_title_body_verts)
        input_subvert = layers.Lambda(
            lambda x: x[:, hparams.title_size + hparams.body_size + 1:hparams.title_size
                                                                      + hparams.body_size
                                                                      + 1 + 1]
        )(input_title_body_verts)
        sequence_input_entity = layers.Lambda(
            lambda x: x[:, hparams.title_size + hparams.body_size + 1 + 1:]
        )(input_title_body_verts)

        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr = self._build_bodyencoder(embedding_layer)(sequences_input_body)
        vert_repr = self._build_vertencoder()(input_vert)
        subvert_repr = self._build_subvertencoder()(input_subvert)
        entity_repr = self._build_entityencoder(entity_embedding_layer)(sequence_input_entity)
        concate_repr = layers.Concatenate(axis=-2)(
            # [title_repr, body_repr, vert_repr,subvert_repr,entity_repr]
            [title_repr, body_repr, vert_repr, entity_repr]   #去掉子类别
            # [title_repr, body_repr, subvert_repr,entity_repr]  #缺主类别
            # [ body_repr, vert_repr, subvert_repr, entity_repr] #缺标题
            # [title_repr,  vert_repr, subvert_repr, entity_repr]  #缺摘要
            # [title_repr, body_repr, vert_repr, subvert_repr]  #缺知识实体
        )


        print("concate_repr")
        print(concate_repr)    #shape=(?, 5, 400)
        # y = layers.Dropout(hparams.dropout)(concate_repr)
        # y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        # y = layers.Dropout(hparams.dropout)(y)
        # pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(concate_repr)
        pred_title = tf.reduce_mean(concate_repr, axis=1)
        model = keras.Model(input_title_body_verts, pred_title, name="news_encoder")
        return model
    def _build_entityencoder(self, entity_embedding_layer):
        """build title encoder of NAML news encoder.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the title encoder of NAML.
        """
        hparams = self.hparams
        sequences_input_entity = keras.Input(shape=(hparams.entity_size,), dtype="int32")
        embedded_sequences_entity = entity_embedding_layer(sequences_input_entity)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_entity)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        # y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
        #     [y] * 3
        # )
        y = layers.Dropout(hparams.dropout)(y)
        pred_entity = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_entity = layers.Reshape((1, hparams.filter_num))(pred_entity)

        model = keras.Model(sequences_input_entity, pred_entity, name="entity_encoder")
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
        his_input_vert = keras.Input(shape=(hparams.his_size, 1), dtype="int32")
        his_input_subvert = keras.Input(shape=(hparams.his_size, 1), dtype="int32")
        his_input_entity = keras.Input(shape=(hparams.his_size, hparams.entity_size), dtype="int32")


        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_body = keras.Input(
            shape=(hparams.npratio + 1, hparams.body_size), dtype="int32"
        )
        pred_input_vert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")
        pred_input_subvert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")
        pred_input_entity = keras.Input(shape=(hparams.npratio + 1, hparams.entity_size), dtype="int32")



        pred_input_title_one = keras.Input(
            shape=(hparams.title_size,), dtype="int32"
        )
        pred_input_body_one = keras.Input(shape=(hparams.body_size,), dtype="int32")
        pred_input_vert_one = keras.Input(shape=(1), dtype="int32")
        pred_input_subvert_one = keras.Input(shape=(1), dtype="int32")
        pred_input_entity_one = keras.Input(shape=(1, hparams.entity_size,), dtype="int32")



        his_title_body_verts = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_body, his_input_vert, his_input_subvert, his_input_entity])

        pred_title_body_verts = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_body, pred_input_vert, pred_input_subvert, pred_input_entity])

        pred_input_title_one_reshape = layers.Reshape((1, hparams.title_size))(pred_input_title_one)
        pred_input_body_one_reshape = layers.Reshape((1, hparams.body_size))(pred_input_body_one)
        pred_input_vert_one_reshape = layers.Reshape((1, 1))(pred_input_vert_one)
        pred_input_subvert_one_reshape = layers.Reshape((1, 1))(pred_input_subvert_one)
        pred_input_entity_one_reshape = layers.Reshape((1,hparams.entity_size))(pred_input_entity_one)
        pred_title_body_verts_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one_reshape,
                pred_input_body_one_reshape,
                pred_input_vert_one_reshape,
                pred_input_subvert_one_reshape,
                pred_input_entity_one_reshape,
            ]
        )
        pred_title_one_reshape = layers.Reshape((-1,))(pred_title_body_verts_one)


        # 加的
        user_indexes = keras.Input(shape=(1,), dtype="int32")


        #word embedding_layer
        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )
        entity_embedding_layer = layers.Embedding(
            self.entitty2vec_embedding.shape[0],
            hparams.entity_emb_dim,
            weights=[self.entitty2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer,entity_embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder,type=hparams.type)
        self.newsencoder = titleencoder

        # 加了user_index
        user_present = self.userencoder([his_title_body_verts, user_indexes])
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_verts)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [user_indexes,
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                his_input_entity,
                pred_input_title,
                pred_input_body,
                pred_input_vert,
                pred_input_subvert,
                pred_input_entity,
            ],
            preds,
        )

        scorer = keras.Model(
            [user_indexes,
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                his_input_entity,
                pred_input_title_one,
                pred_input_body_one,
                pred_input_vert_one,
                pred_input_subvert_one,
                pred_input_entity_one,
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

        # y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
        #     [y] * 3
        # )
        # 自己改动了，加了GRU和PersonalizedAttentivePooling，屏蔽了AttLayer2
        # short_uemb = layers.GRU(
        #     hparams.gru_unit,
        #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #     recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        #     bias_initializer=keras.initializers.Zeros(),
        # )(layers.Masking(mask_value=0.0)(y))
        # print(short_uemb)     #shape=(?, 400)
        # pred_title = PersonalizedAttentivePooling(
        #     hparams.his_size,
        #     hparams.filter_num,
        #     hparams.attention_hidden_dim,
        #     seed=self.seed,
        # )([y, layers.Dense(hparams.attention_hidden_dim)(short_uemb)])
        # print(user_present)  #shape=(?, 400)


        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        # pred_title = tf.reduce_mean(y, axis=1)
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
        embedded_sequences_body = embedding_layer(sequences_input_body)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_body)
        y = layers.Dense(400)(y)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [y] * 3
        )
        # with tf.name_scope("multi_head_attention"):
        #     multihead_attention_outputss= self_multi_head_attn_v2(y,num_units=400, num_heads=4,dropout_rate=0,is_training=True)
        #     for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
        #         multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs_v2,200*4,activation=tf.nn.relu)
        #         multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs3,200*2)
        #         multihead_attention_outputs_v2 = multihead_attention_outputs3+multihead_attention_outputs_v2
        # # y = layers.Dropout(hparams.dropout)(y)
        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        # pred_body = tf.reduce_mean(y,axis=1)
        # pred_body = tf.keras.backend.mean(y, axis=1)
        # pred_body = (lambda x:tf.reduce_mean(x,axis=1))(y)

        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)
        # pred_body = tf.reshape(pred_body,(-1,1,400))
        model = keras.Model(sequences_input_body, pred_body, name="body_encoder")
        return model

    def _build_vertencoder(self):
        """build vert encoder of NAML news encoder.

        Return:
            object: the vert encoder of NAML.
        """
        hparams = self.hparams
        input_vert = keras.Input(shape=(1,), dtype="int32")

        vert_embedding = layers.Embedding(
            hparams.vert_num, hparams.vert_emb_dim, trainable=True
        )

        vert_emb = vert_embedding(input_vert)
        pred_vert = layers.Dense(
            hparams.filter_num,
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(vert_emb)
        pred_vert = layers.Reshape((1, hparams.filter_num))(pred_vert)

        model = keras.Model(input_vert, pred_vert, name="vert_encoder")
        return model

    def _build_subvertencoder(self):
        """build subvert encoder of NAML news encoder.

        Return:
            object: the subvert encoder of NAML.
        """
        hparams = self.hparams
        input_subvert = keras.Input(shape=(1,), dtype="int32")

        subvert_embedding = layers.Embedding(
            hparams.subvert_num, hparams.subvert_emb_dim, trainable=True
        )

        subvert_emb = subvert_embedding(input_subvert)
        pred_subvert = layers.Dense(
            hparams.filter_num,
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(subvert_emb)
        pred_subvert = layers.Reshape((1, hparams.filter_num))(pred_subvert)

        model = keras.Model(input_subvert, pred_subvert, name="subvert_encoder")
        return model