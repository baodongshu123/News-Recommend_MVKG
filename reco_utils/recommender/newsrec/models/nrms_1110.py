# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import AttLayer2, SelfAttention,PersonalizedAttentivePooling

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
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
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
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
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
        # model, scorer = self._build_nrms()
        # return model, scorer
        model = self._build_nrms()
        return model

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS. 

        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title_body_verts = keras.Input(
            shape=(hparams.his_size, hparams.title_size + hparams.body_size + 2),
            dtype="int32",
        )
        userandnews_present = keras.Input(
            shape=(hparams.npratio + 1, hparams.filter_num),dtype="float32"
        )
        #(?,50,82)-->(?,50,400)
        click_news_presents = layers.TimeDistributed(titleencoder)(
            his_input_title_body_verts
        )


        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            click_news_presents
        )

        #开始处理gru
        news_present_inner = userandnews_present
        short_uemb = layers.GRU(
            hparams.gru_unit,
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            bias_initializer=keras.initializers.Zeros(),
        )(layers.Masking(mask_value=0.0)(click_news_presents))

        user_present_21_pers = PersonalizedAttentivePooling(
            hparams.npratio + 1,
            hparams.filter_num,
            hparams.attention_hidden_dim,
            seed=self.seed,
        )([news_present_inner, layers.Dense(hparams.attention_hidden_dim)(short_uemb)])

        user_present_21_pers_reshape = layers.Reshape((1, hparams.filter_num))(user_present_21_pers)
        user_present_reshape = layers.Reshape((1, hparams.filter_num))(user_present)

        user_final = layers.Concatenate(axis=-2)([user_present_21_pers_reshape, user_present_reshape])
        user_present_final = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            user_final
        )


        model = keras.Model(
            [his_input_title_body_verts,userandnews_present], user_present_final, name="user_encoder"
        )
        return model


    def _build_newsencoder(self, embedding_layer,embedding_layer_b):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.
        
        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        title_size = hparams.title_size
        body_size = hparams.body_size
        input_title_body_verts = keras.Input(
            shape=(82,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : 30])(
            input_title_body_verts
        )
        sequences_input_body = layers.Lambda(
            lambda x: x[:, 30: 30 + 50]
        )(input_title_body_verts)
        input_vert = layers.Lambda(
            lambda x: x[
                      :,
                      30
                      + 50: 30
                                           + 50
                                           + 1,
                      ]
        )(input_title_body_verts)
        input_subvert = layers.Lambda(
            lambda x: x[:, 30 + 50 + 1:]
        )(input_title_body_verts)

        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr = self._build_bodyencoder(embedding_layer_b)(sequences_input_body)
        vert_repr = self._build_vertencoder()(input_vert)
        # subvert_repr = self._build_subvertencoder()(input_subvert)

        concate_repr = layers.Concatenate(axis=-2)(
            [title_repr, body_repr, vert_repr]
        )

        # y = layers.Dropout(hparams.dropout)(concate_repr)
        # y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        # y = layers.Dropout(hparams.dropout)(y)
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
        his_input_vert = keras.Input(shape=(hparams.his_size, 1), dtype="int32")
        his_input_subvert = keras.Input(shape=(hparams.his_size, 1), dtype="int32")
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_body = keras.Input(
            shape=(hparams.npratio + 1, hparams.body_size), dtype="int32"
        )
        pred_input_vert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")
        pred_input_subvert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")
        pred_input_title_one = keras.Input(
            shape=(hparams.title_size,), dtype="int32"
        )
        pred_input_body_one = keras.Input(shape=(hparams.body_size,), dtype="int32")
        pred_input_vert_one = keras.Input(shape=(hparams.vert_size), dtype="int32")
        pred_input_subvert_one = keras.Input(shape=(hparams.vert_size), dtype="int32")

        his_title_body_verts = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_body, his_input_vert, his_input_subvert]
        )

        pred_title_body_verts = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_body, pred_input_vert, pred_input_subvert]
        )

        pred_input_title_one_reshape = layers.Reshape((1, hparams.title_size))(pred_input_title_one)
        pred_input_body_one_reshape = layers.Reshape((1, hparams.body_size))(pred_input_body_one)
        pred_input_vert_one_reshape = layers.Reshape((1, 1))(pred_input_vert_one)
        pred_input_subvert_one_reshape = layers.Reshape((1, 1))(pred_input_subvert_one)

        pred_title_body_verts_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one_reshape,
                pred_input_body_one_reshape,
                pred_input_vert_one_reshape,
                pred_input_subvert_one_reshape,
            ]
        )



        pred_title_one_reshape = layers.Reshape((82,))(pred_title_body_verts_one)
        #word embedding_layer
        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
            name="Embedding"
        )
        embedding_layer_b = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
            name="Embedding_b"
        )

        titleencoder = self._build_newsencoder(embedding_layer,embedding_layer_b)
        self.newsencoder = titleencoder

        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_verts)
        self.userencoder = self._build_userencoder(titleencoder)
        user_present = self.userencoder([his_title_body_verts,news_present])


        news_present_one = self.newsencoder(pred_title_one_reshape)


        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = tf.keras.Model(
            [
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title,
                pred_input_body,
                pred_input_vert,
                pred_input_subvert,
            ],
            preds,
        )

        # scorer = tf.keras.Model(
        #     [
        #         his_input_title,
        #         his_input_body,
        #         his_input_vert,
        #         his_input_subvert,
        #         pred_input_title_one,
        #         pred_input_body_one,
        #         pred_input_vert_one,
        #         pred_input_subvert_one,
        #     ],
        #     pred_one,
        # )
        '''
        ValueError: Graph disconnected: cannot obtain value for tensor Tensor("input_5:0", shape=(?, 5, 30), dtype=int32) at layer "input_5". 
        The following previous layers were accessed without issue: ['input_12', 'input_11', 'input_10', 'input_9']
        '''

        return model

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
        r=5
        # globbal_vtteor = tf.reduce_mean(embeddings3, 2, keep_dims=True)
        # fc1_avg = tf.contrib.layers.fully_connected(inputs=tf.transpose(globbal_vtteor, perm=[0, 2, 1]),
        #                                             num_outputs=int(10 / r))
        # fc1_avg = tf.nn.relu(fc1_avg)
        # fc2_avg = tf.contrib.layers.fully_connected(inputs=fc1_avg, num_outputs=22)
        # max_vetor = tf.reduce_max(embeddings3, 2, keep_dims=True)
        # fc1_max = tf.contrib.layers.fully_connected(inputs=tf.transpose(max_vetor, perm=[0, 2, 1]),
        #                                             num_outputs=int(10 / r))
        # fc1_max = tf.nn.relu(fc1_max)
        # fc2_max = tf.contrib.layers.fully_connected(inputs=fc1_max, num_outputs=22)
        # fc_sum = fc2_avg + fc2_max
        # fc2 = tf.sigmoid(fc_sum)
        # residual = tf.multiply(tf.transpose(fc2, perm=[0, 2, 1]), embeddings3)
        globbal_vtteor = layers.Lambda(lambda x: tf.reduce_mean(x, axis=2, keepdims=True))(embedded_sequences_title)
        fc1_avg = layers.Dense(units=int(30 / r),
                               name="Dense_t1",
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                               activation="relu")(layers.Lambda(lambda y: tf.transpose(y, perm=[0, 2, 1]))(globbal_vtteor))
        fc2_avg = layers.Dense(units=30,
                               name="Dense_t2",
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(fc1_avg)

        max_vetor = layers.Lambda(lambda x: tf.reduce_max(x, axis=2, keepdims=True))(embedded_sequences_title)
        fc1_max = layers.Dense(units=int(30 / r),
                               name="Dense_t3",
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                               activation="relu")(layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(max_vetor))
        fc2_max = layers.Dense(units=30,
                               name="Dense_t4",
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(fc1_max)
        fc_sum = layers.Add()([fc2_avg, fc2_max])
        fc2 = layers.Lambda(lambda z: tf.sigmoid(z))(fc_sum)
        fc2_fin = (layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(fc2))
        # residual = layers.Lambda(lambda k: tf.multiply(k, embedded_sequences_title_test))(fc2_fin)
        residual = keras.layers.Multiply()([fc2_fin,embedded_sequences_title])

        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            name="Conv1D_1",
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(residual)

        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed,name="AttLayer_t")(y)
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
        r = 5

        globbal_vtteor = layers.Lambda(lambda x: tf.reduce_mean(x, axis=2, keepdims=True))(embedded_sequences_body)
        fc1_avg = layers.Dense(units=int(50 / r),
                               name="Dense_b1",
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                               activation="relu")(layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(globbal_vtteor))
        fc2_avg = layers.Dense(units=50,
                               name="Dense_b2",
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                               )(fc1_avg)
        max_vetor = layers.Lambda(lambda x: tf.reduce_max(x, axis=2, keepdims=True))(embedded_sequences_body)
        fc1_max = layers.Dense(units=int(50 / r),
                               name="Dense_b3",
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                               activation="relu")(
            layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(max_vetor))
        fc2_max = layers.Dense(units=50,
                               name="Dense_b4",
                               bias_initializer=keras.initializers.Zeros(),
                               kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                               )(fc1_max)
        fc_sum = layers.Add()([fc2_avg, fc2_max])
        fc2 = layers.Lambda(lambda x: tf.sigmoid(x))(fc_sum)
        fc2_fin = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(fc2)
        # residual = layers.Lambda(lambda x: tf.multiply(x, embedded_sequences_body))(
        #     layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(fc2))

        residual = keras.layers.Multiply()([fc2_fin, embedded_sequences_body])

        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed,name="SelfAttention_b")(
            [residual] * 3
        )
        # y = layers.Dropout(hparams.dropout)(y)
        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed,name="AttLayer_b")(y)
        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)

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
            hparams.vert_num, hparams.vert_emb_dim, trainable=True,name="Embedding_v"
        )

        vert_emb = vert_embedding(input_vert)
        pred_vert = layers.Dense(
            hparams.filter_num,
            name="Dense_v",
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(vert_emb)
        pred_vert = layers.Reshape((1, hparams.filter_num))(pred_vert)

        model = tf.keras.Model(input_vert, pred_vert, name="vert_encoder")
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

        model = tf.keras.Model(input_subvert, pred_subvert, name="subvert_encoder")
        return model

