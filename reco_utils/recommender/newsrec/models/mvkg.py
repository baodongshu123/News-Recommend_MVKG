import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import AttLayer2, SelfAttention,\
    PersonalizedAttentivePooling,AGATLayer,Attention,AttentivePooling,MAt,Te_MAt,Reshape_tensor,SelfAttentionUser
__all__ = ["MVKGModel"]


class MVKGModel(BaseModel):
    def __init__(
        self, hparams, iterator_creator, seed=None,
    ):
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.entitty2vec_embedding = self._init_embedding(hparams.entityEmb_file)

        super().__init__(
            hparams, iterator_creator, seed=seed,
        )

    def _get_input_label_from_iter(self, batch_data):
       
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_title_entity_batch"],
            batch_data["clicked_title_neighbor_batch"],
            batch_data["clicked_abstract_entity_batch"],
            batch_data["clicked_abstract_neighbor_batch"],
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_title_entity_batch"],
            batch_data["candidate_title_neighbor_batch"],
            batch_data["candidate_abstract_entity_batch"],
            batch_data["candidate_abstract_neighbor_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
       
        input_feature = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_title_entity_batch"],
            batch_data["clicked_title_neighbor_batch"],
            batch_data["clicked_abstract_entity_batch"],
            batch_data["clicked_abstract_neighbor_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _get_news_feature_from_iter(self, batch_data):
       
        input_feature = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_title_entity_batch"],
            batch_data["candidate_title_neighbor_batch"],
            batch_data["candidate_abstract_entity_batch"],
            batch_data["candidate_abstract_neighbor_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _build_graph(self):
      
        hparams = self.hparams
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_nrms(self):
        
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        his_input_body = keras.Input(
            shape=(hparams.his_size, hparams.body_size), dtype="int32"
        )
        his_input_vert = keras.Input(shape=(hparams.his_size, 1), dtype="int32")
        his_input_title_entity = keras.Input(shape=(hparams.his_size, hparams.entity_size), dtype="int32")
        his_input_title_neighbor = keras.Input(shape=(hparams.his_size, hparams.neighbor_size), dtype="int32")
        his_input_abstract_entity = keras.Input(shape=(hparams.his_size, hparams.entity_size), dtype="int32")
        his_input_abstract_neighbor = keras.Input(shape=(hparams.his_size, hparams.neighbor_size), dtype="int32")

        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_body = keras.Input(
            shape=(hparams.npratio + 1, hparams.body_size), dtype="int32"
        )
        pred_input_vert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")
        pred_input_title_entity = keras.Input(shape=(hparams.npratio + 1, hparams.entity_size), dtype="int32")
        pred_input_title_neighbor = keras.Input(shape=(hparams.npratio + 1, hparams.neighbor_size), dtype="int32")
        pred_input_abstract_entity = keras.Input(shape=(hparams.npratio + 1, hparams.entity_size), dtype="int32")
        pred_input_abstract_neighbor = keras.Input(shape=(hparams.npratio + 1, hparams.neighbor_size), dtype="int32")

        pred_input_title_one = keras.Input(
            shape=(hparams.title_size,), dtype="int32"
        )
        pred_input_body_one = keras.Input(shape=(hparams.body_size,), dtype="int32")
        pred_input_vert_one = keras.Input(shape=(1), dtype="int32")
        pred_input_title_entity_one = keras.Input(shape=(hparams.entity_size,), dtype="int32")
        pred_input_title_neighbor_one = keras.Input(shape=(hparams.neighbor_size,), dtype="int32")
        pred_input_abstract_entity_one = keras.Input(shape=(hparams.entity_size,), dtype="int32")
        pred_input_abstract_neighbor_one = keras.Input(shape=(hparams.neighbor_size,), dtype="int32")

        his_title_body_verts = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_body, his_input_vert, his_input_title_entity, his_input_title_neighbor,
             his_input_abstract_entity, his_input_abstract_neighbor]
        )

        pred_title_body_verts = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_body, pred_input_vert, pred_input_title_entity, pred_input_title_neighbor,
             pred_input_abstract_entity, pred_input_abstract_neighbor]
        )

        pred_input_title_one_reshape = layers.Reshape((1, hparams.title_size))(pred_input_title_one)
        pred_input_body_one_reshape = layers.Reshape((1, hparams.body_size))(pred_input_body_one)
        pred_input_vert_one_reshape = layers.Reshape((1, 1))(pred_input_vert_one)
        pred_input_title_entity_one_reshape = layers.Reshape((1, hparams.entity_size))(pred_input_title_entity_one)
        pred_input_title_neighbor_one_reshape = layers.Reshape((1, hparams.neighbor_size))(
            pred_input_title_neighbor_one)
        pred_input_abstract_entity_one_reshape = layers.Reshape((1, hparams.entity_size))(pred_input_title_entity_one)
        pred_input_abstract_neighbor_one_reshape = layers.Reshape((1, hparams.neighbor_size))(
            pred_input_title_neighbor_one)
        pred_title_body_verts_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one_reshape,
                pred_input_body_one_reshape,
                pred_input_vert_one_reshape,
                pred_input_title_entity_one_reshape,
                pred_input_title_neighbor_one_reshape,
                pred_input_abstract_entity_one_reshape,
                pred_input_abstract_neighbor_one_reshape
            ]
        )

        pred_title_one_reshape = keras.backend.squeeze(pred_title_body_verts_one, axis=1)
        # word embedding_layer
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

        titleencoder = self._build_newsencoder(embedding_layer, entity_embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        user_present = self.userencoder(his_title_body_verts)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_verts)
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
                his_input_title_entity,
                his_input_title_neighbor,
                his_input_abstract_entity,
                his_input_abstract_neighbor,
                pred_input_title,
                pred_input_body,
                pred_input_vert,
                pred_input_title_entity,
                pred_input_title_neighbor,
                pred_input_abstract_entity,
                pred_input_abstract_neighbor,
            ],
            preds,
        )

        scorer = tf.keras.Model(
            [
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_title_entity,
                his_input_title_neighbor,
                his_input_abstract_entity,
                his_input_abstract_neighbor,
                pred_input_title_one,
                pred_input_body_one,
                pred_input_vert_one,
                pred_input_title_entity_one,
                pred_input_title_neighbor_one,
                pred_input_abstract_entity_one,
                pred_input_abstract_neighbor_one,
            ],
            pred_one,
        )

        return model, scorer
    def _build_userencoder(self, titleencoder):
        
        hparams = self.hparams
        his_input_title_body_verts = keras.Input(
            shape=(hparams.his_size, hparams.title_size + hparams.body_size + 1 + hparams.entity_size + hparams.neighbor_size + hparams.entity_size + hparams.neighbor_size),
            dtype="int32",
        )
        #(?,50,82)-->(?,50,400)
        click_news_presents = layers.TimeDistributed(titleencoder)(
            his_input_title_body_verts
        )
        
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(
            click_news_presents
        )
        print(user_present)
        model = keras.Model(
            his_input_title_body_verts,user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer,entity_embedding_layer):

        hparams = self.hparams
        input_title_body_verts = keras.Input(shape=(hparams.title_size + hparams.body_size + 1 + hparams.entity_size + hparams.neighbor_size+ hparams.entity_size + hparams.neighbor_size,), dtype="int32")
        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(input_title_body_verts)
        sequences_input_body = layers.Lambda(
            lambda x: x[:, hparams.title_size: hparams.title_size + hparams.body_size])(input_title_body_verts)
        input_vert = layers.Lambda(
            lambda x: x[:,hparams.title_size+ hparams.body_size: hparams.title_size+ hparams.body_size+ 1,])(input_title_body_verts)
        sequence_input_title_entity = layers.Lambda(
            lambda x: x[:, hparams.title_size + hparams.body_size + 1:hparams.title_size + hparams.body_size + 1 + hparams.entity_size])(input_title_body_verts)
        sequence_input_title_neighbor = layers.Lambda(
            lambda x: x[:, hparams.title_size + hparams.body_size + 1 + hparams.entity_size:hparams.title_size + hparams.body_size + 1 + hparams.entity_size+hparams.neighbor_size])(input_title_body_verts)
        sequence_input_abstract_entity = layers.Lambda(
            lambda x: x[:,hparams.title_size + hparams.body_size + 1 + hparams.entity_size+hparams.neighbor_size:hparams.title_size + hparams.body_size + 1 + hparams.entity_size+hparams.neighbor_size+hparams.entity_size])(input_title_body_verts)
        sequence_input_abstract_neighbor = layers.Lambda(
            lambda x: x[:,hparams.title_size + hparams.body_size + 1 + hparams.entity_size + hparams.neighbor_size+hparams.entity_size:hparams.title_size + hparams.body_size + 1 + hparams.entity_size + hparams.neighbor_size + hparams.entity_size+hparams.neighbor_size])(input_title_body_verts)

        title_repr,title_cnn = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr,body_mh = self._build_bodyencoder(embedding_layer)(sequences_input_body)
        vert_repr = self._build_vertencoder()(input_vert)
        mat = MAt(attention_hidden_dim=hparams.attention_hidden_dim,seed=self.seed)([title_cnn,body_mh])
        from tensorflow.keras.layers import Concatenate  
        h1_title = Te_MAt(entity_embedding_layer=entity_embedding_layer)([mat, sequence_input_title_entity]) 
        h2_title = Te_MAt(entity_embedding_layer=entity_embedding_layer)([mat, sequence_input_title_neighbor])  
        combined_title = Concatenate(axis=1)([h1_title, h2_title])  
        title_en_ne_att = AttLayer2(hparams.filter_num, seed=self.seed)(combined_title)  
        title_en_ne_repr = Reshape_tensor(hparams.filter_num)(title_en_ne_att) 
        h1_abstract = Te_MAt(entity_embedding_layer=entity_embedding_layer)([mat, sequence_input_abstract_entity])  
        h2_abstract = Te_MAt(entity_embedding_layer=entity_embedding_layer)([mat, sequence_input_abstract_neighbor])  
        combined_abstract = Concatenate(axis=1)([h1_abstract, h2_abstract])  
        abstract_en_ne_att = AttLayer2(hparams.filter_num, seed=self.seed)(combined_abstract)  
        abstract_en_ne_repr = Reshape_tensor(hparams.filter_num)(abstract_en_ne_att)  

        concate_repr = layers.Concatenate(axis=-2)(
            [title_repr, body_repr, vert_repr,title_en_ne_repr,abstract_en_ne_repr]
        )
      
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(concate_repr)
        model = keras.Model(input_title_body_verts, pred_title, name="news_encoder")
        return model
    


    def _build_titleencoder(self, embedding_layer):
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y_1 = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y_1)
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, hparams.filter_num))(pred_title)

        model = tf.keras.Model(sequences_input_title, [pred_title,y_1], name="title_encoder")
        return model

    def _build_bodyencoder(self, embedding_layer):
        hparams = self.hparams
        sequences_input_body = keras.Input(shape=(hparams.body_size,), dtype="int32")
        embedded_sequences_body = embedding_layer(sequences_input_body)
        y = layers.Dropout(hparams.dropout)(embedded_sequences_body)
        y_1 = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [y] * 3
        )
        y_1 = layers.Dropout(hparams.dropout)(y_1)
        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y_1)
        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)

        model = tf.keras.Model(sequences_input_body, [pred_body,y], name="body_encoder")
        return model


    def _build_vertencoder(self):
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

        model = tf.keras.Model(input_vert, pred_vert, name="vert_encoder")
        return model
