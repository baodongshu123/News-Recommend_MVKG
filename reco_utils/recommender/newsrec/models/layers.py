import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Dense
class SelfAttentionUser(layers.Layer):
    def __init__(self, multiheads, head_dim, seed=0, mask_right=False, name=None, **kwargs):

        # self.multiheads = multiheads
        self.multiheads = 10
        self.head_dim = 40
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        self.self_name = name
        super(SelfAttentionUser, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape):
        self.WQ = self.add_weight(
            name="wq",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )  # (300,400)
        self.WK = self.add_weight(
            name="wk",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )  # (300,400)
        self.WV = self.add_weight(
            name="wv",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )  # (300,400)
        self.WDAXW = self.add_weight(
            name="wdaxw",
            shape=(self.head_dim,self.head_dim),
            initializer=keras.initializers.glorot_uniform(seed = self.seed),
            trainable=True,
        )
        super(SelfAttentionUser, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="add"):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, axis=1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs, **kwargs):
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        Q_seq_ori = QKVs[0]
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads, self.head_dim)
        )
        Q_seq = K.permute_dimensions(Q_seq, pattern=(0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim)
        )
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim)
        )  # (?,?,10,40)
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))  # (?,10,?,40)

        A = tf.matmul(Q_seq, K.permute_dimensions(K_seq,(0,1,3,2))) / K.sqrt(
            K.cast(self.head_dim, dtype="float32")
        )

        D = tf.matrix_diag([40]*40)
        D = tf.matrix_inverse(tf.cast(D, tf.float32))
        DA = tf.matmul(A,D)
        DAX = tf.matmul(DA,V_seq)
        DAXW = tf.matmul(DAX,self.WDAXW)
        O_seq = keras.layers.ReLU()(DAXW)

        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))  # (?,?,10,40)

        O_seq = K.reshape(O_seq, shape=(-1, K.shape(O_seq)[1], self.output_dim))  # (?,?,400)
        # O_seq = O_seq + Q_seq_ori
        O_seq = self.Mask(O_seq, Q_len, "mul")  # (?,?,400)
        return O_seq

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config

class Reshape_tensor(layers.Layer):
    def __init__(self,filter_num,**kwargs):
        self.filter_num = filter_num
        super(Reshape_tensor, self).__init__(**kwargs)
    def build(self, input_shape):
        return input_shape
    def call(self, inputs, **kwargs):
        return keras.layers.Reshape((1, self.filter_num))(inputs)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1,input_shape[-1])

class Te_MAt(layers.Layer):
    def __init__(self,entity_embedding_layer,**kwargs):
        self.entity_embedding_layer = entity_embedding_layer
        self.dim = 400
        super(Te_MAt, self).__init__(**kwargs)
    def build(self, input_shape):
        self.dense200 = Dense(self.dim,activation='tanh')
        self.dense2002 = Dense(self.dim,activation='tanh')
        self.dense1 = Dense(1)
        return input_shape
    def call(self, inputs, **kwargs):
        title_input = inputs[0]  # (?,30,400)
        entity_input = inputs[1]  # (?,10)
        entity_input = keras.backend.cast(entity_input,"float32")
        entity_emb = self.entity_embedding_layer(entity_input)  # (?,10,100)
        entity_dense = self.dense200(entity_emb)  # (?,10,200)
        entity_att = keras.layers.Dot(axes=-1)([entity_dense, entity_dense])
        entity_softmax = keras.layers.Softmax()(keras.layers.Dense(1)(entity_att))
        entity_self_repr = entity_softmax * entity_dense  # (?,10,200)
        title_input = self.dense2002(title_input)  # (?,30,200)
        cross_att = keras.layers.Dot(axes=-1)([title_input, entity_self_repr])  # (?,30,10)
        cross_att_entity = keras.layers.Softmax()(cross_att)  # (?,30,10)
        cross_att_entity = keras.layers.Dot(axes=[-1, -2])([cross_att_entity, entity_self_repr])  # (?,30,200)
        cross_att_entity = cross_att_entity + title_input
        cross_att_title = keras.layers.Softmax()(
            keras.layers.Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(cross_att))
        cross_att_title = keras.layers.Dot(axes=[-1, -2])([cross_att_title, title_input])  # (?,10,200)
        repr = keras.layers.Concatenate(axis=-2)([cross_att_title, cross_att_entity])  # (?,40,200)
        return repr
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1]+input_shape[1][1],self.dim)

class AttentivePooling(layers.Layer):
    def __init__(self,seed,**kwargs):
        self.seed = seed
        super(AttentivePooling, self).__init__(**kwargs)
    def build(self, input_shape):
        super(AttentivePooling, self).build(input_shape)
    def call(self, inputs, **kwargs):
        user_att = Dense(400, activation='tanh')(inputs)
        user_att = keras.layers.Flatten()(Dense(1)(user_att))
        user_att = keras.layers.Activation('softmax')(user_att)
        user_vec = keras.layers.Dot((1, 1))([inputs, user_att])
        user_vec = keras.layers.Reshape([1,400])(user_vec)
        return user_vec
    def compute_output_shape(self, input_shape):
        return (input_shape[0],tf.Dimension(1),input_shape[2])
    def get_config(self):
        config = super(AttentivePooling, self).get_config()
        config.update({"seed":self.seed})
        return config

class MAt(layers.Layer):
    def __init__(self,attention_hidden_dim,seed,**kwargs):
        self.attention_hidden_dim = attention_hidden_dim
        self.seed = seed
        super(MAt, self).__init__(**kwargs)

    def build(self, input_shape):#(?,30,300)-(?,None,None,400)
        self.dense = Dense(400,activation='tanh')
        self.dense1 = Dense(400,activation='tanh')
        super(MAt, self).build(input_shape)

    def call(self, inputs, **kwargs):
      title_input = inputs[0]
      abstract_input = inputs[1]
      repr = keras.layers.Concatenate(axis=1)([title_input, abstract_input])
      entity_vecs = SelfAttention(20,20,seed=0)([repr, repr, repr])
      entity_vecs1 = keras.layers.Dropout(0.2)(entity_vecs)

      return entity_vecs1
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1]+input_shape[1][1],self.attention_hidden_dim*2)


class Attention(layers.Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head  # 20
        self.size_per_head = size_per_head  # 20
        self.output_dim = nb_head * size_per_head  # 400
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x

        Q_seq1 = K.dot(Q_seq, self.WQ)
        Q_seq1 = K.reshape(Q_seq1, (-1, K.shape(Q_seq1)[1], self.size_per_head))
        Q_seq1 = K.permute_dimensions(Q_seq1, (0, 1, 2))
        K_seq1 = K.dot(K_seq, self.WK)
        K_seq1 = K.reshape(K_seq1, (-1, K.shape(K_seq1)[1], self.size_per_head))
        K_seq1 = K.permute_dimensions(K_seq1, (0, 1,2))
        V_seq1 = K.dot(V_seq, self.WV)
        V_seq1 = K.reshape(V_seq1, (-1, K.shape(V_seq1)[1], self.size_per_head))
        V_seq1 = K.permute_dimensions(V_seq1, (0, 1,2))
        A = K.batch_dot(Q_seq1, K_seq1, axes=[2, 2]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 2, 1))
        A = K.softmax(A)
        O_seq1 = K.batch_dot(A, V_seq1, axes=[2,1])
        O_seq1 = K.permute_dimensions(O_seq1, (0,1,2))
        O_seq1 = K.reshape(O_seq1, (-1, K.shape(O_seq1)[1], self.output_dim))
        O_seq1 = self.Mask(O_seq1, Q_len, 'mul')
        Q_seq2 = K.dot(Q_seq, self.WQ)
        Q_seq2 = K.reshape(Q_seq2, (-1, K.shape(Q_seq2)[1], self.size_per_head))
        Q_seq2 = K.permute_dimensions(Q_seq2, (0, 1, 2))
        K_seq2 = K.dot(K_seq, self.WK)
        K_seq2 = K.reshape(K_seq2, (-1, K.shape(K_seq2)[1], self.size_per_head))
        K_seq2 = K.permute_dimensions(K_seq2, (0, 1, 2))
        V_seq2 = K.dot(V_seq, self.WV)
        V_seq2 = K.reshape(V_seq2, (-1, K.shape(V_seq2)[1], self.size_per_head))
        V_seq2 = K.permute_dimensions(V_seq2, (0, 1, 2))
        A = K.batch_dot(Q_seq2, K_seq2, axes=[2, 2]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 2, 1))
        A = K.softmax(A)
        O_seq2 = K.batch_dot(A, V_seq2, axes=[2, 1])
        O_seq2 = K.permute_dimensions(O_seq2, (0, 1, 2))
        O_seq2 = K.reshape(O_seq2, (-1, K.shape(O_seq2)[1], self.output_dim))
        O_seq2 = self.Mask(O_seq2, Q_len, 'mul')
        res = keras.layers.concatenate(axis=1,inputs=[K.expand_dims(O_seq1,axis=1),K.expand_dims(O_seq2,axis=1)])
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], tf.Dimension(2),input_shape[0][1], self.output_dim)
class AGATLayer(keras.layers.Layer):
    def __init__(self,entity_size,neighbor_size,seed = 0,name="AGATLayer",**kwargs):
        self.entity_size = entity_size
        self.neighbor_size = neighbor_size
        self.seed = seed
        self.att_name = name


        super(AGATLayer,self).__init__(**kwargs)

    def build(self, input_shape):

        super(AGATLayer,self).build(input_shape)

    def call(self, inputs, **kwargs):

        entity_input = inputs[0]
        neighbors_input = inputs[1]
        # entity_input and neighbor_input
        entity = keras.backend.cast(entity_input, dtype="float32")
        neighbors = keras.backend.cast(neighbors_input, dtype="float32")

        entity_ed = keras.backend.expand_dims(entity, axis=-1)
        neighbors_ed = keras.backend.expand_dims(neighbors, axis=-1)
        a_uk = keras.layers.Dot(axes=-1)([entity_ed, neighbors_ed])
        e_a = keras.backend.mean(a_uk, axis=-1)
        n_a = keras.backend.mean(a_uk, axis=-2)
        e_as = keras.layers.Softmax(axis=-1)(e_a)
        e_dense = keras.backend.tanh(e_as)
        entity_concat = keras.layers.multiply([e_dense, entity]) + entity
        n_as = keras.layers.Softmax(axis=-1)(n_a)
        n_dense = keras.backend.tanh(n_as)
        neighbors_concat = keras.layers.multiply([n_dense, n_as]) + neighbors
        entity_repr_concat = keras.layers.concatenate([entity,entity_concat,entity*entity_concat,entity-entity_concat])
        neighbors_repr_concat = keras.layers.concatenate([neighbors,neighbors_concat,neighbors*neighbors_concat,neighbors-neighbors_concat])
        entity_reshape = keras.layers.Reshape([-1,self.entity_size])(entity_repr_concat)
        neighbors_reshape = keras.layers.Reshape([-1,self.neighbor_size])(neighbors_repr_concat)
        entity_repr = keras.backend.mean(entity_reshape,axis=-2)
        neighbors_repr = keras.backend.mean(neighbors_reshape, axis=-2)
        e_n_result = keras.layers.Concatenate(axis=-1)([entity_repr, neighbors_repr])
        return e_n_result

    def compute_output_shape(self, input_shape):
        return input_shape[0],input_shape[-1]
class AttLayer2(layers.Layer):
    """Soft alignment attention implement.

    Attributes:
        dim (int): attention hidden dim
    """
    def __init__(self, dim=200, seed=0, name="AttLayer", gamma_initializer = "ones",**kwargs):
        """Initialization steps for AttLayer2.
        
        Args:
            dim (int): attention hidden dim
        """

        self.dim = dim
        self.seed = seed
        self.att_name = name
        super(AttLayer2, self).__init__(**kwargs)
        self.gamma = self.add_weight(
            shape=(),
            initializer=gamma_initializer,
            trainable=True,
            name="gamma",
            dtype=self.dtype,
        )

    def build(self, input_shape):
        """Initialization for variables in AttLayer2
        There are there variables in AttLayer2, i.e. W, b and q.

        Args:
            input_shape (object): shape of input tensor.
        """

        assert len(input_shape) == 3
        dim = self.dim
        self.W = self.add_weight(
            name=self.att_name+"att_w",
            shape=(int(input_shape[-1]), dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name=self.att_name+"att_b",
            shape=(dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name=self.att_name+"att_q",
            shape=(dim, 1),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        super(AttLayer2, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        """Core implemention of soft attention

        Args:
            inputs (object): input tensor.

        Returns:
            object: weighted sum of input tensors.
        """

        attention = K.tanh(K.dot(inputs, self.W) + self.b)
        attention = K.dot(attention, self.q)

        attention = K.squeeze(attention, axis=2)

        if mask == None:
            attention = K.exp(attention)
        else:
            attention = K.exp(attention) * K.cast(mask, dtype="float32")

        attention_weight = attention / (
            K.sum(attention, axis=-1, keepdims=True) + K.epsilon()
        )

        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight

        weighted_input=self.gamma*weighted_input

        return K.sum(weighted_input, axis=1)


    def compute_mask(self, input, input_mask=None):
        """Compte output mask value

        Args: 
            input (object): input tensor.
            input_mask: input mask
        
        Returns:
            object: output mask.
        """
        return None

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor

        Args:
            input_shape (tuple): shape of input tensor.
        
        Returns:
            tuple: shape of output tensor.
        """
        return input_shape[0], input_shape[-1]
    def get_config(self):
        """ add multiheads, multiheads and mask_right into layer config.

        Returns:
            dict: config of SelfAttention layer.
        """
        config = super(AttLayer2, self).get_config()
        config.update(
            {
                "dim": self.dim
            }
        )
        return config


class SelfAttention(layers.Layer):
    """Multi-head self attention implement.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimention of each head.
        mask_right (boolean): whether to mask right words.

    Returns:
        object: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False,name=None,**kwargs):
        """Initialization steps for AttLayer2.
        
        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimention of each head.
            mask_right (boolean): whether to mask right words.
        """

        self.multiheads = 10
        self.head_dim = 40
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        self.self_name = name
        super(SelfAttention, self).__init__(**kwargs)


    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Returns:
            tuple: output shape tuple.
        """

        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape):
        """Initialization for variables in SelfAttention.
        There are three variables in SelfAttention, i.e. WQ, WK ans WV.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.

        Args:
            input_shape (object): shape of input tensor.
        """

        self.WQ = self.add_weight(
            name="wq",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WK = self.add_weight(
            name="wk",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WV = self.add_weight(
            name="wv",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        super(SelfAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): sequence length of inputs.
            mode (str): mode of mask.
        
        Returns:
            object: tensors after masking.
        """

        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self,QKVs,**kwargs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. qeury, key and value.

        Returns:
            object: ouput tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads, self.head_dim)
        )
        Q_seq = K.permute_dimensions(Q_seq, pattern=(0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim)
        )
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim)
        )
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))

        A = tf.matmul(Q_seq, K.permute_dimensions(K_seq,(0,1,3,2)))
        A = K.permute_dimensions(
            A, pattern=(0, 3, 2, 1)
        )

        A = self.Mask(A, V_len, "add")
        A = K.permute_dimensions(A, pattern=(0, 3, 2, 1))

        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1, num_upper=0)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = K.softmax(A)
        O_seq = tf.matmul(A, V_seq)
        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))

        O_seq = K.reshape(O_seq, shape=(-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")
        return O_seq

    def get_config(self):
        """ add multiheads, multiheads and mask_right into layer config.

        Returns:
            dict: config of SelfAttention layer.  
        """
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config



def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
    """Soft alignment attention implement.

    Attributes:
        dim1 (int): first dimention of value shape.
        dim2 (int): second dimention of value shape.
        dim3 (int): shape of query
    
    Returns:
        object: weighted summary of inputs value.
    """

    vecs_input = keras.Input(shape=(dim1, dim2))
    query_input = keras.Input(shape=(dim3,))

    user_vecs = vecs_input
    user_att = layers.Dense(
        dim3,
        activation="tanh",
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
        bias_initializer=keras.initializers.Zeros(),
    )
    user_att = user_att((user_vecs))
    user_att2 = layers.Dot(axes=-1)([query_input, user_att])#(?,50)
    user_att_4 = layers.Activation("softmax")(user_att2)
    user_vec = layers.Dot((1, 1))([user_vecs, user_att_4])

    model = keras.Model([vecs_input, query_input], user_vec)
    return model

    '''
    dim1:30
    dim2:300
    dim3:50
    '''
def ChannelAttention(dim1,dim2,r=5,seed=0):
    vecs_input = keras.Input(shape=(dim1, dim2), dtype="float32")
    user_vecs = layers.Dropout(0.2)(vecs_input)
    globbal_vector = keras.backend.mean(user_vecs,2,keepdims=True)
    u_input = layers.Reshape((1,dim1))(globbal_vector)
    fc1_avg = layers.Dense(
        int(dim1/r),
        activation="relu",
    )(u_input)
    fc2_avg = layers.Dense(
        dim1
    )(fc1_avg)

    globbal_vector2 = keras.backend.max(user_vecs,2,keepdims=True)
    u_input = layers.Reshape((1,dim1))(globbal_vector2)
    fc1_max = layers.Dense(
        int(dim1 / r),
        activation="relu"
    )(u_input)
    fc2_max = layers.Dense(
        dim1
    )(fc1_max)
    fc_sum = fc2_avg + fc2_max
    fc2 = layers.Activation("sigmoid")(fc_sum)
    fc2 = layers.Reshape((dim1, 1))(fc2)
    user_vec = layers.multiply([fc2,vecs_input])
    model = keras.Model(vecs_input,user_vec)
    return model

class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.

    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = K.not_equal(inputs, 0)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape


class OverwriteMasking(layers.Layer):
    """Set values at spasific positions to zero.

    Args:
        inputs (list): value tensor and mask tensor.
    
    Returns:
        object: tensor after setting values to zero.
    """

    def __init__(self, **kwargs):
        super(OverwriteMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OverwriteMasking, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[0] * K.expand_dims(inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]
