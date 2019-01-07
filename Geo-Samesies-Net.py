#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Toponym Net
from keras import Input, Model
from keras.layers import Dense, Dropout, GRU, Bidirectional, GlobalMaxPooling1D
from keras.layers import dot, concatenate, multiply, Lambda, Layer, Masking, Permute, Activation, Multiply, Add
from keras.layers.core import Masking
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras import initializers
import numpy as np
# from sklearn.model_selection import train_test_split
import pandas as pd


# In[2]:


# Import Data

# Subset of the data (10000 record)
# names_df = pd.read_csv('test.tsv', 
#             sep='\t', 
#             names = ["name_1", "name_2", "is_same", "id_1", "id_2", "lang_1", "lang_2", "country_1", "country_2"])

# names_df = pd.read_csv('Toponym-Matching/dataset/dataset-string-similarity.txt', 
#             sep='\t', 
#             names = ["name_1", "name_2", "is_same", "id_1", "id_2", "lang_1", "lang_2", "country_1", "country_2"])

# names_df.to_csv('dataset.tsv', sep='\t', index=False)


names_df = pd.read_csv('dataset.tsv', sep='\t', header=0)


# In[3]:


names_df = names_df.replace(np.nan, '', regex=True)
len(names_df)


# In[4]:


# Check for unparsible values
for name in names_df['name_2'].tolist():
    if isinstance(name, float):
        print(name)


# In[28]:


names_df = names_df[['name_1', 'name_2', 'is_same']]
# Prepare Data
x_1 = np.array([np.array(list(bytearray(name.lower(), encoding='utf-8'))) for name in names_df['name_1'].tolist()[0:100000]])
x_2 = np.array([np.array(list(bytearray(name.lower(), encoding='utf-8'))) for name in names_df['name_2'].tolist()[0:100000]])

maxlen = max(len(max(x_1,key=len)), len(max(x_2,key=len)))
len_chars = 255  # Max Byte value
numb_examples = len(x_1)


# In[37]:


X_1 = sequence.pad_sequences(x_1, maxlen=maxlen)
X_1 = to_categorical(X_1, dtype='float16')
# X_1 = X_1.reshape(numb_examples, maxlen, 1)
_, _, len_chars = X_1.shape
print('len_chars', len_chars)

X_2 = sequence.pad_sequences(x_2, maxlen=maxlen)
X_2 = to_categorical(X_2, dtype='float16')
# X_2 = X_2.reshape(numb_examples, maxlen, 1)

y = np.array([0 if val else 1 for val in names_df['is_same']][0:100000])  # 0 for True since the names are close


# In[19]:


# Train Test Split
# X = np.array(list(zip(X_1, X_2)))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# X_1_train, X_2_train = X_train[:, 0], X_train[:, 1]
# print('Training', X_1_train.shape)
# print('Labels', y_train.shape)
# X_1_test, X_2_test = X_test[:, 0], X_test[:, 1]
# print('Validation', X_1_test.shape)
# print('Labels', y_test.shape)


# In[31]:


class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalMaxPooling1DMasked, self).__init__(**kwargs)
    def build(self, input_shape): super(GlobalMaxPooling1DMasked, self).build(input_shape)
    def call(self, x, mask=None): return super(GlobalMaxPooling1DMasked, self).call(x)


# In[32]:


class Attention(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(Attention, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# In[33]:


def AlignmentAttention(input_1, input_2):
    def unchanged_shape(input_shape): return input_shape
    def softmax(x, axis=-1):
        ndim = K.ndim(x)
        if ndim == 2: return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else: raise ValueError('Cannot apply softmax to a tensor that is 1D')
    w_att_1 = dot([input_1, input_2], axes=-1)
    w_att_1 = Lambda(lambda x: softmax(x, axis=1), output_shape=unchanged_shape)(w_att_1)
    
    w_att_2 = dot([input_1, input_2], axes=-1)
    w_att_2 = Lambda(lambda x: softmax(x, axis=2), output_shape=unchanged_shape)(w_att_2)
    w_att_2 = Permute((2,1))(w_att_2)

    in1_aligned = dot([w_att_1, input_1], axes=1)
    in2_aligned = dot([w_att_2, input_2], axes=1)
    
    q1_combined = concatenate([input_1, in2_aligned])
    q2_combined = concatenate([input_2, in1_aligned])
    return q1_combined, q2_combined


# In[34]:


def siamese_net(max_len, len_chars, hidden_units=60):
    characters_1 = Input(shape=(maxlen, len_chars, ))
    characters_2 = Input(shape=(maxlen, len_chars, ))
    
    gru1 = Bidirectional(GRU(hidden_units, implementation=2, return_sequences=True))
    gru2 = Bidirectional(GRU(hidden_units, implementation=2, return_sequences=True))
    
    # Left Branch
    left_branch = Masking(mask_value=0, input_shape=(max_len, len_chars))(characters_1)
    # shortcut
    left_branch_aux1 = gru1(left_branch)
    left_branch_aux2 = concatenate([left_branch, left_branch_aux1])
    left_branch = left_branch_aux2
    # dropout
    left_branch = Dropout(0.01)(left_branch)
    left_branch = gru2(left_branch)
    left_branch = Dropout(0.01)(left_branch)
    
    # Right Branch
    right_branch = Masking(mask_value=0, input_shape=(max_len, len_chars))(characters_2)
    # shortcut
    right_branch_aux1 = gru1(right_branch)
    right_branch_aux2 = concatenate([right_branch, right_branch_aux1])
    right_branch = right_branch_aux2
    # dropout
    right_branch = Dropout(0.01)(right_branch)
    right_branch = gru2(right_branch)
    right_branch = Dropout(0.01)(right_branch)
    
    # Alignment Mechanism
    left_branch, right_branch = AlignmentAttention(left_branch, right_branch)
    
    # Attention Mechanism
    attention = Attention(100)
    left_branch = attention(left_branch)
    right_branch = attention(right_branch)

    # Combine Branches
    concat_layer = concatenate([left_branch, right_branch])
    multiply_layer = multiply([left_branch, right_branch])
    diff_layer = Lambda(lambda x: x[0] - x[1], output_shape=lambda x: x[0])([left_branch, right_branch])
    
    # Final Layer
    final = concatenate([concat_layer, multiply_layer, diff_layer])
    final = Dropout(0.01)(final)
    final = Dense(hidden_units, activation='relu')(final)
    final = Dropout(0.01)(final)
    final = Dense(1, activation='sigmoid')(final)
    siamese_net = Model(inputs=[characters_1, characters_2], outputs=final)
    return siamese_net


# In[35]:


siamese_model = siamese_net(maxlen, len_chars)
siamese_model.summary()


# In[36]:


from keras.callbacks import ModelCheckpoint
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
siamese_model.fit(x=[X_1, X_2], 
                  y=y, 
                  epochs=20,
                  validation_split=0.01,
                 callbacks=[ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')])


# In[93]:


# TEST
# 'Salhon','La Salencon Rivière',1
# 'Abḏul','Sēyyed Gholām `Abd ol',0
# 'aibrahym abad','aabramabad',1
# 'jiu yuan gou cun','Jiuyuan Goucun',0
# 'Assaisse','Assaki (CR)',1


# x_test_1 = np.array([np.array(list(bytearray('Assaisse', encoding='utf-8')))])
# x_test_1 = sequence.pad_sequences(x_test_1, maxlen=maxlen).reshape(1, maxlen, )
# x_test_1 = to_categorical(x_test_1, num_classes=255, dtype='float32')
# x_test_2 = np.array([np.array(list(bytearray('Assaki (CR)', encoding='utf-8')))])
# x_test_2 = sequence.pad_sequences(x_test_2, maxlen=maxlen).reshape(1, maxlen, )
# x_test_2 = to_categorical(x_test_2, num_classes=255, dtype='float32')
# siamese_model.predict(x=[x_test_1, x_test_2])


# In[ ]:




