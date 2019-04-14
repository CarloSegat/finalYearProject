import numpy as np
import tensorflow as tf
from keras.activations import softmax
from tensorflow import einsum, reshape, matmul, tile, reduce_sum, transpose, norm
from tensorflow.python.client import session
import keras.backend as K

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors

def get_norm(vector):
    return norm(vector, ord=2, axis=0)

def length_vec(x):
    r = [a*a for a in x]
    r = sum(r)
    return np.sqrt(r)

def paper_regulariser(weights):
    normalised_weights = weights / tf.sqrt(tf.reduce_sum(tf.square(weights), axis=0, keepdims=True))
    dot_prod_between_topic_matrices = tf.matmul(tf.transpose(normalised_weights), normalised_weights)
    return dot_prod_between_topic_matrices



a = tf.constant([[1,2], [3,4]])
b = tf.constant([[10,10], [10,10]])
w=tf.constant([[1],[2],[3],[4],[5],[6],[7],[8],[9]])
nlw =  tf.constant([
                      [
                       [1.0, 2.0, 3.0],
                       [10.0, 11.0, 12.0],
                       [1.0, 2.0, 3.0],
                       [10.0, 11.0, 12.0],
                       [1.0, 2.0, 3.0]
                      ],

                      [
                       [1.0, 2.0, 3.0],
                       [10.0, 11.0, 12.0],
                       [1.0, 2.0, 3.0],
                       [10.0, 11.0, 12.0],
                       [1.0, 2.0, 3.0],
                      ],
                    ])

t = tf.constant([0.5, 0.5, 0.5, 0.5, 0.5])

with tf.Session() as sess:
    rang = transpose(tf.range(0.0, 256.0, 1.0))
    r = sess.run(squash(t))
    rr = sess.run(squash(rang))
    r = get_norm(r)
    rr = get_norm(rr)

    test_weights = tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32)
    test_weights = tf.stack([test_weights,tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),test_weights,
              tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),
              tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32),tf.convert_to_tensor(np.random.rand(256), dtype=tf.float32)], axis=1)

    b = sess.run(test_weights / tf.sqrt(tf.reduce_sum(tf.square(test_weights), axis=0, keepdims=True)))
    b = np.matmul(np.transpose(b), b)

    normalised_weights = sess.run(paper_regulariser(test_weights))

    rang = transpose(tf.range(0.0, 256.0, 1.0))
    rang = tile([transpose(rang)], [80, 1])
    rang = tile([rang], [1, 1, 1])
    b = sess.run(call(tf.ones((256, 12)), rang))
    r = sess.run(tf.tile(w, [10, 10]))
    print(r)
    embed = sess.run(tf.reshape(nlw, [-1, 3]))
    #print(embed)
    # h = tf.matmul(embed, t)
    # h = tf.reshape(h, [-1, 2, 1])
