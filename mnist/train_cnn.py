from hashlib import sha1
from turtle import shape
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Función de inicialización de los pesos


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

# Inicialización del sesgo


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

# Devuelve una convolución de x y W
# X --> [batch, Height, Width, Channels]
# W --> [Filter H, Filter W, Channels In, Channels OUT]


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    # x --> [batch, h, w, c]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape=shape)
    b = init_bias(shape=[shape[3]])

    return tf.nn.relu(conv2d(input_x, W)+b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# Descargamos los datos
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Definimos los puntos de entrada del modelo
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

# Tenemos que poner la imagen en formato original 28x28
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Definición de las capas del modelo
conv_layer1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
conv_pooling_layer1 = max_pool_2by2(conv_layer1)

# Segunda capa convolucional
conv_layer2 = convolutional_layer(conv_pooling_layer1, shape=[6, 6, 32, 64])
conv_pooling_layer2 = max_pool_2by2(conv_layer2)

# Aplanamos y obtenemos la salida de una capa totalmente conectada con función de activación relu
convo_2_flat = tf.reshape(conv_pooling_layer2, [-1, 7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

# Se añade un dropout para evitar el overfitting
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

# La salida de la red será la capa dónde se hace dropout conectada densamente a 10 nodos que representan las 10 clases
y_pred = normal_full_layer(full_one_dropout, 10)

# Definicimos la función de pérdida para realizar el entrenamiento
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# El optimizador que vamos a emplear
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

# Inicializamos las variables globales
init = tf.global_variables_initializer()

# Número de iteraciones a realizar
steps = 5000

# Entrenamos dentro de la sesión
with tf.Session() as sess:
    # Inicializamos
    sess.run(init)
    # Para cada ejecución
    for i in range(steps):
        # Obtemos el batch sobre el que vamos a operar
        batch_x, batch_y = mnist.train.next_batch(50)
        # Ejecutamos el entrenamiento
        sess.run(train, feed_dict={x: batch_x,
                 y_true: batch_y, hold_prob: 0.5})

        # Cada 100 iteraciones mostrarmos como va
        if i % 100 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Evaluamos el modelo
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            # Calculamos el accuracy igual que en el caso del train anterior
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            # Mostramos los valores
            print(sess.run(acc, feed_dict={
                  x: mnist.test.images, y_true: mnist.test.labels, hold_prob: 1.0}))
            print('\n')
