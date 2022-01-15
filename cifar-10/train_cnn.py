import tensorflow as tf
import numpy as np

############## Funciones auxiliares ####################################
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

####################### DATOS ###########################################


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


def one_hot_encode(vec, vals=10):
    '''
        One-Hot enconding para las etiquetas de las clases definidas
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarHelper():
    """
        Esta clase define métodos que nos ayudarán a cargar el dataset
        e ir obteniendo cada batch para cada iteración del ciclo de
        entrenamiento.
    """

    def __init__(self):
        self.i = 0

        # Obtiene todos los batches para realizar el entrenamiento
        self.all_train_batches = [
            data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]

        # Obtiene una lista de todos los batches que se usarán para test
        self.test_batch = [test_batch]

        # Inicializa las variables que están vacías
        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        """
            Este método se usa para establecer las imágenes dentro del objeto definido
        """

        # Creamos un stack vertical para las imágenes de entrenamiento
        self.training_images = np.vstack(
            [d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        # Reshapes and normalizes training images
        # Cambiamos el tamaño de ndarray, normalizamos el dataset de entrenamiento
        self.training_images = self.training_images.reshape(
            train_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
        # Realizamos el procceso de One-hot para las etiquetas ([0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(
            np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        # Repetimos el proceso para el conjunto de pruebas/test
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        # Cambiamos el tamaño de ndarray, normalizamos el dataset de test
        self.test_images = self.test_images.reshape(
            test_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
        # Realizamos el procceso de One-hot para las etiquetas ([0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(
            np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size=100):
        """
            Este método será el encargado de obtener el siguiente batch para el proceso 
            de entrenamiento.
        """
        x = self.training_images[self.i:self.i +
                                 batch_size].reshape(batch_size, 32, 32, 3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

    def get_test(self):
        return self.test_images, self.test_labels

    def get_test_images(self):
        return self.test_images

    def get_test_labels(self):
        return self.test_labels

#########################################################################


# Directorio dónde está el dataset
CIFAR_DIR = 'cifar-10-dataset/'

dirs = ['batches.meta', 'data_batch_1', 'data_batch_2',
        'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0, 1, 2, 3, 4, 5, 6]


for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)


# Definicion de datos

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]


# Creamos el modelo

# Definimos los puntos de entrada del modelo
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

# Definimos la probabilidad para realizar un dropout
hold_prob = tf.placeholder(tf.float32)

# Capas de modelo
convo_layer1 = convolutional_layer(x, shape=[4, 4, 3, 32])
max_pool_layer1 = max_pool_2by2(convo_layer1)

convo_layer2 = convolutional_layer(max_pool_layer1, shape=[5, 5, 32, 64])
max_pool_layer2 = max_pool_2by2(convo_layer2)

# Aplanamos y obtenemos la salida de una capa totalmente conectada con función de activación relu
convo_2_flat = tf.reshape(max_pool_layer2, [-1, 8*8*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

# Capa dónde realizaremos el dropout
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

# Inicializamos el dataset
cifar = CifarHelper()
cifar.set_up_images()

# Entrenamos dentro de la sesión
with tf.Session() as sess:
    # Inicializamos
    sess.run(init)
    # Para cada ejecución
    for i in range(steps):
        # Obtemos el batch sobre el que vamos a operar

        batch_x, batch_y = cifar.next_batch(100)
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
                  x: cifar.get_test_images(), y_true: cifar.get_test_labels(), hold_prob: 1.0}))
            print('\n')
