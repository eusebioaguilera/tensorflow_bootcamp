from turtle import shape
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Descargamos el dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Creamos el modelo
# y = Wx + b
x = tf.placeholder(tf.float32, shape=[None, 784])
# Cada image es de 28x28 lo que aplanado queda en [1, 784]. Para cada imagen tendremos 10 posibles salidas.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Creamos el grafo del modelo
y = tf.matmul(x, W) + b

# Creamos el y_true
y_true = tf.placeholder(tf.float32, [None, 10])

# Función de loss del modelo
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# Creamos un optimizador para entrenar el modelo
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
# Y le indicamos que queremos minimzar el error
train = optimizer.minimize(cross_entropy)

# Inicizalizamos las variables
init = tf.global_variables_initializer()

# Ejecución del entrenamiento
with tf.Session() as sess:
    # Ejecución de la inicialización
    sess.run(init)

    # Entrenamiento del modelo durante 1000 épocas
    for step in range(1000):
        # Obtenemos un batch de tamaño 100 para cada época
        batch_x, batch_y = mnist.train.next_batch(100)
        # Ejecutamos el entrenamiento para cada batch
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # Evaluación del modelo. Obtenemos las predicciones y las clases del GT.
    # Hay que tener en cuenta que las predicciones son la salida de una softmax,
    # por lo que aplicamos el argmax para que nos indique el índice de la clase
    # con mayor probabilidad.
    # El resultado final de matches es un vector con los aciertos para cada elemento [True, False, ..., True]
    matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    # Para el accuracy primero pasamos de bool a float32 [True, False, ..., True] --> [1.0, 0.0, ..., 1.0]
    # Posteriomente calculamos la media de ese vector que se corresponde con el accuracy
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

    # Calculamos y mostramos el accuracy obtenido para el conjunto de test
    print(sess.run(acc, feed_dict={
          x: mnist.test.images, y_true: mnist.test.labels}))
