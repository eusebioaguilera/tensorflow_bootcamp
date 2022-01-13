import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
import math

n_steps = int(sys.argv[1])
if n_steps == None:
    n_steps = 1000

# Leemos el fichero de datos
housing = pd.read_csv('cal_housing_clean.csv')

# Obtenemos los datos entrada (todo menos la columna de precios)
x_data = housing.drop(['medianHouseValue'], axis=1)

# Separamos la columna de precios
y_data = housing['medianHouseValue']

# Hacemos el split de datos train/eval
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=101)

# Normalizamos los valores numéricos de los atributos
scaler = MinMaxScaler()
scaler.fit(X_train)

# Creamos los nuevos datos ya normalizados
X_train = pd.DataFrame(data=scaler.transform(
    X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(data=scaler.transform(
    X_test), columns=X_test.columns, index=X_test.index)

# Se crean las columnas del modelo de regresión para TensorFlow
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')
feat_cols = [age, rooms, bedrooms, pop, households, income]

# Creamos la función de entrada
input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# Creamos el modelo de regresión
model = tf.estimator.DNNRegressor(
    hidden_units=[6, 6, 6], feature_columns=feat_cols)

# Entrenamos el modelo
model.train(input_fn=input_func, steps=n_steps)

# Creamos la función de entrada para la evaluación. En este caso no queremos que baraje los casos, por lo que shuffle=False
test_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)

# Realizamos las predicciones sobre el grupo de evaluación
pred_gen = model.predict(input_fn=test_input_func)
predictions = list(pred_gen)

final_preds = list()
# Sacamos las predicciones del diccionario
for pred in predictions:
    final_preds.append(pred['predictions'])

# Calculamos el RMSE de las predicciones de nuestro modelo
RMSE = math.sqrt(mean_squared_error(y_test, final_preds))

print("Model RMSE: %lf" % (RMSE))
