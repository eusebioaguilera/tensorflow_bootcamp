from cgi import test
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


def convert_income_brackect_to_int(str):
    if str == ' <=50K':
        return 0
    else:
        return 1


# Leemos el dataset
census = pd.read_csv('census_data.csv')

# Cambiamos la etiqueta del dataset por valores numéricos
census['income_bracket'] = census['income_bracket'].apply(
    convert_income_brackect_to_int)

# Separamos los datos y las etiquetas
x_data = census.drop('income_bracket', axis=1)
y_data = census['income_bracket']

# Separamos en train/test
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=101)

# Creamos las columnas del modelo
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", [
                                                                   "Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(
    "marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket(
    "relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket(
    "workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

feat_cols = [gender, occupation, marital_status, relationship, education, workclass, native_country,
             age, education_num, capital_gain, capital_loss, hours_per_week]

# Creamos la función de entrada
input_func = tf.estimator.inputs.pandas_input_fn(
    x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

# Creamos un modelo lineal de clasificación
linear_model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

# Lo entrenamos
linear_model.train(input_fn=input_func, steps=5000)

# Funcion de entrada para la evaluacion
pred_func = tf.estimator.inputs.pandas_input_fn(
    x=x_test, batch_size=len(x_test), num_epochs=1, shuffle=False)


predictions = linear_model.predict(input_fn=pred_func)

# Sacamos las predicciones
final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(classification_report(y_test, final_preds))
