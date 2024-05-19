# Tratamiento de datos
# ==============================================================================
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Datos
# ==============================================================================
inputs = [1,2,3,4,5,6,7,8,9,10]
outputs = [20,36,48,60,66,74,80,84,87,89]

plt.scatter(inputs, outputs)
plt.show()

datos = pd.DataFrame({'inputs': inputs, 'outputs': outputs})

# División de los datos en train y test
# ==============================================================================
X = datos[['inputs']]
y = datos['outputs']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo
# ==============================================================================
print("Recta de regresión: y=%.2f+%.2f*X" % (modelo.intercept_[0], modelo.coef_.flatten()[0]))
# Correlación lineal entre las dos variables
# ==============================================================================
corr_test = pearsonr(x = datos['inputs'], y =  datos['outputs'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("Coeficiente de determinación R^2:", modelo.score(X, y))

# Se puede ver con la recta y con los coeficientes que están claramente correladas directamente Pearson casi 1

# Datos
# ==============================================================================
inputs = [30,27,24,32,20,35]
outputs = [0.5,1,0.9,0.6,1.2,0.4]

plt.clf()
plt.scatter(inputs, outputs)
plt.show()

datos = pd.DataFrame({'inputs': inputs, 'outputs': outputs})

# División de los datos en train y test
# ==============================================================================
X = datos[['inputs']]
y = datos['outputs']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo
# ==============================================================================
print("Recta de regresión: y=%.2f%.2f*X" % (modelo.intercept_[0], modelo.coef_.flatten()[0]))
# Correlación lineal entre las dos variables
# ==============================================================================
corr_test = pearsonr(x = datos['inputs'], y =  datos['outputs'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("Coeficiente de determinación R^2:", modelo.score(X, y))

# Se puede ver con la recta y con los coeficientes que están claramente correladas inversamente Pearson casi -1

import pandas
from sklearn import linear_model

# Importar los datos en csv

df = pandas.read_csv("data.csv")

# Definir las variables aleatorias X e y

X = df[['Weight', 'Volume']]
y = df['CO2']

# Llamar directamente a la función de regresión lineal

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)

# Explicar el video
