import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('Loan_Eligibility_1200.csv')
df.describe()
print(df.columns)
nuevas_columnas = [
    'ID_Cliente', 'Genero', 'Casado', 'Dependientes', 'Educacion', 'Independiente',
    'Ingreso_Solicitante', 'Ingreso_CoSolicitante', 'Monto_Prestamo',
    'Plazo_Prestamo', 'Historial_Credito', 'Zona_Propiedad', 'Estado_Prestamo'
]
df.columns = nuevas_columnas

print("--- Información del Dataset ---")
print(df.info())
print(df.head())

# Limpieza y Transformación de Datos
df = df.drop('ID_Cliente', axis=1)
# Genero: Male -> 1, Female -> 0
df['Genero'] = df['Genero'].map({'Male': 1, 'Female': 0})
# Casado: Yes -> 1, No -> 0
df['Casado'] = df['Casado'].map({'Yes': 1, 'No': 0})
# Educacion: Graduate -> 1, Not Graduate -> 0
df['Educacion'] = df['Educacion'].map({'Graduate': 1, 'Not Graduate': 0})
# Independiente: Yes -> 1, No -> 0
df['Independiente'] = df['Independiente'].map({'Yes': 1, 'No': 0})
# Estado_Prestamo (Variable Objetivo): Y -> 1, N -> 0
df['Estado_Prestamo'] = df['Estado_Prestamo'].map({'Y': 1, 'N': 0})
# Zona_Propiedad: Asignamos valores numéricos arbitrarios para poder procesarlos
# Urban -> 2, Semiurban -> 1, Rural -> 0
df['Zona_Propiedad'] = df['Zona_Propiedad'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})

print(df.shape)
# Validar valores nulos
print(df.isna().sum())

# Definición de variables X (Predictoras) e y (Objetivo)
X = df.drop('Estado_Prestamo', axis=1)
y = df['Estado_Prestamo']

# División del dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test.count()

# Entrenamiento del modelo
modelo = LogisticRegression(max_iter=1000) # max_iter aumentado para asegurar convergencia
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Medidas
#Accurary (Exactitud)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.2f}")
#Presicion (Presición)
print(f"Presición: {metrics.precision_score(y_test, y_pred):.2f}")
#Recall (Exhaustividad)
print(f"Recall: {metrics.recall_score(y_test, y_pred):.2f}")
#F1 Score
print(f"F1 Score: {metrics.f1_score(y_test, y_pred):.2f}")


print("\n--- Matriz de Confusión ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print (modelo.coef_)
print (modelo.intercept_)
ce=modelo.coef_[0]
print(f"h(x) = {modelo.intercept_[0]} + {ce[0]} X1 + {ce[1]} X2 + {ce[3]} X3 + {ce[4]} X4 + {ce[3]} X5")
print("Salida final g(h(x))  ---    1 / (1 + e(-z))")

# Visualización de la matriz de confusión
etiquetas = ['No le presta', 'Si le presta']
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=etiquetas, 
            yticklabels=etiquetas)
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.savefig('confusion_matrix_loan.png')

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=etiquetas))