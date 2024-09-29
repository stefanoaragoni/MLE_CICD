# Importar librerías necesarias
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle

# Cargar el modelo y evaluar
def evaluar_modelo():
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    # Cargar el modelo entrenado
    with open('model/modelo_decision_tree.pkl', 'rb') as file:
        model = pickle.load(file)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy del modelo: {accuracy:.2f}')

if __name__ == "__main__":
    evaluar_modelo()
