# Importar librerías necesarias
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
import os

def evaluar_modelo():

    # Crear la carpeta 'data' si no existe
    os.makedirs('data', exist_ok=True)

    # Crear la carpeta 'model' si no existe
    os.makedirs('model', exist_ok=True)

    X_test = pd.read_csv('./data/X_test.csv')
    y_test = pd.read_csv('./data/y_test.csv').values.ravel()

    with open('./model/modelo_decision_tree.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    # Guardar el accuracy en un archivo
    with open('./data/accuracy.txt', 'w') as f:
        f.write(f'{accuracy:.2f}')

    print(f'Accuracy del modelo: {accuracy:.2f}')
    return accuracy

if __name__ == "__main__":
    accuracy = evaluar_modelo()
