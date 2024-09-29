# Importar librerías necesarias
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar y preprocesar el dataset
def preprocesar():

    # Crear la carpeta 'data' si no existe
    os.makedirs('data', exist_ok=True)

    # Cargar el dataset
    df = pd.read_csv('./data/iris.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar los datos preprocesados
    pd.DataFrame(X_train_scaled).to_csv('./data/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv('./data/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('./data/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('./data/y_test.csv', index=False)

if __name__ == "__main__":
    preprocesar()
