# Importar librerías necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Cargar los datos preprocesados
def entrenar_modelo():
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()

    # Crear y entrenar el modelo
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    with open('model/modelo_decision_tree.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    entrenar_modelo()
