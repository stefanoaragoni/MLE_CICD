# Importar librerías necesarias
import pandas as pd
from sklearn.datasets import load_iris

# Cargar el dataset de Iris
def recopilar_datos():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    # Guardar el dataset como un archivo CSV
    df.to_csv('./data/iris.csv', index=False)

if __name__ == "__main__":
    recopilar_datos()
