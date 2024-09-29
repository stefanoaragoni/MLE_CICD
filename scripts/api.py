# Importar librerías necesarias
from flask import Flask, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

def cargar_modelo():
    with open('./model/modelo_decision_tree.pkl', 'rb') as file:
        return pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convertir los datos de entrada a un DataFrame
    input_data = pd.DataFrame(data, index=[0])
    model = cargar_modelo()
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    # Leer el accuracy del modelo
    with open('./data/accuracy.txt', 'r') as f:
        accuracy = float(f.read().strip())

    # Iniciar el servidor solo si el accuracy es mayor a 0.7
    if accuracy > 0.7:
        #app.run(host='0.0.0.0', port=5000)

        print("Se llama a algún otro servicio para iniciar el servidor.")
        print("Esto es solo un ejemplo.")

    else:
        print("El modelo no cumple con el requisito de accuracy.")
