from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np
import joblib

# Crear app
app = Flask(__name__)

# Cargar scaler
scaler = joblib.load("scaler_estudiantes.pkl")

# Cargar modelo ONNX
session = rt.InferenceSession("modelo_estudiantes.onnx")
input_name = session.get_inputs()[0].name

@app.route('/', methods=['GET'])
def home():
    return "API funcionando"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        nivel = float(data["nivel"])
        nota = float(data["nota"])
        socio = float(data["socio"])
        motiv = float(data["motivacion"])
        respon = float(data["responsabilidad"])

        X = np.array([[nivel, nota, socio, motiv, respon]], dtype=np.float32)

        X_scaled = scaler.transform(X)

        pred = session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
        pred_value = float(pred[0][0])

        return jsonify({"prediccion": pred_value})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)




