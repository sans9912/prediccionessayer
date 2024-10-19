from flask import Flask, jsonify
import pyodbc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
# Conexión a la base de datos SQL Server

connection_string = "DRIVER={SQL Server};SERVER=sql.bsite.net\MSSQL2016;DATABASE=distsayer_sayer;UID=distsayer_sayer;PWD=sayereportes;TrustServerCertificate=yes"

def obtener_datos(query):
    conn = pyodbc.connect(connection_string)
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def predecir_ventas():
    query = "SELECT fecha, venta FROM reportes WHERE fecha >= '2023-01-01'"
    data = obtener_datos(query)
    data['fecha'] = pd.to_datetime(data['fecha'])
    data['mes'] = data['fecha'].dt.to_period('M')
    ventas_mensuales = data.groupby('mes')['venta'].sum().reset_index()
    ventas_mensuales['mes_numeric'] = np.arange(len(ventas_mensuales))
    ventas_mensuales['sin_mes'] = np.sin(2 * np.pi * ventas_mensuales['mes_numeric'] / 12)
    ventas_mensuales['cos_mes'] = np.cos(2 * np.pi * ventas_mensuales['mes_numeric'] / 12)
    X = ventas_mensuales[['mes_numeric', 'sin_mes', 'cos_mes']]
    y = ventas_mensuales['venta']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    modelo_lasso = Lasso(alpha=0.6)
    modelo_lasso.fit(X_train, y_train)
    ventas_futuras_lasso = modelo_lasso.predict(scaler.transform([[len(ventas_mensuales), np.sin(2 * np.pi * len(ventas_mensuales) / 12), np.cos(2 * np.pi * len(ventas_mensuales) / 12)]]))
    return ventas_futuras_lasso[0]

def predecir_utilidad():
    query = "SELECT fecha, utilidad FROM reportes WHERE fecha >= '2023-01-01'"
    data = obtener_datos(query)
    data['fecha'] = pd.to_datetime(data['fecha'])
    data['mes'] = data['fecha'].dt.to_period('M')
    utilidades_mensuales = data.groupby('mes')['utilidad'].sum().reset_index()
    utilidades_mensuales['mes_numeric'] = np.arange(len(utilidades_mensuales))
    utilidades_mensuales['sin_mes'] = np.sin(2 * np.pi * utilidades_mensuales['mes_numeric'] / 12)
    utilidades_mensuales['cos_mes'] = np.cos(2 * np.pi * utilidades_mensuales['mes_numeric'] / 12)
    X = utilidades_mensuales[['mes_numeric', 'sin_mes', 'cos_mes']]
    y = utilidades_mensuales['utilidad']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    modelo_ridge = Ridge(alpha=1.0)
    modelo_ridge.fit(X_train, y_train)
    utilidades_futuras = modelo_ridge.predict(scaler.transform([[len(utilidades_mensuales), np.sin(2 * np.pi * len(utilidades_mensuales) / 12), np.cos(2 * np.pi * len(utilidades_mensuales) / 12)]]))
    return utilidades_futuras[0]

def predecir_unidades():
    query = "SELECT fecha, unidades FROM reportes WHERE fecha >= '2023-01-01'"
    data = obtener_datos(query)
    data['fecha'] = pd.to_datetime(data['fecha'])
    data['mes'] = data['fecha'].dt.to_period('M')
    ventas_mensuales = data.groupby('mes')['unidades'].sum().reset_index()
    ventas_mensuales['mes_numeric'] = np.arange(len(ventas_mensuales))
    X = ventas_mensuales[['mes_numeric']]
    y = ventas_mensuales['unidades']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    unidades_futuras = modelo.predict([[len(ventas_mensuales)]])
    return unidades_futuras[0]

@app.route('/predicciones', methods=['GET'])
def obtener_predicciones():
    prediccion_ventas = predecir_ventas()
    prediccion_utilidad = predecir_utilidad()
    prediccion_unidades = predecir_unidades()

    resultado = {
        "ventas": prediccion_ventas,
        "utilidad": prediccion_utilidad,
        "unidades": prediccion_unidades
    }
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)