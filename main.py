#import types
from flask import Flask
from flask_cors import CORS
from flask_mysqldb import MySQL
from flask import jsonify
import pandas as pd
from queries.mccd import modulos, modulosHeaders, resultadosModVar, resultadosModVarHeaders, resultadosModulos
from models.mcdd.kmeansMcdd import KmeansMCDD, KmeansMCDDCounts, KmeansMCDDPredict, AnalisisMCDD, KmeansMCDDAnalisisCompetencias
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['MYSQL_HOST'] = os.getenv('DB_MYSQL_HOST')#107.21.217.238
app.config['MYSQL_USER'] = os.getenv('DB_MYSQL_USER')#competidor
app.config['MYSQL_PASSWORD'] = os.getenv('DB_MYSQL_PASSWORD')#Kompetim0s1979
app.config['MYSQL_DB'] = os.getenv('DB_MYSQL_DB')#competencias
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')#5b74b63af609d44aa00613b7b7e6c3b035e95a2a6f
#app.config["MYSQL_PORT"] = 3306

mysql = MySQL(app)

@app.route('/hello')
def hello():
  print("=>", os.getenv('SECRET_KEY'))
  return "Hello"

@app.route('/')
def index():
  cur = mysql.connection.cursor()
  cur.execute(resultadosModVar)
  data = cur.fetchall()

  df = pd.DataFrame(data)
  df.columns = resultadosModVarHeaders
  nan_value = float("NaN")
  df.replace("", nan_value, inplace=True)
  df.dropna()
  modelResult = KmeansMCDD(df)
  return jsonify(modelResult)

@app.route('/analisis')
def analisisMCDD():
  cur = mysql.connection.cursor()
  cur.execute(resultadosModulos)
  data = cur.fetchall()

  df = pd.DataFrame(data)
  #df.columns = resultadosModVarHeaders
  nan_value = float("NaN")
  df.replace("", nan_value, inplace=True)
  df.dropna()
  modelResult = AnalisisMCDD(df)
  return jsonify(modelResult)

@app.route('/count')
def counts():
  cur = mysql.connection.cursor()
  cur.execute(resultadosModVar)
  data = cur.fetchall()

  df = pd.DataFrame(data)
  df.columns = resultadosModVarHeaders
  nan_value = float("NaN")
  df.replace("", nan_value, inplace=True)
  df.dropna()
  modelResult = KmeansMCDDCounts(df)
  #jsonResponse = modelResult.to_json(orient = 'records')
  #jsonResponse = jsonify(KmeansMCDD(df))
  #return jsonResponse
  return jsonify(modelResult)
@app.route('/predict')
def prediction():
  cur = mysql.connection.cursor()
  cur.execute(resultadosModVar)
  data = cur.fetchall()

  df = pd.DataFrame(data)
  df.columns = resultadosModVarHeaders
  nan_value = float("NaN")
  df.replace("", nan_value, inplace=True)
  df.dropna()
  modelResult = KmeansMCDDPredict(df)
  return jsonify(modelResult)

@app.route('/competencias')
def competencias():
  cur = mysql.connection.cursor()
  cur.execute(resultadosModVar)
  data = cur.fetchall()

  df = pd.DataFrame(data)
  df.columns = resultadosModVarHeaders
  nan_value = float("NaN")
  df.replace("", nan_value, inplace=True)
  df.dropna()
  modelResult = KmeansMCDDAnalisisCompetencias(df)
  return jsonify(modelResult)

if __name__ == '__main__':
  app.run(port=5000, debug=True)
  #app.run(port=5000, debug=True, ssl_context=context)
  #app.run()