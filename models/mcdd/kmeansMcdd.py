import pandas as pd
#import modin.pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
#from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# Davies Bouldin score for K means
from sklearn.metrics import davies_bouldin_score
# Dendogram for Heirarchical Clustering
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn import metrics
#import tensorflow
from time import time
import pickle
from queries.mccd import resultadosModHeaders

def KmeansMCDD(data):
  # copy new dataframe
  #db_1 = db_select.copy(2)
  db_1 = data
  db_1 = db_1.dropna()
  #db_1 = db_1.replace(',','', regex=True)
  #print(db_1.dtypes)
  #db_1 = db_1.astype(str)
  #print(db_1.dtypes)
  #print(db_1.head())

  db_1 = db_1.astype(np.float64)
  db_1 = db_1.astype(np.int64)

  np.random.seed(1)

  scaler = StandardScaler()
  db_scaled = scaler.fit_transform(db_1)

  kmeans = pickle.load(open('models/mcdd/kmeans_mcdd', 'rb'))
  kmeans.fit_predict(db_scaled)
  labels = kmeans.labels_
  #kmeans.cluster_centers_.shape

  cluster_centers = pd.DataFrame(data= kmeans.cluster_centers_, columns= [db_1])

  cluster_centers = scaler.inverse_transform(cluster_centers)
  cluster_centers = pd.DataFrame(data = cluster_centers, columns = [db_1.columns])
  cluster_centers.round(0)

  db_cluster = pd.concat([db_1, pd.DataFrame({'cluster':labels})], axis=1)
  #db_cluster.head()

  s = db_cluster.cluster
  counts=s.value_counts(dropna=True)
  per100= s.value_counts(dropna=True,normalize=True).mul(100).round(2).astype(str) + '%'
  #pd.DataFrame({'counts': counts, '%':per100})

  # obtain the principal component
  pca = PCA(n_components=2)
  principal_components = pca.fit_transform(db_scaled)

  # create dataframe with the two components
  pca_df = pd.DataFrame(data = principal_components, columns = ['pca1','pca2'])

  # concatenate the clusters labels to the dataframe
  pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels})], axis = 1)

  return pca_df.to_dict(orient = 'records')

def AnalisisMCDD(data):

  db_1 = data
  db_1 = db_1.dropna()

  db_1 = db_1.astype(np.float64)
  db_1 = db_1.astype(np.int64)

  db_1 = db_1.dropna()
  db_1.columns = resultadosModHeaders
  #df_mcdd_analisis_categoria = pd.DataFrame({
  #    "Compromiso profesional": {"moda": list(db_1['UNO'].mode()), "promedio":db_1['UNO'].mean(), "mediana": db_1['UNO'].median()},
  #    "Recursos Digitales": {"moda": list(db_1['DOS'].mode()), "promedio":db_1['DOS'].mean(), "mediana": db_1['DOS'].median()},
  #    "Enseñanza y aprendizaje": {"moda": list(db_1['TRES'].mode()), "promedio":db_1['TRES'].mean(), "mediana": db_1['TRES'].median()},
  #    "Evaluación y retroalimentación": {"moda": list(db_1['CUATRO'].mode()), "promedio":db_1['CUATRO'].mean(), "mediana": db_1['CUATRO'].median()},
  #    "Empoderar a los estudiantes": {"moda": list(db_1['CINCO'].mode()), "promedio":db_1['CINCO'].mean(), "mediana": db_1['CINCO'].median()},
  #    "Desarrollo de la competencia digital de los estudiantes": {"moda": list(db_1['SEIS'].mode()), "promedio":db_1['SEIS'].mean(), "mediana": db_1['SEIS'].median()}
  #})
  df_mcdd_analisis_categoria = pd.DataFrame({
    "desviacion": list(db_1.std().round(decimals=2)),
    "promedio": list(db_1.mean().round(decimals=2)),
    "mediana": list(db_1.median().round(decimals=2)),
    "maxima": list(db_1.max()),
    "minima": list(db_1.min()),
    "moda": list(db_1.mode().max().round(decimals=2)),
    "competencia": list(db_1.columns)
  })
  print(db_1.head())
  return df_mcdd_analisis_categoria.to_dict(orient = 'records')

def KmeansMCDDCounts(data):

  db_1 = data
  db_1 = db_1.dropna()

  db_1 = db_1.astype(np.float64)
  db_1 = db_1.astype(np.int64)

  np.random.seed(1)

  scaler = StandardScaler()
  db_scaled = scaler.fit_transform(db_1)

  kmeans = pickle.load(open('models/mcdd/kmeans_mcdd', 'rb'))
  kmeans.predict(db_scaled)
  labels = kmeans.labels_
  #kmeans.cluster_centers_.shape

  cluster_centers = pd.DataFrame(data= kmeans.cluster_centers_, columns= [db_1])

  cluster_centers = scaler.inverse_transform(cluster_centers)
  cluster_centers = pd.DataFrame(data = cluster_centers, columns = [db_1.columns])
  cluster_centers.round(0)

  db_cluster = pd.concat([db_1, pd.DataFrame({'cluster':labels})], axis=1)
  #db_cluster.head()

  s = db_cluster.cluster
  counts=s.value_counts(dropna=True)
  #per100= s.value_counts(dropna=True,normalize=True).mul(100).round(2).astype(str) + '%'
  per100= s.value_counts(dropna=True,normalize=True).mul(100).round(2)
  dfPercet = pd.DataFrame({'cluster': counts.index, 'counts': counts, 'percent':per100})

  return dfPercet.to_dict(orient = 'records')


def KmeansMCDDPredict(data):
  # copy new dataframe
  #db_1 = db_select.copy(2)
  db_1 = data
  db_1 = db_1.dropna()
  #db_1 = db_1.replace(',','', regex=True)
  #print(db_1.dtypes)
  #db_1 = db_1.astype(str)
  #print(db_1.dtypes)
  #print(db_1.head())

  db_1 = db_1.astype(np.float64)
  db_1 = db_1.astype(np.int64)

  np.random.seed(1)

  scaler = StandardScaler()
  db_scaled = scaler.fit_transform(db_1)

  kmeans = pickle.load(open('models/mcdd/kmeans_mcdd', 'rb'))
  kmeans.fit_predict(db_scaled)
  labels = kmeans.labels_
  #kmeans.cluster_centers_.shape

  cluster_centers = pd.DataFrame(data= kmeans.cluster_centers_, columns= [db_1])

  cluster_centers = scaler.inverse_transform(cluster_centers)
  cluster_centers = pd.DataFrame(data = cluster_centers, columns = [db_1.columns])
  cluster_centers.round(0)

  db_cluster = pd.concat([db_1, pd.DataFrame({'cluster':labels})], axis=1)
  db_cluster.dropna()
  #db_cluster.head()
  db_cluter_dict = db_cluster.to_dict(orient = 'records')

  s = db_cluster.cluster
  counts=s.value_counts(dropna=True)
  per100= s.value_counts(dropna=True,normalize=True).mul(100).round(2)
  #pd.DataFrame({'counts': counts, '%':per100})
  dfPercet = pd.DataFrame({'cluster': counts.index, 'counts': counts, 'percent':per100})
  dfPercet.dropna()

  dfPercet = dfPercet.to_dict(orient = 'records')

  # obtain the principal component
  pca = PCA(n_components=2)
  principal_components = pca.fit_transform(db_scaled)

  # create dataframe with the two components
  pca_df = pd.DataFrame(data = principal_components, columns = ['pca1','pca2'])
  pca_df.dropna()

  # concatenate the clusters labels to the dataframe
  pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels})], axis = 1)
  pca_df = pca_df.to_dict(orient = 'records')


  return {"kmeans": pca_df, "counts": dfPercet, "results": db_cluter_dict}

def KmeansMCDDAnalisisCompetencias(data):

  db_1 = data
  db_1 = db_1.dropna()

  db_1 = db_1.astype(np.float64)
  db_1 = db_1.astype(np.int64)

  np.random.seed(1)

  scaler = StandardScaler()
  db_scaled = scaler.fit_transform(db_1)

  kmeans = pickle.load(open('models/mcdd/kmeans_mcdd', 'rb'))
  kmeans.fit_predict(db_scaled)
  labels = kmeans.labels_
  #kmeans.cluster_centers_.shape

  cluster_centers = pd.DataFrame(data= kmeans.cluster_centers_, columns= [db_1])

  cluster_centers = scaler.inverse_transform(cluster_centers)
  cluster_centers = pd.DataFrame(data = cluster_centers, columns = [db_1.columns])
  cluster_centers.round(0)

  db_cluster = pd.concat([db_1, pd.DataFrame({'cluster':labels})], axis=1)
  db_cluster.dropna()

  db_cluter_mode = db_cluster.groupby('cluster').agg(lambda x: x.mode().max())
  db_cluter_mode['cluster'] = db_cluter_mode.index
  db_cluter_mode['reporte'] = 'puntaje_moda_cluster'
  db_cluter_mode = db_cluter_mode.to_dict(orient = 'records')

  db_cluter_prom = db_cluster.groupby('cluster').agg(lambda x: x.mean()).round(2)
  db_cluter_prom['cluster'] = db_cluter_prom.index
  db_cluter_prom['reporte'] = 'puntaje_promedio_cluster'
  db_cluter_prom = db_cluter_prom.to_dict(orient = 'records')

  db_cluter_media = db_cluster.groupby('cluster').agg(lambda x: x.median())
  db_cluter_media['cluster'] = db_cluter_media.index
  db_cluter_media['reporte'] = 'puntaje_media_cluster'
  db_cluter_media = db_cluter_media.to_dict(orient = 'records')

  db_cluter_max = db_cluster.groupby('cluster').agg(lambda x: x.max())
  db_cluter_max['cluster'] = db_cluter_max.index
  db_cluter_max['reporte'] = 'puntaje_maximo_alcanzado_cluster'
  db_cluter_max = db_cluter_max.to_dict(orient = 'records')

  db_cluter_min = db_cluster.groupby('cluster').agg(lambda x: x.min())
  db_cluter_min['cluster'] = db_cluter_min.index
  db_cluter_min['reporte'] = 'puntaje_minimo_alcanzado_cluster'
  db_cluter_min = db_cluter_min.to_dict(orient = 'records')

  db_cluter_num_max = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == x.max()))
  db_cluter_num_max['cluster'] = db_cluter_num_max.index
  db_cluter_num_max['reporte'] = 'participantes_puntaje_maximo_cluster'
  db_cluter_num_max = db_cluter_num_max.to_dict(orient = 'records')

  db_cluter_num_min = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == x.min()))
  db_cluter_num_min['cluster'] = db_cluter_num_min.index
  db_cluter_num_min['reporte'] = 'participantes_puntaje_minimo_cluster'
  db_cluter_num_min = db_cluter_num_min.to_dict(orient = 'records')

  db_cluter_num_media = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == x.median()))
  db_cluter_num_media['cluster'] = db_cluter_num_media.index
  db_cluter_num_media['reporte'] = 'participantes_puntaje_media_cluster'
  db_cluter_num_media = db_cluter_num_media.to_dict(orient = 'records')

  db_cluter_num_prom = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == int(x.mean())))
  db_cluter_num_prom['cluster'] = db_cluter_num_prom.index
  db_cluter_num_prom['reporte'] = 'participantes_puntaje_promedio_cluster'
  db_cluter_num_prom = db_cluter_num_prom.to_dict(orient = 'records')

  db_cluter_num_moda = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == x.mode().max()))
  db_cluter_num_moda['cluster'] = db_cluter_num_moda.index
  db_cluter_num_moda['reporte'] = 'participantes_puntaje_moda_cluster'
  db_cluter_num_moda = db_cluter_num_moda.to_dict(orient = 'records')

  db_cluter_percent_prom = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == int(x.mean())) / x.count() * 100).round(2)
  db_cluter_percent_prom['cluster'] = db_cluter_percent_prom.index
  db_cluter_percent_prom['reporte'] = 'porcentaje_puntaje_promedio_cluster'
  db_cluter_percent_prom = db_cluter_percent_prom.to_dict(orient = 'records')

  db_cluter_percent_max = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == x.max())/ x.count() * 100).round(2)
  db_cluter_percent_max['cluster'] = db_cluter_percent_max.index
  db_cluter_percent_max['reporte'] = 'porcentaje_puntaje_maximo_cluster'
  db_cluter_percent_max = db_cluter_percent_max.to_dict(orient = 'records')

  db_cluter_percent_min = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == x.min())/ x.count() * 100).round(2)
  db_cluter_percent_min['cluster'] = db_cluter_percent_min.index
  db_cluter_percent_min['reporte'] = 'porcentaje_puntaje_minimo_cluster'
  db_cluter_percent_min = db_cluter_percent_min.to_dict(orient = 'records')

  db_cluter_percent_moda = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == x.mode().max())/ x.count() * 100).round(2)
  db_cluter_percent_moda['cluster'] = db_cluter_percent_moda.index
  db_cluter_percent_moda['reporte'] = 'porcentaje_puntaje_moda_cluster'
  db_cluter_percent_moda = db_cluter_percent_moda.to_dict(orient = 'records')

  db_cluter_percent_media = db_cluster.groupby('cluster').agg(lambda x: np.sum(1 for i in x if i == x.median())/ x.count() * 100).round(2)
  db_cluter_percent_media['cluster'] = db_cluter_percent_media.index
  db_cluter_percent_media['reporte'] = 'porcentaje_puntaje_media_cluster'
  db_cluter_percent_media = db_cluter_percent_media.to_dict(orient = 'records')

  return db_cluter_mode + db_cluter_prom + db_cluter_media + db_cluter_max + db_cluter_min + db_cluter_num_moda + db_cluter_num_max + db_cluter_num_min + db_cluter_num_media + db_cluter_num_prom + db_cluter_percent_prom + db_cluter_percent_max + db_cluter_percent_min + db_cluter_percent_moda + db_cluter_percent_media