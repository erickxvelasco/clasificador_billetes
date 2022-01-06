import glob
import sys
import os
import cv2
import csv

import numpy as np
import pandas as pd

from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow

from sklearn.cluster import KMeans 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#ruta de las imagenes
imagen10="./billete/10"
imagen20="./billete/20"
imagen50="./billete/50"
imagen100="./billete/100"
imagen200="./billete/200"


files_billetes10= os.listdir(imagen10)
files_billetes20= os.listdir(imagen20)
files_billetes50= os.listdir(imagen50)
files_billetes100= os.listdir(imagen100)
files_billetes200= os.listdir(imagen200)

#devuelve la media del rgb de la imagen
def devolver_rgb_medio(ruta_imagen):
    im = cv2.imread(ruta_imagen)

    width = 2
    height = 2 # keep original height
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    
    #print('Resized Dimensions : ',resized.shape)
    
    #cv2.imshow("Resized image", resized)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) 

    colors, count = np.unique(hsv.reshape(-1, hsv.shape[-1]), axis=0, return_counts=True)
    #print(colors[np.argsort(-count)])
    r,g,b=0,0,0
    for i in colors:
        r+=i[0]
        g+=i[1]
        b+=i[2]
    r=r/len(colors)
    g=g/len(colors)
    b=b/len(colors)

    rgb=(int(r),int(g),int(b))
    return rgb
#recorre los directorios con las imagenes para procesar las imagenes
#y generar el dataset
def generar_dataset():
    c=0
    print("imagenes procesadas :")
    for archivo in files_billetes10:
        print(archivo)
        aux='./billete/10/' + archivo
        '''
        im=imread(aux)
        #nn=im.resize((256,256))
        plt.figure(num=1, figsize=(1, 1), dpi=10)
        imshow(im)
        image_to_pandas(im)
        '''
        rgb=devolver_rgb_medio(aux)
        with open('dataset.csv','a', newline='')as csvfile:
            fieldnames=['R','G','B']
            writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
            if (c==0):
                writer.writeheader()
            writer.writerow({'R':rgb[0],'G':rgb[1],'B':rgb[2]})
            c+=1
    for archivo in files_billetes20:
        print(archivo)
        aux='./billete/20/' + archivo       
        rgb=devolver_rgb_medio(aux)
        with open('dataset.csv','a', newline='')as csvfile:
            fieldnames=['R','G','B']
            writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
            #writer.writeheader()
            writer.writerow({'R':rgb[0],'G':rgb[1],'B':rgb[2]})
    for archivo in files_billetes50:
        print(archivo)
        aux='./billete/50/' + archivo       
        rgb=devolver_rgb_medio(aux)
        with open('dataset.csv','a', newline='')as csvfile:
            fieldnames=['R','G','B']
            writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
            #writer.writeheader()
            writer.writerow({'R':rgb[0],'G':rgb[1],'B':rgb[2]})
    for archivo in files_billetes100:
        print(archivo)
        aux='./billete/100/' + archivo       
        rgb=devolver_rgb_medio(aux)
        with open('dataset.csv','a', newline='')as csvfile:
            fieldnames=['R','G','B']
            writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
            #writer.writeheader()
            writer.writerow({'R':rgb[0],'G':rgb[1],'B':rgb[2]})
    for archivo in files_billetes200:
        print(archivo)
        aux='./billete/200/' + archivo       
        rgb=devolver_rgb_medio(aux)
        with open('dataset.csv','a', newline='')as csvfile:
            fieldnames=['R','G','B']
            writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
            #writer.writeheader()
            writer.writerow({'R':rgb[0],'G':rgb[1],'B':rgb[2]})

#generamos el numero de clusters con el metodo del codo
#generamos los clusters en una grafica 2d 
def generar_kmeans2D_codo():

    # Carga de datos pre entrenados
    dataset = pd.read_csv('dataset.csv')
    X = dataset.iloc[:, [0, 1]].values

    # Metodo de Codos mediante wcss(x)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Grafica de la suma de las distancias
    plt.plot(range(1, 11), wcss)
    plt.title('Metodo Codo')
    plt.xlabel('Numero de clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Visualizacion grafica de los clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Clúster1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Clúster2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Clúster3')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroide')

    plt.title('Billetes(RGB)')
    plt.xlabel('valor')
    plt.ylabel('Porcentaje (1-100)')
    plt.legend()
    plt.show()

#generamos los clusters en una grafica 3d
def generar_kmeans3D():
    csv_file = open('dataset.csv', 'r')
    data = list(csv.DictReader(csv_file))
    lim=len(data)
    
    X = np.zeros([lim, 3])
    Y=np.zeros([lim,3])
   
    i = 0
    for item in data:
        if (item['R'] != 'Null'):
            X[i][0] = float(item['R'])
            Y[i][0] = float(item['R'])
        if (item['G'] != 'Null'):
            X[i][1] = float(item['G'])
            Y[i][1] = float(item['G'])
        if (item['B'] != 'Null'):
            X[i][2] = float(item['B'])
            Y[i][2] = float(item['B'])
        i = i + 1
    # Define columnas para los centroides
    kmeans = KMeans()
    kmeans.fit(X)
    j = 0
    df = pd.DataFrame(Y, columns=['R', 'G', 'B'])

    # Kmeans
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(df['R'])
    y = np.array(df['G'])
    z = np.array(df['B'])
    damn = ['latitude', 'longitude', 'month']
    ax.scatter(x, y, z, marker="s", c=df["R"], s=80, label=damn, cmap="RdBu")
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    # Creacion de los centroides
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=10, alpha=1)
    # end
    plt.show()

generar_dataset()
generar_kmeans2D_codo()
generar_kmeans3D()
