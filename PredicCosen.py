import pandas as pd
import numpy as np
import math
from time import time
from numba import cuda
import csv
import os.path
from os import path


class Sistema_recomendacion:
    def cargar_ratings(self, ruta_archivo,delimit):
        self.ratings = pd.read_csv(
            ruta_archivo,
            sep='"',
            delimiter=delimit,
            error_bad_lines=False,
            header=None)

    def cargar_items(self, ruta_archivo,delimit):
        self.items = pd.read_csv(
            ruta_archivo,
            delimiter=delimit,
            sep='"',
            error_bad_lines=False,
            header=None)

    def limpiar_data_cargar_items(self):
        self.items = self.items[[0, 1]]
        self.items.columns = ['itemId', 'name']
        self.items = self.items.replace(to_replace='"', value='', regex=True)


    def limpiar_data_cargar_ratings(self):
        self.ratings = self.ratings[[0, 1, 2]]
        self.ratings.columns = ['userId', 'itemId', 'rating']
        self.ratings = self.ratings.replace(to_replace='"', value='', regex=True)
        self.ratings = self.ratings.astype({
            "rating": 'int8',
            'userId': 'int16'
        })

    def generar_medias(self):
        dataframe = self.ratings
        medias = dataframe.groupby("userId", as_index=False)
        media = medias['rating'].mean()
        maximo = medias['rating'].max()
        minimo = medias['rating'].min()
        self.datos = pd.merge(maximo,minimo,on='userId')
        media.columns = ['userId','promedio']
        return media

        
    def distancias_entre_media(self):
        media = self.generar_medias()
        self.ratings = self.ratings.set_index('userId','itemId')
        self.items = self.items.set_index('itemId')
        self.rating_avg = pd.merge(self.ratings, media, on='userId')
        self.rating_avg['adg_rating'] = self.rating_avg['rating'] - self.rating_avg['promedio']

    def mostrar(self, dataframe):
        print(dataframe.head(1000))
    
    def coseno_ajustado(self,item1,item2):
        item1 = self.items.loc[self.items.loc[:,'name'] == item1].index.values[0]
        item2 = self.items.loc[self.items.loc[:,'name'] == item2].index.values[0]
        list_items1 = self.rating_avg.loc[self.rating_avg.loc[:,'itemId'] == item1 ]
        list_items2 = self.rating_avg.loc[self.rating_avg.loc[:,'itemId'] == item2 ]

        mezclar = pd.merge(list_items1, list_items2, how='inner', on='userId')
        if mezclar.shape[0]==0:
          return 0
        Sum = 0
        SumC1 = 0
        SumC2 = 0
        for i in range(mezclar.shape[0]):
          Sum += mezclar.loc[i,'adg_rating_x']*mezclar.loc[i,'adg_rating_y']
          SumC1 += mezclar.loc[i,'adg_rating_x']**2
          SumC2 += mezclar.loc[i,'adg_rating_y']**2
        return Sum/((SumC1**0.5)*(SumC2**0.5))

          

    def coseno_ajustado2(self,item1,item2,IsID=False):
        if not IsID:
          item1 = self.items.loc[self.items.loc[:,'name'] == item1].index.values[0]
          item2 = self.items.loc[self.items.loc[:,'name'] == item2].index.values[0]
        list_items1 = self.rating_avg.loc[self.rating_avg.loc[:,'itemId'] == item1 ]
        list_items2 = self.rating_avg.loc[self.rating_avg.loc[:,'itemId'] == item2 ]
        mezclar = pd.merge(list_items1, list_items2, how='inner', on='userId')
        if(mezclar.shape[0]==0):
            print("entro")
            return 0
        rating_x = mezclar[['adg_rating_x']].to_numpy().transpose()[0]
        rating_y = mezclar[['adg_rating_y']].to_numpy().transpose()[0]
        potencia_x = np.sum(pow(rating_x,2))
        potencia_y = np.sum(pow(rating_y,2))
        calc = rating_y.dot( rating_x )

        div = potencia_x*potencia_y
        div = div**0.5
        print(calc)
        print(div)

        if(div == 0):
            return 0
        calc = calc / div
        return calc

    
#Predecir Franco
    def Normalizar2(self):
        self.datos = pd.merge(self.ratings, self.datos, on='userId')
        self.datos['NR'] = (2*(self.datos['rating']-self.datos['rating_y'])-(self.datos['rating_x'] - self.datos['rating_y'] ))/(self.datos['rating_x']-self.datos['rating_y'])

    def Predecir2(self,usuario,item):
        item = self.items.loc[self.items.loc[:,'name'] == item].index.values[0]
        ratings = self.datos.loc[self.datos.loc[:,'userId'] == usuario]
        SumRa = 0.0
        SumPre = 0.0
        for index, row in ratings.iterrows():
            x= self.coseno_ajustado2(item,row['itemId'],True)
            #print("Cos: ", x)
            #print("ID: ",row['itemId'])
            y= x * row['NR']
            SumRa += abs(x)
            SumPre += y
        if SumRa!=0:
            return SumPre/SumRa
        return 0
    def DesNormalizar2(self,usuario,item):
        ratings = self.datos.loc[self.datos.loc[:,'userId'] == usuario]
        maxi = self.datos.loc[self.rating_avg.loc[:,'userId'] == usuario].max()['rating']
        mini = self.datos.loc[self.rating_avg.loc[:,'userId'] == usuario].min()['rating']
        Predicho=1/2*((self.Predecir2(usuario,item)+1)*(maxi-mini))+mini        
        return Predicho

#Predecir Nico
    def Normalizar(self,usuario):
        ratings = self.rating_avg.loc[self.rating_avg.loc[:,'userId'] == usuario]
        max=ratings.max()['rating']
        min=ratings.min()['rating']
        Normlize=[]
        Pelis = []
        for n in ratings.index:
          if max==min:
            Normalize.append(0)
          else:
            Normlize.append((2*(ratings.loc[n,'rating']-min)-(max-min))/(max-min))
          Pelis.append(ratings.loc[n,'itemId'])
        return Normlize,Pelis

    def Predecir(self, usuario, item):
        Norm, Pelis = self.Normalizar(usuario)
        item = self.items.loc[self.items.loc[:,'name'] == item].index.values[0]
        SumRa = 0.0
        SumPre=0.0
        for n, i in zip(Pelis, range(len(Pelis))):
            x= self.coseno_ajustado2(item,n,True)
            y= x * Norm[i]
            SumRa += abs(x)
            SumPre += y
        return SumPre/SumRa
    def DesNormalizar(self,usuario,item):
        ratings = self.rating_avg.loc[self.rating_avg.loc[:,'userId'] == usuario]
        max=ratings.max()['rating']
        min=ratings.min()['rating']
        Predicho=1/2*((self.Predecir(usuario,item)+1)*(max-min))+min
        return Predicho

    def DesviacionEsta(self,item1,item2,IsID=False):
        if not IsID:
          item1 = self.items.loc[self.items.loc[:,'name'] == item1].index.values[0]
          item2 = self.items.loc[self.items.loc[:,'name'] == item2].index.values[0]
        list_items1 = self.ratings.loc[self.ratings.loc[:,'itemId'] == item1]
        list_items2 = self.ratings.loc[self.ratings.loc[:,'itemId'] == item2]
        list_merge = pd.merge(list_items1, list_items2, how='inner', on='userId')
        list_items1 = list_merge[['rating_x']].to_numpy()
        list_items2 = list_merge[['rating_y']].to_numpy()
        Cardi = len(list_items1)
        if not Cardi:
            return 0 , 0
        list_items = np.subtract(list_items2,list_items1)
        list_items = np.sum(np.true_divide(list_items, Cardi))
        return list_items,Cardi

    def Slope_One(self,usuario,item):
        item = self.items.loc[self.items.loc[:,'name'] == item].index.values[0]
        list_itemsUsu = self.rating_avg.loc[self.rating_avg.loc[:,'userId'] == usuario]
        CardiSum = 0
        Desviacion = 0
        Cardinalidad = 0
        for n in list_itemsUsu.index:
            Des,C = self.DesviacionEsta(list_itemsUsu.loc[n,'itemId'],item,True)
            Desviacion += (list_itemsUsu.loc[n,'rating']+Des)*C
            Cardinalidad += C 
        if Cardinalidad:
            return Desviacion / Cardinalidad
        return 0

def main():
    movies = Sistema_recomendacion()
    movies.cargar_ratings('MovieRatings.csv',',')
    movies.cargar_items('MovieRatingsItem.csv',',')
    movies.limpiar_data_cargar_items()
    movies.limpiar_data_cargar_ratings()
    movies.generar_medias()
    movies.distancias_entre_media()
    movies.Normalizar2()
    #movies.mostrar(movies.ratings)
    #movies.mostrar(movies.datos)
    

    """Coseno
    start_time = time()
    print(movies.coseno_ajustado('Toy Story (1995)','Andrew Dice Clay: Dice Rules (1991)'))
    elapsed_time = time() - start_time
    print("Nico elapsed time: %0.10f seconds." % elapsed_time)
    start_time = time()
    print(movies.coseno_ajustado2('Toy Story (1995)','Andrew Dice Clay: Dice Rules (1991)'))
    elapsed_time = time() - start_time
    print("Franco elapsed time: %0.10f seconds." % elapsed_time)
    """
    """Predecir
    start_time = time()
    print(movies.DesNormalizar(1,"Kacey"))
    elapsed_time = time() - start_time
    print("Nico elapsed time: %0.10f seconds." % elapsed_time)
    start_time = time()
    print(movies.DesNormalizar2(1,"Kacey"))
    elapsed_time = time() - start_time
    print("Franco elapsed time: %0.10f seconds." % elapsed_time)
    """
    movies.mostrar(movies.rating_avg)
    start_time = time()
    print(movies.Slope_One(1,'Village'))
    elapsed_time = time() - start_time
    print("elapsed time: %0.10f seconds." % elapsed_time)
main()



def generar_matriz(): 
    tamanio = DFItenId.shape[0]
    
    matriz_obtenida_esparsa  = pd.DataFrame(columns = DFItenId[1], index = DFItenId[1] )
    

    
    for iterador2 in range(0,tamanio):
        p=peliculas(DFItenId[0][iterador2]) 
        print(iterador2)
        #p[iterador]  ->  representa p => vector de peliculas a comparar referetente a una pelicula
        for iterador in range(0,p.size):
            
            #matriz[iterador2][iterador] = coseno_ajustado(DFItenId[0][iterador2],DFItenId[0][iterador])
            #DFItenID  => dataframe solo peliculas
            pelicula1 = str(DFItenId[1][iterador2])
            pelicula2 = otenerItem(p[iterador])
            
            if pelicula2!= None:
              matriz_obtenida_esparsa.loc[ pelicula1 , pelicula2 ] = coseno_ajustado2(DFItenId[0][iterador2],p[iterador])

    export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
    matriz_obtenida_esparsa.to_csv (export_file_path, index = True, header=True)





