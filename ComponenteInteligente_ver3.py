# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 02:01:05 2019

@author: Camilo
"""

# -*- coding: utf-8 -*-

import time
import pyedflib
import numpy as np
import scipy
from scipy import stats
import statistics as stats
import warnings
import pywt
import math
from scipy.fftpack import fft
import mne
import os
from datetime import datetime
import requests
import sys
# Entropia Shannon
def shannon(a):
    
    #calculada con la funcion al cuadrado, problema con los negativos
    #problema con 0 por eso se suma 1
    v=a*a + 1
    z=v*[math.log10(i) for i in v]
    return (-1.0*z.sum())

def energia(canal):
    
    cuad=[0]*len(canal)
    for i in range(0,len(canal)):
            cuad[i]=canal[i]*canal[i]
    
    energia=0
    for i in cuad:
        energia=energia+i
    return energia

def hjorth(a):
    
    activity = np.mean(a ** 2)
        
    return activity

def renyi(c):
    d=c*c
    suma=d.sum()+1
    return (-1.0*math.log10(suma))



def hurst(signal):
    
    tau = []; lagvec = []

    #  Step through the different lags
    for lag in range(2,18):

    #  produce price difference with lag
        pp = np.subtract(signal[lag:],signal[:-lag])

    #  Write the different lags into a vector
        lagvec.append(lag)

    #  Calculate the variance of the difference vector
        tau.append(np.std(pp))
        
    #  linear fit to double-log graph (gives power)
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)

    # calculate hurst
    hurst = m[0]

    return hurst

def ApEn(U, m, r):
    
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))



def featurescalculator(sigbufs,n):
    
    
    Desface=12000    
    x=0+Desface  # desface en # de muestras donde inicia el examen
    aumento=0

    segundos=int((len(sigbufs[3, :])-Desface)/200) # numero de segundos del examen
    Features=np.empty(((n-5)*segundos,32)) # matriz de caracteristicas
    
    for a in np.arange(segundos):
        for i in np.arange((n-5)):      #n-2  numero de canales
            
            warnings.filterwarnings("ignore")
            #TIEMPO
            minimo=scipy.stats.tmin(sigbufs[i, x:x+200])
            maximo=scipy.stats.tmax(sigbufs[i, x:x+200])
            kurto=scipy.stats.kurtosis(sigbufs[i, x:x+200])
            energ=energia(sigbufs[i, x:x+200])
            sha=shannon(sigbufs[i, x:x+200])
            #DWT
            cA5,cD4,cD3,cD2,cD1 = pywt.wavedec(sigbufs[i, x:x+200], 'db4', level=4)
            varianzaA5=stats.variance(cA5)
            energA5=energia(cA5)
            shaA5=shannon(cA5)
            actiA5=hjorth(cA5)
            varianzaD4=stats.variance(cD4)
            energD4=energia(cD4)
            rD4=renyi(cD4)
            shaD4=shannon(cD4)
            EHD4=hurst(cD4)
            actiA4=hjorth(cD4)
            varianzaD3=stats.variance(cD3)
            desviacionD3=stats.stdev(cD3)
            energD3=energia(cD3)
            rD3=renyi(cD3)
            apenD3=ApEn(cD3, 2, 3)
            shaD3=shannon(cD3)
            minimoD2=scipy.stats.tmin(cD2)
            maximoD2=scipy.stats.tmax(cD2)
            desviacionD2=stats.stdev(cD2)
            kurtoD2=scipy.stats.kurtosis(cD2)
            energD2=energia(cD2)
            rD2=renyi(cD2)
            shaD2=shannon(cD2)
            minimoD1=scipy.stats.tmin(cD1)
            maximoD1=scipy.stats.tmax(cD1)
            rD1=renyi(cD1)
            #FFT
            nee=len(sigbufs[i, x:x+200]) # tamaño  
            Y=fft(sigbufs[i, x:x+200])/nee
            Yn=abs(Y)
            mediaf=stats.mean(Yn)
                                             
            #print (signal_labels[i]) 
        
            Features[i+aumento]=[minimo,maximo,kurto,energ,sha,varianzaA5,energA5,shaA5,actiA5,varianzaD4,energD4,rD4,shaD4,EHD4,actiA4,varianzaD3,desviacionD3,energD3,rD3,apenD3,shaD3,minimoD2,maximoD2,desviacionD2,kurtoD2,energD2,rD2,shaD2,minimoD1,maximoD1,rD1,mediaf]
            #Labels=signal_labels[i]
            #print (Labels)
        
        
        x=x+200
        aumento=aumento+18 ##16 -- n-2, 21 -- n-4, 15  -- n-3, 19  -- n-4,   43 ---- n-8


   
    return Features


def GetTime(sec): 
    from datetime import timedelta 
    secf = timedelta(seconds=sec)
     
          
    return secf


############################################################33
### CREAr edf anotaciones
    
def write_edf(mne_raw, fname, vector,picks=None, tmin=0, tmax=None, overwrite=False):
    
    
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
    # static settings
    file_type = pyedflib.FILETYPE_EDFPLUS 
    sfreq = mne_raw.info['sfreq']
    date = datetime.now().strftime( '%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None

    
    # convert data
    channels = mne_raw.get_data(picks, 
                                start = first_sample,
                                stop  = last_sample)
    
    # convert to microvolts to scale up precision
    channels *= 1e6
    
    # set conversion parameters
    dmin, dmax = [-32768,  32767]
    pmin, pmax = [channels.min(), channels.max()]
    n_channels = len(channels)
    
    # create channel from this   
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels, 
                               file_type=file_type)
        
        channel_info = []
        data_list = []
        
        for i in range(n_channels):
            ch_dict = {'label': mne_raw.ch_names[i], 
                       'dimension': 'uV', 
                       'sample_rate': sfreq, 
                       'physical_min': pmin, 
                       'physical_max': pmax, 
                       'digital_min':  dmin, 
                       'digital_max':  dmax, 
                       'transducer': '', 
                       'prefilter': ''}
        
            channel_info.append(ch_dict)
            data_list.append(channels[i])
        f.set_number_of_annotation_signals(2)
        for i in vector:
            f.writeAnnotation(i,1,str(i),str_format='utf-8')
            
        
        f.setTechnician('mne-gist-save-edf-skjerns')
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(data_list)
        
        
    except Exception as e:
        print(e)
        return False
    finally:
        f.close()    
    return True


##############################    




def main(arg1):
    tic = time.clock() # MEDIR EL TIEMPO DE COMPILACION
   
    
    # descargar el archivo .edf
    
    url = arg1
    myfile = requests.get(url)

    open('EDF_DESCARGADO.edf', 'wb').write(myfile._content)
    
    
    # IMPORTAR EXAMEN .EDF
    Nombre_Direccion="EDF_DESCARGADO.edf"
    f = pyedflib.EdfReader(Nombre_Direccion)


    n = f.signals_in_file # numero de canales del examen (incluidas EKG, etc)
    #signal_labels = f.getSignalLabels() # nombre de canal
    sigbufs = np.empty((n, f.getNSamples()[0]))
    # VECTOR DE CANALES DESEADOS
    h=0
    for i in np.arange(n):
        h=h+1
        if h<22:
            sigbufs[i, :] = f.readSignal(h)
        else:
            h=22
            sigbufs[i, :] = f.readSignal(h)
            
    # CALCULAR CARACTERISTICAS
    Features=featurescalculator(sigbufs,n)
    
    #print (Features)
    
    
        
    # NORMALIZAR CARACTERISTICAS
    import pickle
    scalerfile = 'Normalizacion.sav'
    scalerNew = pickle.load(open(scalerfile, 'rb'))
    X3=scalerNew.transform(Features)
    #print (X3)
    
        
    # CARGAR MODELO DE CLASIFICACION
    filename = 'MODEL.sav'
    model = pickle.load(open(filename, 'rb'))
    
    y_pred = model.predict(X3)
    
        
            
            
    # EXTRAER LOS CANALES ANORMALES, NUMERO DE PAGINA
    count=1
    count_page=1
    
    Segundos_Anotaciones=[]
      
    anormalidades=0
    tiempo=1
    n=1
    for var in y_pred:
        
        if count_page >n*18:
            tiempo=tiempo+1
            n=n+1
        if count>18:
            count=1
        
        if var=='A':
            #print ("Diagnostico: ",var)
            #print ("Canal: ",signal_labels[count])
            #print ("Pagina: ",Pagina)
            
            seg_anot=(tiempo+60) #### +60
            anormalidades=anormalidades+1
            #print ("tiempo A:", tiempo)
            
            
            #print ("hora: ",sumdt.time())
           
            
            Segundos_Anotaciones.append(seg_anot)
                      
        count=count+1
                    
        count_page=count_page+1
        #print (count_page)
    #CSV DE DIAGNOSTICO
    
   
    ### solo los segundos no repetidos
    Segundos_Anotaciones2=set(Segundos_Anotaciones)
            
        
    ### edf anotaciones 
    
    
    edf=mne.io.read_raw_edf(Nombre_Direccion,stim_channel='auto', preload=True )
    
    write_edf(edf,'EDF_Anotado.edf',Segundos_Anotaciones2)    


    
    #CERRAR EXAMEN
    f._close()
    
    toc = time.clock()
    print ("Tiempo de Compilacion:", toc - tic, " segundos ", (toc-tic)/60, " min")





    
if __name__ == '__main__':
    main(sys.argv[1])
    
