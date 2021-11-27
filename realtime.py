import sys
import serial
sys.path.append('C:/Python37/Lib/site-packages')
from IPython.display import clear_output
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import  QMessageBox
from PyQt5.QtGui import QMovie
import pyqtgraph as pg
import random
from pyOpenBCI import OpenBCICyton
import threading
import time
import numpy as np
from scipy import signal
from pyqtgraph import PlotWidget
import os
import sys
from scipy.signal import butter, sosfilt, sosfreqz
from threading import Lock, Thread
import datetime
# Plot final
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from playsound import playsound

from GUI_DATOS import Ui_Datos_Paciente
import subprocess


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing



##########################################################
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget, GraphicsLayoutWidget
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
data = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
SCALE_FACTOR = (4500000)/24/(2**23-1) #From the pyOpenBCI repo
colors = 'bbbbbbbb'
i = 0
s = 0
m = 0
t = 0
fila = 0 
stm =0
inittimer = False
prueba = 0
Cfiltro = 0
LowF = 0
HighF = 0
features=[]
datafilt=[]
entrada=625
salida=4
## Conexión a bluetooth 
# puerto   = serial.Serial(port = 'COM9',
#                          baudrate = 9600,
#                          bytesize = serial.EIGHTBITS,
#                          parity   = serial.PARITY_NONE,
#                          stopbits = serial.STOPBITS_ONE)
# print("Bluetooth connected")


input_shape = (entrada,16)
inputs = keras.layers.Input(input_shape)
x = keras.layers.Conv1D(16, 250, activation='relu',input_shape=input_shape)(inputs)
x = keras.layers.MaxPool1D(pool_size=2, strides = 2, padding = 'same')(x)
x = keras.layers.Conv1D(16, 80, activation='relu',input_shape=input_shape)(x)
x = keras.layers.MaxPool1D(pool_size=2, strides = 2, padding = 'same')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128,activation = 'relu')(x)
x = keras.layers.Dense(256,activation = 'relu')(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(units=salida, activation= 'softmax')(x)

model = keras.Model(inputs = inputs, outputs=outputs, name ='model_sing')

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
model.load_weights('new83.h5')
#model.load_weights('pesos4.h5')
#model.load_weights('pesos1.h5')
#model.load_weights('pesos1(1).h5')
#model.load_weights('MejorPrueba_Normal.h5')
#model.load_weights('MejorPrueba_MinMax.h5')
#model.load_weights('MejorPrueba_sin_Scaling.h5')
#model.load_weights('MejorPrueba_Scaling7.h5')
#model.load_weights("pesos1.h5")


class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1339, 721)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("LOGOLASER.jpg"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Plot primeros 8 canales EEG
        self.graphicsView = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(180, 10, 371, 651))
        self.graphicsView.setObjectName("graphicsView")
        self.ts_plots = [self.graphicsView.addPlot(row=i, col=0, colspan=2, title='Channel %d' % i, labels={'left': 'uV'}) for i in range(1,9)]

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 660, 131, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.close_application)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 580, 131, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.inittimer)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 55, 141, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.clicked.connect(self.carpeta_gesto)

        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 15, 141, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(self.crear_paciente)

        self.lcdNumber = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber.setGeometry(QtCore.QRect(30, 330, 111, 61))
        self.lcdNumber.setSmallDecimalPoint(True)
        self.lcdNumber.setDigitCount(4)
        self.lcdNumber.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber.setObjectName("lcdNumber")
        self.lcdNumber.display("0:00")


        self.lcdNumber1 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber1.setGeometry(QtCore.QRect(130,293,32,25))
        self.lcdNumber1.setSmallDecimalPoint(True)
        self.lcdNumber1.setDigitCount(2)
        self.lcdNumber1.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber1.setObjectName("lcdNumber1")
        self.lcdNumber1.display("--")


        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 165, 81, 41))
        self.label.setObjectName("label")

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(20, 170, 71, 31))
        self.textEdit.setObjectName("textEdit")

        self.graphicsView_2 = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(570, 10, 371, 651))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.ts_plots_2 = [self.graphicsView_2.addPlot(row=i, col=0, colspan=2, title='Channel %d' % i, labels={'left': 'uV'}) for i in range(9,17)]

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 220, 141, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.filtros)

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 620, 131, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setEnabled(False)
        self.pushButton_5.clicked.connect(self.plot_final)


        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(310, 660, 101, 41))
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(690, 660, 101, 41))
        self.label_3.setObjectName("label_3")


        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 430, 131, 151))
        self.label_5.setObjectName("label_5")
        self.movie = QMovie('manos.gif')
        self.label_5.setMovie(self.movie)
        self.movie.start()

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(40, 400, 81, 41))
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(10,100,141,20))
        self.label_7.setObjectName("label_7")
        
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(15,295,141,20))
        self.label_8.setObjectName("label_8")
        
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(110,190,141,20))
        self.label_9.setObjectName("label_9")


        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(10, 130, 141, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        MainWindow.setCentralWidget(self.centralwidget)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "EEG  detection"))
        self.pushButton.setText(_translate("MainWindow", "SALIR"))
        self.pushButton_2.setText(_translate("MainWindow", "INICIO"))
        self.pushButton_3.setText(_translate("MainWindow", "INICIAR SESIÓN"))
        self.label.setText(_translate("MainWindow", "SEGUNDOS"))
        self.pushButton_4.setText(_translate("MainWindow", "SIN FILTRO"))
        self.pushButton_5.setText(_translate("MainWindow", "GRAFICAR"))
        self.pushButton_6.setText(_translate("MainWindow", "CREAR PACIENTE"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">EEG 8 Canales</p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">EEG 8 Canales</p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Timer</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow","<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Seleccione el gesto </span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow","<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Sesión numero: </span></p></body></html>"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Gesto_1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Gesto_2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Gesto_3"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Gesto_4"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Gesto_5"))
        self.comboBox.setItemText(5, _translate("MainWindow", "Gesto_6"))
##############################################################################################################################
    def crear_paciente(self):

        subprocess.Popen("python GUI_DATOS.py", shell=True)
        self.pushButton_3.setEnabled(True)

    
    def carpeta_gesto(self):
        global carpetaEEG 
        global  carpetaPaciente
        global j
        Archivo = True
        j = 1
        self.lcdNumber.display("0:00")
        self.pushButton_2.setEnabled(True)
        nombre_capeta = self.comboBox.currentText()
        with open('dato_carpeta.csv') as f:
            carpetaPaciente = f.read()  # add trailing new line character

        carpetaEEG = carpetaPaciente+ "/" + self.comboBox.currentText()+"/" + "Datos_CSV_ULTRACORTEX"
        if not os.path.exists(carpetaPaciente+ "/" +nombre_capeta):
            os.makedirs(carpetaPaciente + "/"+ self.comboBox.currentText()+"/" + "Datos_CSV_ULTRACORTEX" )
        while(Archivo == True):# Se crea un archivo csv en caso de que no exista
            if os.path.isfile(carpetaEEG + "/datos_%d.csv"% j):
                j+=1 
            else:
                with open(os.path.join(carpetaEEG, "datos_%d.csv"% j), 'w') as fp:
                    fp.write("Año;Mes;Dia; Hora")
                    fp.write("\n")
                    fp.write(datetime.datetime.now().strftime("%Y;%m;%d;%H-%M-%S"))
                    fp.write("\n")
                    [fp.write('CH%d ;'%i)for i in range(1,17)]
                    fp.write("\n")
                    self.lcdNumber1.display(str(j))
                    self.lcdNumber.repaint()
                    Archivo = False 
        Archivo = True            
        

    def save_data_EEG(self, sample):
        global data , fila , inittimer,entrada
        data.append([i*SCALE_FACTOR for i in sample.channels_data])
        fila+= 1
        muestreo=250
        # if (fila%muestreo)==0:
        #     print("5sec")
        #     dataG = data
        #     dataG = np.array(dataG).T
        #     dataG = self.butter_bandpass_filter(dataG, 8, 13, 125, order=5)
        #     dataG =dataG.T
        #     data_filt=dataG.T
        #     data_filt=data_filt[:,-entrada:]
        #     data_filt=np.array(data_filt).T
        #     if data_filt.shape==(entrada,16):
        #         #features=Escalar(features)
        #         #features=MinMax(features)
        #         #data_filt=Normalizar(data_filt)
        #         #features=self.Normalizar(features)       
        #         features= np.expand_dims(data_filt,0) 
        #         class_pred = model.predict(features, batch_size=32)
        #         class_pred = np.argmax(class_pred, axis=1)
        #         print(class_pred)  



    def Escalar(self,features_array):
        New_scaled= []
        for values in range(features_array.shape[0]):
            X_scaled = preprocessing.scale(features_array[values])
            New_scaled.append(X_scaled)
        len(New_scaled) ,New_scaled[0].shape
        New_scaled_array = np.array(New_scaled)
        New_scaled_array.shape
        return(New_scaled_array)

    def MinMax(self,features_array):
        New_scaled= []
        for values in range(features_array.shape[0]):
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train_minmax = min_max_scaler.fit_transform(features_array[values])
            New_scaled.append(X_train_minmax)
        len(New_scaled) ,New_scaled[0].shape
        New_scaled_array = np.array(New_scaled)
        New_scaled_array.shape
        return(New_scaled_array)


    def Normalizar(self,features_array):
        New_scaled= []
        for values in range(features_array.shape[0]):
            X_normalized = preprocessing.normalize(features_array[values], norm='l2')
            New_scaled.append(X_normalized)
        len(New_scaled) ,New_scaled[0].shape
        New_scaled_array = np.array(New_scaled)
        New_scaled_array.shape
        return(New_scaled_array)


    def save_csv_eeg(self):
        global prueba, fila, data, c, carpetaEEG,model,datafilt,features
        self.pushButton_3.setText("GUARDANDO")
        self.pushButton_3.setEnabled(False)
        time.sleep(1.5)
        self.pushButton_3.setText("INICIAR SESIÓN")
        self.pushButton_3.setEnabled(True)
        dataG = data
        dataG = np.array(dataG).T
        dataG = self.butter_bandpass_filter(dataG, 8, 13, 125, order=5)
        dataG =dataG.T
   
        data_filt=dataG.T
        data_filt=data_filt[:,prueba:(prueba+125*int(5))]
        #data_filt=data_filt[:,-427:]
        data_filt=np.array(data_filt).T
        print(data_filt.shape)

        with open(os.path.join(carpetaEEG, "datos_%d.csv"% j), 'a') as fp: # Guardar datos en el archivo csv        
            for  k in range(prueba, prueba+(125*int(5))):
                for i in range(0,16):
                    fp.write(str(dataG[k][i])+";")
                fp.write("\n")

        if data_filt.shape==(625,16):
            #features=Escalar(features)
            #features=MinMax(features)
            #data_filt=Normalizar(data_filt)
            #features=self.Normalizar(features)       
            features= np.expand_dims(data_filt,0) 
            class_pred = model.predict(features, batch_size=32)
            class_pred = np.argmax(class_pred, axis=1)
            if class_pred==0:
                print("Ojo izquierdo")
            if class_pred==1:
                print("Ojo derecho")
            if class_pred==2:
                print("Cejas")
            if class_pred==3:
                print("Nada")
            print(class_pred)
            cadena = str(class_pred)
            # try:
            #     puerto.write(cadena.encode())
            #     time.sleep(1)
            #     #puerto.close()
            # except serial.SerialException:
            #     print('Port is not available') 

            # except serial.portNotOpenError:
            #     print('Attempting to use a port that is not open')
            #     print('End of script')       




    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos


    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y
    def filtros(self):
        global LowF, HighF, Cfiltro
        if Cfiltro==0:
            LowF = 0
            HighF =0
            self.pushButton_4.setText("SIN FILTRO")
        elif  Cfiltro==1:
            LowF = 8
            HighF =13
            self.pushButton_4.setText("8-13 Hz")
        elif  Cfiltro==2:
            LowF = 8
            HighF =50
            self.pushButton_4.setText("8-50 Hz")
        Cfiltro +=1
        if Cfiltro == 3:
            Cfiltro= 0
        




    def updater_EEG(self):
        global data, colors,board,stm, LowF,HighF,features,model
        

        if board.read_state==0:
            stm += 1
            if stm==50:
                self.Errores("ULTRACORTEX DESCONECTADO")
        else:
            stm = 0
        
        t_data = np.array(data[-1250:]).T #transpose data
        data_filt = self.butter_bandpass_filter(t_data, 8, 13, 125, order=5)
     #   data_filt=data_filt[:,-625:]
     #  data_filt=np.array(data_filt).T
        if LowF!=0 or HighF != 0:
             t_data = self.butter_bandpass_filter(t_data, LowF, HighF, 125, order=5)
        # Plot a time series EEG of the raw data
        for j in range(8):
            # t_data[j] = self.butter_bandpass_filter(t_data[j], 8, 13, 125, order=5)
            self.ts_plots[j].clear()
            self.ts_plots[j].plot(pen=colors[j]).setData(t_data[j][400:])
        for k in range(8,16):
            # t_data[k] = self.butter_bandpass_filter(t_data[k], 8, 13, 125, order=5)
            self.ts_plots_2[k-8].clear()
            self.ts_plots_2[k-8].plot(pen=colors[1]).setData(t_data[k][400:]) 
        #self.ts_plots_2[7].plot(pen="r").setData(data_filt[0][:])    
    # Metodo Arranque MYO
        #if data_filt.shape==(625,16):
        #  class_pred = model.predict(features, batch_size=32)
        #    class_pred = np.argmax(class_pred, axis=1)
        #    print(class_pred)
        


    def close_application(self):
        sys.exit()

    def plot_final(self):
        global j , carpetaEEG 

        current_dir_eeg = os.path.dirname(os.path.realpath(__file__)) 
        filenameeeg = os.path.join(current_dir_eeg, carpetaEEG + "/datos_%d.csv" %j) 
        data_eeg = pd.read_csv(filenameeeg, delimiter=';', skiprows= 2)
        containerEEG = []
        for i in range(1,17):
            trace = (go.Scatter(y=data_eeg['CH%d '%i], showlegend=True, 
                                name = 'CH%d '%i))
            containerEEG.append(trace)
        layoutEEG = go.Layout(title='Señales EEG capturadas',
                        plot_bgcolor='rgb(230, 230,230)')
        figEEG = go.Figure(data=containerEEG, layout=layoutEEG)
        figEEG.write_html('EEG.html', auto_open=True)


    def inittimer(self):
        global inittimer, prueba
        if not self.textEdit.toPlainText():
            msg = QMessageBox()
            msg.setWindowTitle("Control")
            msg.setText("Atención!")
            msg.setIcon(QMessageBox.Warning)
            msg.setInformativeText("Por favor ingrese los segundos!")
            msg.exec()
        else:
            inittimer =True
            prueba = fila

            def hiloRunTimmer(arg):
                hiloRunTimmer = threading.currentThread()
                while getattr(hiloRunTimmer, "do_run", True):
                        print("working on %s" % arg)
                        self.OnTimer(None, e=self.textEdit.toPlainText())
                print("Stopping as you wish.")
            self.hiloRunTimmer = threading.Thread(target=hiloRunTimmer,args=("RUN_Timmer",))
            self.hiloRunTimmer.setDaemon(True)

            self.hiloRunTimmer.start()
            def hiloBiocina(arg):
                hiloRunTimmer = threading.currentThread()
                while getattr(hiloRunTimmer, "do_run", True):
                        print("working on %s" % arg)
                        playsound('Bocina.mp3')  
                print("Stopping as you wish.")
            self.hiloBiocina = threading.Thread(target=hiloBiocina,args=("RUN_Bocina",))

   
            


    
    def OnTimer(self, event, e):
        global i , inittimer, fila, prueba
        global c , s
        c = e
        if(i < int(c)):
            i += 1   
            if (i == 2):
                self.hiloBiocina.start()      
            if (i == 3):
                self.hiloBiocina.do_run = False
                self.hiloBiocina.join()                   
            time.sleep(1)
            self.TimerGo(None)
           
            
            
        else:
            self.hiloRunTimmer.do_run = False
            inittimer = False
            self.stopsaved= True
            i = 0 
            s = 0
            self.pushButton_2.setEnabled(False)
            self.pushButton_5.setEnabled(True)
            self.save_csv_eeg()   

    
    def TimerGo(self, event):
        global s
        global m
        global t
        global c
        s = int(s)
        m = int(m)
        if(s < 59):
            s += 1
        elif(s == 59):
            s = 0
            if(m < 59):
                m += 1
            elif(m == 59):
                m = 0
        if (int(s) < 10):
            s = str(0) + str(s)
        if(int(s) > 9):
            s = str(s)
        t = str(m) + ":" + str(s)
        self.lcdNumber.display(t)
        self.OnTimer(None, c)
    
    ######################### Metodos de aguardar datos
    def Errores(self,equipo):
        self.pushButton_6.setEnabled(False)
        self.label_2.setVisible(False)
        self.label_3.setFont(QtGui.QFont("Times", 15, QtGui.QFont.Bold))
        self.label_3.setStyleSheet('color: red')
        self.label_3.setGeometry(QtCore.QRect(500, 660, 101, 41))
        self.label_3.setText("ERROR DISPOSITIVO: "+str(equipo)) 
        self.label_3.adjustSize()

# Metodo Arranque Ultracortex
def start_board_Ultracortex():
    global board

    try:
        board = OpenBCICyton( "COM5", daisy= True)
        board.start_stream(ui.save_data_EEG)
    
    except:
        ui.Errores("DONGLE DESCONECTADO")


if __name__ == "__main__":
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'): ## verificar q python no se este corriendo en modo interactivo y tenga instalado pyqt5
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        hilo_conexion_ultracortes = threading.Thread(target=start_board_Ultracortex) 
        hilo_conexion_ultracortes.daemon = True
        hilo_conexion_ultracortes.start()
        time.sleep(3)
        timerEEG = QtCore.QTimer()
        timerEEG.timeout.connect(ui.updater_EEG)
        timerEEG.start(60)
        sys.exit(app.exec_())
