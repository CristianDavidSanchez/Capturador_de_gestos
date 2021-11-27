# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI_DATOS_PACIENTE.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

import os
from PyQt5.QtWidgets import  QMessageBox

import subprocess

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import pandas as pd


class Ui_Datos_Paciente(object):

    
    def setupUi(self, Datos_Paciente):
        Datos_Paciente.setObjectName("Datos_Paciente")
        Datos_Paciente.resize(881, 771)
        self.centralwidget = QtWidgets.QWidget(Datos_Paciente)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(320, 0, 221, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 60, 751, 71))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(200, 160, 261, 16))
        self.label_3.setObjectName("label_3")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(480, 150, 151, 31))
        self.textEdit.setStyleSheet("font: 14pt \".SF NS Text\";")
        self.textEdit.setObjectName("textEdit")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(330, 260, 131, 16))
        self.label_4.setObjectName("label_4")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(480, 260, 151, 31))
        self.textEdit_2.setStyleSheet("font: 14pt \".SF NS Text\";")
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(300, 370, 161, 16))
        self.label_5.setObjectName("label_5")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(480, 370, 151, 31))
        self.textEdit_3.setStyleSheet("font: 14pt \".SF NS Text\";")
        self.textEdit_3.setObjectName("textEdit_3")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(480, 200, 151, 41))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(310, 210, 141, 20))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(130, 450, 371, 91))
        self.label_7.setObjectName("label_7")
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(480, 470, 151, 41))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(110, 610, 361, 61))
        self.label_8.setObjectName("label_8")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(510, 630, 51, 20))
        self.radioButton.setObjectName("radioButton")
        self.radioButton.clicked.connect(self.crear_paciente)
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(570, 630, 51, 20))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton.setEnabled(False)
        self.radioButton_2.setEnabled(False)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setEnabled(False)
        self.pushButton.setGeometry(QtCore.QRect(520, 680, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.enviar_carpeta)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 680, 113, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.close_application)
        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(480, 310, 151, 31))
        self.textEdit_4.setStyleSheet("font: 14pt \".SF NS Text\";")
        self.textEdit_4.setObjectName("textEdit_4")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(290, 310, 171, 20))
        self.label_9.setObjectName("label_9")
        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setGeometry(QtCore.QRect(480, 420, 151, 41))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(210, 430, 261, 20))
        self.label_11.setObjectName("label_11")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(475, 540, 173, 25))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.showpdf)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(210, 545, 261, 20))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(170, 580, 261, 20))
        self.label_13.setObjectName("label_13")

        self.radioButton_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_3.setGeometry(QtCore.QRect(510, 580, 51, 20))
        self.radioButton_3.setObjectName("radioButton")
        self.radioButton_3.clicked.connect(self.consentimiento)
        self.radioButton_4 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_4.setGeometry(QtCore.QRect(570, 580, 51, 20))
        self.radioButton_4.setObjectName("radioButton_2")

        Datos_Paciente.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Datos_Paciente)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 881, 22))
        self.menubar.setObjectName("menubar")
        Datos_Paciente.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Datos_Paciente)
        self.statusbar.setObjectName("statusbar")
        Datos_Paciente.setStatusBar(self.statusbar)

        self.retranslateUi(Datos_Paciente)
        QtCore.QMetaObject.connectSlotsByName(Datos_Paciente)


    def retranslateUi(self, Datos_Paciente):
        _translate = QtCore.QCoreApplication.translate
        Datos_Paciente.setWindowTitle(_translate("Datos_Paciente", "Información del paciente"))
        self.label.setText(_translate("Datos_Paciente", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:600;\">Datos del Paciente</span></p></body></html>"))
        self.label_2.setText(_translate("Datos_Paciente", "<html><head/><body><p align=\"justify\"><span style=\" font-size:14pt;\">A continuación debe ingresar una información básica que es relevanta para el estudio de las</span></p><p align=\"justify\"><span style=\" font-size:14pt;\">señales, por favor ingrese la siguiente información:</span></p></body></html>"))
        self.label_3.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Número de identificación del estudio:</span></p></body></html>"))
        self.label_4.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Edad del paciente:</span></p></body></html>"))
        self.label_5.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Peso (Kg) del paciente:</span></p></body></html>"))
        self.comboBox.setItemText(0, _translate("Datos_Paciente", "Femenino"))
        self.comboBox.setItemText(1, _translate("Datos_Paciente", "Masculino"))
        self.comboBox.setItemText(2, _translate("Datos_Paciente", "Prefiero no decir"))
        self.label_6.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Género del paciente:</span></p></body></html>"))
        self.label_7.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Presenta alguna condición de </span></p><p><span style=\" font-size:10pt; font-weight:600;\">discapacidad motora o perdida de miembro superior :</span></p></body></html>"))
        self.comboBox_2.setItemText(0, _translate("Datos_Paciente", "Si"))
        self.comboBox_2.setItemText(1, _translate("Datos_Paciente", "No"))
        self.label_8.setText(_translate("Datos_Paciente", "<html><head/><body><p align=\"justify\"><span style=\" font-weight:600;\">Autoriza al equipo de investigador exponer sus datos </span></p><p align=\"justify\"><span style=\" font-weight:600;\">consignados con anterioridad a la comunidad cientifica: </span></p><p align=\"justify\"><span style=\" font-weight:600;\"><br/></span></p></body></html>"))
        self.radioButton.setText(_translate("Datos_Paciente", "Si"))
        self.radioButton_2.setText(_translate("Datos_Paciente", "No"))
        self.radioButton_3.setText(_translate("Datos_Paciente", "Si"))
        self.radioButton_4.setText(_translate("Datos_Paciente", "No"))
        self.pushButton.setText(_translate("Datos_Paciente", "SIGUIENTE"))
        self.pushButton_2.setText(_translate("Datos_Paciente", "CANCELAR"))
        self.pushButton_3.setText(_translate("Datos_Paciente", "VER CONSENTIMIENTO"))
        self.label_9.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Altura del paciente (cm):</span></p></body></html>"))
        self.comboBox_3.setItemText(0, _translate("Datos_Paciente", "Derecho"))
        self.comboBox_3.setItemText(1, _translate("Datos_Paciente", "Izquierdo"))
        self.label_11.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Ubicación del brazalete en el paciente:</span></p></body></html>"))
        self.label_12.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Consentimiento informado del estudio:</span></p></body></html>"))
        self.label_13.setText(_translate("Datos_Paciente", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Leyó con cuidado el consentimiento informado:</span></p></body></html>"))
    
    def close_application(self):
        sys.exit()
        
    def closeEvent(self, event):
        pass

    def consentimiento(self):
        self.radioButton.setEnabled(True)
        self.radioButton_2.setEnabled(True)
    def showpdf(self):
        subprocess.call(['open', "consentimiento.pdf"])

    def enviar_carpeta(self):
        global carpetaPaciente 
        with open(os.path.join( "dato_carpeta.csv"), 'w') as fp:
            fp.write(carpetaPaciente)
        sys.exit()

    def crear_paciente(self):
        if not (self.textEdit_3.toPlainText() and self.textEdit.toPlainText() and self.textEdit_2.toPlainText() and self.textEdit_4.toPlainText())  :
            print("El QLineEdit esta vacio")
            msg = QMessageBox()
            msg.setWindowTitle("Control")
            msg.setText("Atención!")
            msg.setIcon(QMessageBox.Warning)
            msg.setInformativeText("Por favor ingrese todos los campos!")
            msg.exec()
            self.radioButton.setAutoExclusive(False)
            self.radioButton.setChecked(False)
        else:
            bmi_paciente =  (int(self.textEdit_3.toPlainText())/ (int(self.textEdit_4.toPlainText())/100)**2)
            global carpetaPaciente 
            print(self.comboBox.currentText())
            carpetaPaciente = f"Paciente_ID_{self.textEdit.toPlainText()}" #Creacion de carpetas para guarda archivos si no existe
            if not os.path.exists(carpetaPaciente):
                os.mkdir(carpetaPaciente)
                with open(os.path.join(carpetaPaciente, "datos_paciente.txt"), 'w') as fp:
                    fp.write("Número de identificación del estudio: %s" %self.textEdit.toPlainText())
                    fp.write("\n")
                    fp.write("Género del paciente: %s" %self.comboBox.currentText())
                    fp.write("\n")
                    fp.write("Edad del paciente: %s" %self.textEdit_2.toPlainText())
                    fp.write("\n")
                    fp.write("Altura del paciente (cm): %s" %self.textEdit_4.toPlainText())
                    fp.write("\n")
                    fp.write("Peso (Kg) del paciente: %s" %self.textEdit_3.toPlainText())
                    fp.write("\n")
                    fp.write("BMI del paciente: %s" %str(bmi_paciente))
                    fp.write("\n")
                    fp.write("Ubicación del brazalete en el paciente: %s" %self.comboBox_3.currentText())
                    fp.write("\n")
                    fp.write("Paciente con condición de discapacidad: %s" %self.comboBox_2.currentText())
                    fp.write("\n")
                    fp.write("Frecuencia de muestreo de la MYO: 200 (Hz)")
                    fp.write("\n")
                    fp.write("Frecuencia de muestreo del Ultracortex (Cyton + Daisy): 125 (Hz)")

            self.pushButton.setEnabled(True)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Datos_Paciente = QtWidgets.QMainWindow()
    ui = Ui_Datos_Paciente()
    ui.setupUi(Datos_Paciente)
    Datos_Paciente.show()
    sys.exit(app.exec_()) 
