# This Python file uses the following encoding: utf-8
import sys
import pandas as pd
import io
import csv
import warnings
import os
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap,QIcon
from PyQt5.QtWidgets import  QMessageBox, QPushButton, QVBoxLayout, QWidget, QRadioButton, QComboBox
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QFile,QSize
from PyQt5 import QtWidgets, uic
#linear regresyon model
from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics

input1=0
input2=0
input3=0
input4=0
input5=0
input6=0
result=0

def dataset(input_data):
    #read data
    df = pd.read_csv('medical_cost.csv')
    #print(df.shape)
    #print(df.describe)
    #print(df.values)

    x = df.drop(columns = ['charges'])
    y = df['charges']
    #print(x)
    #print(y)

    #organize data
    x[['sex', 'smoker', 'region']] = x[['sex', 'smoker', 'region']].astype('category')
    #print(x.dtypes)

    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    label.fit(x.sex.drop_duplicates())
    x.sex = label.transform(x.sex)
    label.fit(x.smoker.drop_duplicates())
    x.smoker = label.transform(x.smoker)
    label.fit(x.region.drop_duplicates())
    x.region = label.transform(x.region)
    #print(x)

    if input_data == 0:
        return x
    elif input_data == 1:    
        return y


def input_csv_function():
    
    global input1,input2,input3,input4,input5,input6
    #os.remove('input_test.csv')

    input_age = input1
    input_sex = input2
    input_bmi = input3
    input_children = input4
    input_smoker = input5
    input_region = input6

    header = ['age','sex','bmi','children','smoker','region']
    data = [input_age, input_sex, input_bmi, input_children, input_smoker, input_region]

    with open('input_test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)
    

def convertion_csv(input_csv):
    print(input_csv['smoker'][0])
    

    if input_csv['smoker'][0] == 0:
        input_csv['smoker'][0] = 1
    else :
        input_csv['smoker'][0] = 0
"""
    if input_csv['region'][0] == 'Northeast':
        input_csv['region'][0] = 0
    elif input_csv['region'][0] == 'Southeast':
        input_csv['region'][0] = 1
    elif input_csv['region'][0] == 'Southwest':
        input_csv['region'][0] = 2
    else :
        input_csv['region'][0] = 3

    if input_csv['sex'][0] == 0:  
        input_csv['sex'][0] = 0
    else :
        input_csv['sex'][0] = 1
        """


def model_linear_regresyon():
   
    global result
    x = dataset(0)
    y = dataset(1)
    input_csv_function()


    model_lin_reg = LinearRegression()
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
    model_lin_reg.fit(x_train, y_train)


    #read data 
    x_input_csv = pd.read_csv('input_test.csv')
    print(x_input_csv)

    y_test_pred_input_csv = model_lin_reg.predict(x_input_csv)
    result=y_test_pred_input_csv




def model_polynomal_regresyon():
    global result
    x = dataset(0)
    y = dataset(1)
    input_csv_function()

    from sklearn.preprocessing import PolynomialFeatures
    x = x.drop([ 'sex', 'region'], axis = 1)
    #print(x)

    pol = PolynomialFeatures (degree = 2)
    x_pol = pol.fit_transform(x)
    x_train, x_test, y_train, y_test = holdout(x_pol, y, test_size=0.2, random_state=0)

    Pol_reg = LinearRegression()
    Pol_reg.fit(x_train, y_train)

    y_train_pred = Pol_reg.predict(x_train)
    print(y_train_pred)


    x_input_csv = pd.read_csv('input_test.csv')

    x_input_csv = x_input_csv.drop([ 'sex', 'region'], axis = 1)
    x_pol_test_csv = pol.fit_transform(x_input_csv)
    print(x_input_csv)
    
    y_test_pred_csv_out = Pol_reg.predict(x_pol_test_csv)
    result=y_test_pred_csv_out
  
   

def model_random_forest():
    global result
    x = dataset(0)
    y = dataset(1)
    input_csv_function()

    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)

    from sklearn.ensemble import RandomForestRegressor as rfr
    Rfr = rfr(n_estimators = 500, criterion = 'mse', #n_etimators 100 
                              random_state = 1,
                              n_jobs = -1)
    Rfr.fit(x_train,y_train)

    x_input_csv = pd.read_csv('input_test.csv')
    print(x_input_csv)

    y_test_pred_csv_out = Rfr.predict(x_input_csv)
    result=y_test_pred_csv_out

def model_decision_tree():
    global result
    x = dataset(0)
    y = dataset(1)
    input_csv_function()

    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)

    from sklearn.tree import DecisionTreeRegressor
    decision_tree = DecisionTreeRegressor(max_depth = 3)

    decision_tree.fit(x_train, y_train)

    x_input_csv = pd.read_csv('input_test.csv')
    print(x_input_csv)

    y_test_pred_csv_out = decision_tree.predict(x_input_csv)
    result=y_test_pred_csv_out

def model_knn():
    global result
    x = dataset(0)
    y = dataset(1)
    input_csv_function()
        
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)

    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=10)

    knn.fit(x_train,y_train)

    x_input_csv = pd.read_csv('input_test.csv')
    
    print(x_input_csv)

    y_test_pred_csv_out = knn.predict(x_input_csv)
    result=y_test_pred_csv_out


class main(QWidget):
    def __init__(self):
        super(main, self).__init__()
        uic.loadUi('form.ui',self)
        self.label_Result.setVisible(False)
        self.pushButton.clicked.connect(self.trainButtonClicked)
        tau_image = QPixmap("icons/tau.png")
        about_image = QIcon("icons/about.png")
        back_image=QIcon("icons/back.png")
        self.pushButton_about.setIcon(about_image)
        self.pushButton_back.setIcon(back_image)
        size = QSize(30, 30)
        self.pushButton_about.setIconSize(size)
        self.pushButton_back.setIconSize(size)
        self.label_logo.setPixmap(tau_image)
        self.listWidget.itemSelectionChanged.connect(self.selectionChanged)
        self.label_about.setVisible(False)
        self.pushButton_about.clicked.connect(self.aboutButtonClicked)
        self.pushButton_back.setVisible(False)
        self.pushButton_back.clicked.connect(self.backButtonClicked)
        

       

    
        
    def selectionChanged(self):
        """Mod seçimi liste değişimi"""
        print("Selected items: ", self.listWidget.currentRow())  
        
    def backButtonClicked(self):
        self.label_about.setVisible(False)
        self.pushButton_back.setVisible(False)
        self.comboBox_sex.setVisible(True)
        self.comboBox_region.setVisible(True)
        self.comboBox_smoker.setVisible(True)
        self.label_modell_auswahl.setVisible(True)
        self.label_parameter.setVisible(True)

    def aboutButtonClicked(self):
        global input3
        self.label_about.setVisible(True)
        about_resim = QPixmap("icons/about_resim.png")
        self.label_about.setPixmap(about_resim)

        self.pushButton_back.setVisible(True)
        self.comboBox_sex.setVisible(False)
        self.comboBox_region.setVisible(False)
        self.comboBox_smoker.setVisible(False)
        self.label_modell_auswahl.setVisible(False)
        self.label_parameter.setVisible(False)


    def trainButtonClicked(self): 
        """Train butonuna tıklanıldığında aktif olur"""
        self.label_Result.setVisible(True)
        global input1,input2,input3,input4,input5,input6,result
       
        model=self.listWidget.currentRow()
        input6=self.comboBox_region.currentIndex()
        input5=self.comboBox_smoker.currentIndex()
        input2=self.comboBox_sex.currentIndex()
        input1=int(self.lineEdit.text())
        input3=float(self.lineEdit_3.text())
        input4=int(self.lineEdit_4.text())
        print('input1: ',input1)
        print('input2: ',input2)
        print('input3: ',input3)
        print('input4: ',input4)
        print('input5: ',input5)
        print('input6: ',input6)
        
    

        if model == 0 :
            model_linear_regresyon()
        elif model == 1 :
            model_polynomal_regresyon()
        elif model == 2 :
            model_random_forest() 
        elif model == 3 :
            model_decision_tree()
        elif model == 4 :
            model_knn() 
        print("bu resulttaır: " , result[0])
        result = round(result[0])
        #result = str(result)
        #result = result.lstrip("[").rstrip("]")
        self.label_Result.setText("Schätzung: " + str(result) + " $ pro Jahr")     
      

if __name__ == "__main__":
    app = QApplication([])
    widget = main()
    widget.show()
    sys.exit(app.exec_())


""" data visualisation 
#Heatmap, hangi veri ne akdar etkili
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.heatmap(df.corr(), annot=True, cmap='cool')

"""