# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.0.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_main(object):
    def setupUi(self, main):
        if not main.objectName():
            main.setObjectName(u"main")
        main.resize(963, 620)
        main.setStyleSheet(u"")
        self.label = QLabel(main)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(220, 200, 41, 20))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.label.setFont(font)
        self.label_2 = QLabel(main)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(220, 230, 55, 16))
        self.label_2.setFont(font)
        self.label_3 = QLabel(main)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(220, 260, 55, 16))
        self.label_3.setFont(font)
        self.label_4 = QLabel(main)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(220, 290, 91, 16))
        self.label_4.setFont(font)
        self.label_5 = QLabel(main)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(220, 320, 71, 16))
        self.label_5.setFont(font)
        self.label_6 = QLabel(main)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(220, 350, 71, 21))
        self.label_6.setFont(font)
        self.label_logo = QLabel(main)
        self.label_logo.setObjectName(u"label_logo")
        self.label_logo.setGeometry(QRect(880, 10, 71, 61))
        self.label_logo.setAutoFillBackground(False)
        self.label_logo.setScaledContents(True)
        self.listWidget = QListWidget(main)
        QListWidgetItem(self.listWidget)
        QListWidgetItem(self.listWidget)
        QListWidgetItem(self.listWidget)
        QListWidgetItem(self.listWidget)
        self.listWidget.setObjectName(u"listWidget")
        self.listWidget.setGeometry(QRect(590, 220, 231, 121))
        self.listWidget.setStyleSheet(u"QListWidget::item:selected\n"
"{\n"
"    background: \n"
"	color: rgb(85, 170, 255);\n"
"}")
        self.pushButton = QPushButton(main)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(460, 400, 141, 41))
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setPointSize(16)
        font1.setUnderline(False)
        self.pushButton.setFont(font1)
        self.pushButton.setAcceptDrops(False)
        self.pushButton.setStyleSheet(u"QPushButton { color: #000000; border: 2px solid #555; border-radius: 20px; border-style: outset; background: qradialgradient( cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4, radius: 5, stop: 0 #7ddcff, stop: 1 #888 ); padding: 5px; }\n"
"color: rgb(125, 220, 255);")
        self.pushButton.setInputMethodHints(Qt.ImhNone)
        self.pushButton.setAutoRepeat(False)
        self.lineEdit = QLineEdit(main)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(350, 200, 121, 22))
        self.lineEdit_3 = QLineEdit(main)
        self.lineEdit_3.setObjectName(u"lineEdit_3")
        self.lineEdit_3.setGeometry(QRect(350, 260, 121, 22))
        self.lineEdit_4 = QLineEdit(main)
        self.lineEdit_4.setObjectName(u"lineEdit_4")
        self.lineEdit_4.setGeometry(QRect(350, 290, 121, 22))
        self.label_Title = QLabel(main)
        self.label_Title.setObjectName(u"label_Title")
        self.label_Title.setGeometry(QRect(180, 70, 621, 41))
        font2 = QFont()
        font2.setPointSize(18)
        font2.setBold(True)
        self.label_Title.setFont(font2)
        self.label_Title.setTextFormat(Qt.AutoText)
        self.label_Title.setScaledContents(False)
        self.label_Title.setAlignment(Qt.AlignCenter)
        self.label_Result = QLabel(main)
        self.label_Result.setObjectName(u"label_Result")
        self.label_Result.setGeometry(QRect(240, 470, 561, 61))
        font3 = QFont()
        font3.setPointSize(16)
        font3.setBold(True)
        self.label_Result.setFont(font3)
        self.label_Result.setAlignment(Qt.AlignCenter)
        self.pushButton_about = QPushButton(main)
        self.pushButton_about.setObjectName(u"pushButton_about")
        self.pushButton_about.setGeometry(QRect(770, 350, 51, 31))
        self.pushButton_about.setStyleSheet(u"border:none")
        self.pushButton_about.setIconSize(QSize(50, 50))
        self.label_about = QLabel(main)
        self.label_about.setObjectName(u"label_about")
        self.label_about.setGeometry(QRect(160, 160, 671, 361))
        self.label_about.setStyleSheet(u"border-style: double; border-width: 4px; border-color: rgb(125, 220, 255); border-radius: 4px; \n"
"")
        self.label_about.setScaledContents(True)
        self.pushButton_back = QPushButton(main)
        self.pushButton_back.setObjectName(u"pushButton_back")
        self.pushButton_back.setGeometry(QRect(160, 540, 61, 41))
        self.pushButton_back.setStyleSheet(u"border:none")
        self.pushButton_back.setIconSize(QSize(50, 50))
        self.comboBox_region = QComboBox(main)
        self.comboBox_region.addItem("")
        self.comboBox_region.addItem("")
        self.comboBox_region.addItem("")
        self.comboBox_region.addItem("")
        self.comboBox_region.setObjectName(u"comboBox_region")
        self.comboBox_region.setGeometry(QRect(350, 350, 121, 22))
        font4 = QFont()
        font4.setPointSize(10)
        self.comboBox_region.setFont(font4)
        self.comboBox_region.setStyleSheet(u"QComboBox { border: 1px solid gray; border-radius: 3px; padding: 1px 18px 1px 3px; min-width: 6em; }")
        self.comboBox_smoker = QComboBox(main)
        self.comboBox_smoker.addItem("")
        self.comboBox_smoker.addItem("")
        self.comboBox_smoker.setObjectName(u"comboBox_smoker")
        self.comboBox_smoker.setGeometry(QRect(350, 320, 121, 22))
        self.comboBox_smoker.setFont(font4)
        self.comboBox_smoker.setStyleSheet(u"QComboBox { border: 1px solid gray; border-radius: 3px; padding: 1px 18px 1px 3px; min-width: 6em; }")
        self.comboBox_sex = QComboBox(main)
        self.comboBox_sex.addItem("")
        self.comboBox_sex.addItem("")
        self.comboBox_sex.setObjectName(u"comboBox_sex")
        self.comboBox_sex.setGeometry(QRect(350, 230, 121, 22))
        self.comboBox_sex.setFont(font4)
        self.comboBox_sex.setStyleSheet(u"QComboBox { border: 1px solid gray; border-radius: 3px; padding: 1px 18px 1px 3px; min-width: 6em; }")
        self.label_modell_auswahl = QLabel(main)
        self.label_modell_auswahl.setObjectName(u"label_modell_auswahl")
        self.label_modell_auswahl.setGeometry(QRect(600, 200, 191, 16))
        self.label_modell_auswahl.setFont(font)
        self.label_parameter = QLabel(main)
        self.label_parameter.setObjectName(u"label_parameter")
        self.label_parameter.setGeometry(QRect(220, 170, 311, 21))
        self.label_parameter.setFont(font)
        self.label_7 = QLabel(main)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(380, 20, 261, 41))
        self.label_7.setFont(font2)

        self.retranslateUi(main)

        QMetaObject.connectSlotsByName(main)
    # setupUi

    def retranslateUi(self, main):
        main.setWindowTitle(QCoreApplication.translate("main", u"main", None))
        self.label.setText(QCoreApplication.translate("main", u"Age:", None))
        self.label_2.setText(QCoreApplication.translate("main", u"Sex:", None))
        self.label_3.setText(QCoreApplication.translate("main", u"BMI:", None))
        self.label_4.setText(QCoreApplication.translate("main", u"Children:", None))
        self.label_5.setText(QCoreApplication.translate("main", u"Smoker:", None))
        self.label_6.setText(QCoreApplication.translate("main", u"Region:", None))
        self.label_logo.setText("")

        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        ___qlistwidgetitem = self.listWidget.item(0)
        ___qlistwidgetitem.setText(QCoreApplication.translate("main", u"Linear Regression", None));
        ___qlistwidgetitem1 = self.listWidget.item(1)
        ___qlistwidgetitem1.setText(QCoreApplication.translate("main", u"Polynomal Regression", None));
        ___qlistwidgetitem2 = self.listWidget.item(2)
        ___qlistwidgetitem2.setText(QCoreApplication.translate("main", u"Random Forest", None));
        ___qlistwidgetitem3 = self.listWidget.item(3)
        ___qlistwidgetitem3.setText(QCoreApplication.translate("main", u"Decision Tree", None));
        self.listWidget.setSortingEnabled(__sortingEnabled)

        self.pushButton.setText(QCoreApplication.translate("main", u"Sch\u00e4tzen", None))
        self.label_Title.setText(QCoreApplication.translate("main", u" Sch\u00e4tzung von medizinischem Aufwand", None))
        self.label_Result.setText(QCoreApplication.translate("main", u"Result", None))
        self.pushButton_about.setText("")
        self.label_about.setText("")
        self.pushButton_back.setText("")
        self.comboBox_region.setItemText(0, QCoreApplication.translate("main", u"Northeast", None))
        self.comboBox_region.setItemText(1, QCoreApplication.translate("main", u"Southeast", None))
        self.comboBox_region.setItemText(2, QCoreApplication.translate("main", u"Southwest", None))
        self.comboBox_region.setItemText(3, QCoreApplication.translate("main", u"Northwest", None))

        self.comboBox_smoker.setItemText(0, QCoreApplication.translate("main", u"No", None))
        self.comboBox_smoker.setItemText(1, QCoreApplication.translate("main", u"Yes", None))

        self.comboBox_sex.setItemText(0, QCoreApplication.translate("main", u"Male", None))
        self.comboBox_sex.setItemText(1, QCoreApplication.translate("main", u"Female", None))

        self.label_modell_auswahl.setText(QCoreApplication.translate("main", u"W\u00e4hlen Sie ein Modell aus", None))
        self.label_parameter.setText(QCoreApplication.translate("main", u"Geben Sie den Inputparametern ein", None))
        self.label_7.setText(QCoreApplication.translate("main", u"INF 523 Projekt", None))
    # retranslateUi

