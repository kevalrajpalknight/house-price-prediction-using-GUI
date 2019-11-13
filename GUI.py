# -*- coding: utf-8 -*-
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import pandas as pd


class Window(QWidget):
    def __init__(self):
        train = pd.read_csv("house-price\\train.csv")
        self.__top_feature = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF',
       '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea']
        
        
        
        train['PoolQC'] = train['PoolQC'].fillna('None')
        
        train['MiscFeature'] = train['MiscFeature'].fillna('None')
        train['Alley'] = train['Alley'].fillna('None')
        train['Fence'] = train['Fence'].fillna('None')
        train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
        train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))
        
        for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            train[col] = train[col].fillna('None')
        
        for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
            train[col] = train[col].fillna(int(0))
        

        for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
            train[col] = train[col].fillna('None')
            
        train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))
        
        train['MasVnrType'] = train['MasVnrType'].fillna('None')
        
        train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]
        
        train = train.drop(['Utilities'], axis=1)
        
              
        for c in self.__top_feature:
            lbl = LabelEncoder() 
            lbl.fit(list(train[c].values)) 
            train[c] = lbl.transform(list(train[c].values))
        
        self.__X = train[self.__top_feature]
        self.__y = train['SalePrice']
        
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X,self.__y,test_size=0.3, random_state=7)
        self.model = LinearRegression()
        self.model_pred_ = 0
  
        self.model.fit(self.__X_train,self.__y_train)
        
        QWidget.__init__(self)
        self.setWindowTitle("House Price Prediction")
        self.showMaximized()
        self.icon = QIcon("brain_and_processor.jpg")
        self.setWindowIcon(self.icon)
        
        layout = QGridLayout()
        self.setLayout(layout)
        layout.setSpacing(20)
        
        
        self.__fireplace_label = QLabel("How many Fire Places")
        layout.addWidget(self.__fireplace_label,1,0,alignment=Qt.AlignLeft)
        
        self.__fireplace_combo = QComboBox()
        self.__fireplace_combo.addItems(["4","3","2","1"])
        layout.addWidget(self.__fireplace_combo,1,1,alignment=Qt.AlignLeft)
        self.__fireplace = self.__fireplace_combo.currentText()
        
        
        self.__overallqual_label = QLabel("How's Over All Quality")
        layout.addWidget(self.__overallqual_label,2,0,alignment=Qt.AlignLeft)
        
        self.__overallqual_combo = QComboBox()
        self.__overallqual_combo.addItems(["10","9","8","7","6","5","4","3","2","1"])
        layout.addWidget(self.__overallqual_combo,2,1,alignment=Qt.AlignLeft)
        self.__overallqual = self.__overallqual_combo.currentText()
                
        
        self.__year_built_label = QLabel("Year Built")
        layout.addWidget(self.__year_built_label,3,0,alignment=Qt.AlignLeft)
        
        self.__year_built_calender = QDateEdit()
        self.__year_built_calender.setCurrentSection(QDateTimeEdit.YearSection)
        self.__year_built_calender.setDisplayFormat("yyyy")
        layout.addWidget(self.__year_built_calender,3,1,alignment=Qt.AlignLeft)
        self.__year_built = self.__year_built_calender.date()
        
        
        self.__remodal_label = QLabel("Last Year ReModal Year")
        layout.addWidget(self.__remodal_label,4,0,alignment=Qt.AlignLeft)
        
        self.__remodal_calender = QDateEdit()
        self.__remodal_calender.setCurrentSection(QDateTimeEdit.YearSection)
        self.__remodal_calender.setDisplayFormat("yyyy")
        layout.addWidget(self.__remodal_calender,4,1,alignment=Qt.AlignLeft)
        self.__remodal = self.__remodal_calender.date()
        
        
        self.__fullbath_label = QLabel("How many Full Bath")
        layout.addWidget(self.__fullbath_label,5,0,alignment=Qt.AlignLeft)
        
        self.__fullbath_combo = QComboBox()
        self.__fullbath_combo.addItems(["4","3","2","1"])
        layout.addWidget(self.__fullbath_combo,5,1,alignment=Qt.AlignLeft)
        self.__fullbath = self.__fullbath_combo.currentText()
        
        
        
        self.__mas_vnr_area_label = QLabel("Masonry Veneer Area")
        layout.addWidget(self.__mas_vnr_area_label,6,0,alignment=Qt.AlignLeft)
        
        self.__mas_vnr_area_text = QLineEdit()
        layout.addWidget(self.__mas_vnr_area_text,6,1,alignment=Qt.AlignLeft)
        self.__mas_vnr_area = self.__mas_vnr_area_text.text()
        
        self.__Tbsm_area_label = QLabel("Total Basement Area")
        layout.addWidget(self.__Tbsm_area_label,7,0,alignment=Qt.AlignLeft)
        
        self.__Tbsm_area_text = QLineEdit()
        layout.addWidget(self.__Tbsm_area_text,7,1,alignment=Qt.AlignLeft)
        self.__Tbsm_area = self.__Tbsm_area_text.text()
        
        
        self.__tot_rms_abv_grd_label = QLabel("Total Room (Does no have bathroom)")
        layout.addWidget(self.__tot_rms_abv_grd_label,8,0,alignment=Qt.AlignLeft)
        
        self.__tot_rms_abv_grd_text = QLineEdit()
        layout.addWidget(self.__tot_rms_abv_grd_text,8,1,alignment=Qt.AlignLeft)
        self.__tot_rms_abv_grd = self.__tot_rms_abv_grd_text.text()
        
        
        self.__garageYr_label = QLabel("Garrage Built Year")
        layout.addWidget(self.__garageYr_label,9,0,alignment=Qt.AlignLeft)
        
        self.__garageYr_calender = QDateEdit()
        self.__garageYr_calender.setCurrentSection(QDateTimeEdit.YearSection)
        self.__garageYr_calender.setDisplayFormat("yyyy")
        layout.addWidget(self.__garageYr_calender,9,1,alignment=Qt.AlignLeft)
        self.__garageYr = self.__garageYr_calender.date()
        
        
        
        
        self.__grLivArea_label = QLabel("Ground Living Area")
        layout.addWidget(self.__grLivArea_label,10,0,alignment=Qt.AlignLeft)
        
        self.__grLivArea_text = QLineEdit()
        layout.addWidget(self.__grLivArea_text,10,1,alignment=Qt.AlignLeft)
        self.__grLivArea = self.__grLivArea_text.text()
        
        
        
        
        self.__garageArea_label = QLabel("Garage Area")
        layout.addWidget(self.__garageArea_label,11,0,alignment=Qt.AlignLeft)
        
        self.__garageArea_text = QLineEdit()
        layout.addWidget(self.__garageArea_text,11,1,alignment=Qt.AlignLeft)
        self.__garageArea = self.__garageArea_text.text()
        
        
        self.__firstflrArea_label = QLabel("First Floor Area")
        layout.addWidget(self.__firstflrArea_label,12,0,alignment=Qt.AlignLeft)
        
        self.__firstflrArea_text = QLineEdit()
        layout.addWidget(self.__firstflrArea_text,12,1,alignment=Qt.AlignLeft)
        self.__firstflrArea = self.__firstflrArea_text.text()
        
        
        
        
        self.__garage_car_label = QLabel("Garage Car Parking")
        layout.addWidget(self.__garage_car_label,13,0,alignment=Qt.AlignLeft)
        
        self.__garage_car_combo = QComboBox()
        self.__garage_car_combo.addItems(["4","3","2","1","0"])
        layout.addWidget(self.__garage_car_combo,13,1,alignment=Qt.AlignLeft)
        self.__garage_car = self.__garage_car_combo.currentText()
        
       
        self.__submit_btn = QPushButton("Check Price")
        self.__submit_btn.clicked.connect(self.check_price)
        layout.addWidget(self.__submit_btn,14,1,alignment=Qt.AlignCenter)
        
        self.__price_label = QLabel("")
        layout.addWidget(self.__price_label,15,1,1,2,alignment=Qt.AlignCenter)
    
    @pyqtSlot()
    def check_price(self):
        new_values = [self.__overallqual_combo.currentText(), self.__year_built_calender.date().year(), self.__remodal_calender.date().year(), self.__mas_vnr_area_text.text(), self.__Tbsm_area_text.text(),
       self.__firstflrArea_text.text(), self.__grLivArea_text.text(), self.__fullbath_combo.currentText(), self.__tot_rms_abv_grd_text.text(), self.__fireplace_combo.currentText(),
       self.__garageYr_calender.date().year(), self.__garage_car_combo.currentText(), self.__garageArea_text.text()]
        
        data_entry = [int(new_values[i]) for i in range(len(new_values))]
        data_entry = np.array(data_entry).reshape(1,-1)
        self.model_pred_ = self.model.predict(data_entry)
        self.__price_label.setText(str("Rs. %.2f" % self.model_pred_))
        

app = QApplication(sys.argv)
screen = Window()
screen.show()
sys.exit(app.exec_())
