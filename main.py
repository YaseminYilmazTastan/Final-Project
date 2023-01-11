# import library which will use in project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#read to dataset
diamond = pd.read_csv('C:/Users/ENİS/Downloads/GOOGLE DATASET/Diamonds Prices2022.csv')
#preprocessing in our data ,drop unnecessery data from dataset
diamond= diamond.drop(["Unnamed: 0"], axis=1)

#there are some values of x,y,z that are 0. we need to remove them
diamond=diamond.drop(diamond[diamond["x"]== 0].index)
diamond=diamond.drop(diamond[diamond["y"]== 0].index)
diamond=diamond.drop(diamond[diamond["z"]== 0].index)

diamond.columns
#Index(['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z'],  dtype='object')

by= LabelEncoder()
diamond.iloc[:,1]=by.fit_transform(diamond.iloc[:,1])
diamond.iloc[:,2]=by.fit_transform(diamond.iloc[:,2])
diamond.iloc[:,3]=by.fit_transform(diamond.iloc[:,3])
diamond.head()

cols=["carat","cut","color","clarity","depth","table","price","x","y","z"]
ml= MinMaxScaler()
diamond=pd.DataFrame(ml.fit_transform(diamond),columns= cols)

#Splitting  columns
X= diamond.drop(["price"],axis=1)
Y=diamond.iloc[:,6]

#divide data two group which are test and train
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2)
# apply regression our data
lr= LinearRegression()
lrp= lr.fit(x_train,y_train)
predicted_value=lrp.predict(x_test)
df=pd.DataFrame({'Actual ': y_test, 'Predicted': predicted_value })

#to see the result  result
print(df)


#the result
###        Actual   Predicted
#22424  0.547981   0.519670
#34236  0.028653   0.095202
#15799  0.324269   0.328704
#34017  0.028329  -0.047582
#25092  0.719306   0.583025
#...         ...        ...
#30020  0.021193   0.001913
#41281  0.047846   0.074102
#7028   0.207277   0.247912
#36206  0.032816   0.024276
#2540  0.119479   0.170179

#[10785 rows x 2 columns]



