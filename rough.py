import pandas as pd

df=pd.read_csv('hello.csv')

print(df)


from sklearn import linear_model

regr=linear_model.LinearRegression()

X=df[['TV','radio','newspaper']]
Y=df['sales']
regr=regr.fit(X,Y)
predictions=regr.predict(X)


df['predictions']=predictions

print(df)


import matplotlib.pyplot as mp

mp.plot(Y,'*r')


pm.plot(predictions,'+b')

from sklearn.metrics import r2_score

a_S=r2_score(list(Y),predictions)
print(a_S)






tv=float(input())
radio=float(input())
newspaper=float(input())

pred=regr.predict([[tv,radio,newspaper]])

print(pred[0])



import Tkinter

top=Tkinter.tk()
top.mainloop()
