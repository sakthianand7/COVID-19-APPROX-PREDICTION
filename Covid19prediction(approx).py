import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
dataset=pd.read_csv('total_cases.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)
poly.fit(X_poly, y)
regressor=LinearRegression()
regressor.fit(X_poly,y)
plt.scatter(X,y,color='yellow')
plt.plot(X,regressor.predict(poly.fit_transform(X)),color='darkblue')
plt.title('COVID-19 CASES PREDICTION')
plt.xlabel('No of Days(From 31-12-2019 - 17-04-2020)')
plt.ylabel('No of Cases')
plt.show()
