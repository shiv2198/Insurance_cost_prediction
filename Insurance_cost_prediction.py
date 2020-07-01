import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def data_cleaning(x,y):
    
    labelencoder_x = LabelEncoder()
    x[:,1] = labelencoder_x.fit_transform(x[:,1])   # Femaele : 0 , Male : 1
    print(x)
    x[:,4] = labelencoder_x.fit_transform(x[:,4])   # No : 0 , Yes : 1
    x[:,5] = labelencoder_x.fit_transform(x[:,5])   # southwest : 3, southeast:2, northwest : 1, northest:0
    
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    y = np.reshape(y, (-1, 1))
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)
    pd_x = pd.DataFrame(x)

    
    
    
def data_train():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_10 = x_test[:10,:]
    y_10 = y_test[:10]

    # Linear Regression
    model = LinearRegression()
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
#    print(y_pred[:5])
    plotting(x_test,y_test,y_pred)
    

#    
##    print(y_pred[:5])
#    plotting(y_test,y_pred)
    
    #Support Vector Regression
    model = SVR(kernel = 'rbf')

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
#    y_pred = sc_y.inverse_transform(y_pred)
    plotting(x_test,y_test,y_pred)
#    print(y_pred[:5])

    



    model = DecisionTreeRegressor(random_state = 0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    plotting(x_test,y_test,y_pred)


    
def plotting(x_test,y_test,y_pred):
    
    
    y_pred = model.predict(x_10)
    plt.scatter(x_10[:,0], y_10, color = 'red')
    plt.scatter(x_10[:,0], model.predict(x_10), color = 'green')
    plt.title('Truth or Bluff (SVR)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    
#    plt.scatter(x_10[:,0],y_10,color = 'red')
#    plt.scatter(x_10[:,0],svr.predict(x_10),color = 'blue')
#    plt.xlabel('Y Test')
#    plt.ylabel('Predicted Y')

if __name__ == "__main__":
    
    df = pd.read_csv("/home/shivansh/Desktop/projects/Insurance_cost_prediction/insurance.csv")
    df = df.dropna()
    df = df.drop_duplicates()
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    
    
    data_cleaning(x,y)
    data_train()
    
