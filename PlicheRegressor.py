import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class PlicheRegressor:
    def __init__(self):
        self.arrayR2 = []
        self.arraymae = []
        self.arraymse = []
        self.regressor = None
        df = pd.read_csv('athleteData.csv')
        df = df[['BMI', 'Height', 'Weight', 'Gender', 'BodyFat', 'SommaPliche']]
        labelencoder = LabelEncoder()
        df['Gender'] = labelencoder.fit_transform(df['Gender'])
        Y = df['SommaPliche']
        X = df[['BMI', 'Height', 'Weight', 'Gender', 'BodyFat']]

        self.XTrain, self.XTest, self.YTrain, self.YTest = train_test_split(X, Y, test_size=0.01)
        for x in range(20):
            self.Regressor()

        plt.subplots(figsize=(10, 5))
        plt.plot(range(20), self.arrayR2, 'o-')
        plt.title('Linear Regressor: R2')
        plt.xlabel('Number of iteration')
        plt.ylabel('Precision R2')
        plt.grid(True)
        plt.show()

        plt.subplots(figsize=(10, 5))
        plt.plot(range(20), self.arraymae, 'o-')
        plt.title('Linear Regressor: MAE')
        plt.xlabel('Number of iteration')
        plt.ylabel('MAE')
        plt.grid(True)
        plt.show()

        plt.subplots(figsize=(10, 5))
        plt.plot(range(20), self.arraymse, 'o-')
        plt.title('Linear Regressor: MSE')
        plt.xlabel('Number of clusters')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()

    def Regressor(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.XTrain, self.YTrain)
        YPred = self.regressor.predict(self.XTest)

        mse = mean_squared_error(self.YTest, YPred)
        mae = mean_absolute_error(self.YTest, YPred)
        r2 = r2_score(self.YTest, YPred)

        self.arrayR2.append(r2)
        self.arraymse.append(mse)
        self.arraymae.append(mae)

    def predict(self, X):
        return self.regressor.predict(X)


PlicheRegressor()
