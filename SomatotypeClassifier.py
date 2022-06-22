import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


class SomatotypeClassifier:
    def __init__(self):
        df = pd.read_csv('SomaDataset.csv')
        df = DataFrame(df.drop(columns=['Unnamed: 0']))
        df.dropna()
        accKNN = []
        accDT = []
        accRF = []
        accLR = []
        labelencoder = LabelEncoder()
        df['Gender'] = labelencoder.fit_transform(df['Gender'])

        Y = df['Endomorfism']
        X = df[['Weight', 'Height', 'SommaPliche']]

        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.10)
        accKNN.append(self.kNearModel(XTrain, XTest, YTrain, YTest, 'Endomorfism'))
        accDT.append(self.decisionTreeModel(XTrain, XTest, YTrain, YTest, 'Endomorfism'))
        accRF.append(self.randomForestModel(XTrain, XTest, YTrain, YTest, 'Endomorfism'))
        accLR.append(self.logiticRegressionModel(XTrain, XTest, YTrain, YTest, 'Endomorfism'))

        # plot Compare Accuracy Endomorfism
        plt.plot(range(20), accKNN[0][0], 'o-', label='KNN')
        plt.plot(range(20), accDT[0], 'o-', label='Decision Tree')
        plt.plot(range(20), accRF[0], 'o-', label='Random Forest')
        plt.plot(range(20), accLR[0], 'o-', label='Logistic Regresor')
        plt.legend()
        plt.title('\nEndomorfism Accuracy Compared')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

        Y = df['Mesomorfism']
        X = df[['Height', 'SommaPliche', 'CalfCircum', 'BicepCircum', 'KneeDiam', 'ElbowDiam']]

        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.10)
        accKNN.append(self.kNearModel(XTrain, XTest, YTrain, YTest, 'Mesomorfism'))
        accDT.append(self.decisionTreeModel(XTrain, XTest, YTrain, YTest, 'Mesomorfism'))
        accRF.append(self.randomForestModel(XTrain, XTest, YTrain, YTest, 'Mesomorfism'))
        accLR.append(self.logiticRegressionModel(XTrain, XTest, YTrain, YTest, 'Mesomorfism'))

        # plot Compare Accuracy Mesomorfism
        plt.plot(range(20), accKNN[1][0], 'o-', label='KNN')
        plt.plot(range(20), accDT[1], 'o-', label='Decision Tree')
        plt.plot(range(20), accRF[1], 'o-', label='Random Forest')
        plt.plot(range(20), accLR[1], 'o-', label='Logistic Regresor')
        plt.legend()
        plt.title('\nMesoformism Accuracy Compared')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

        Y = df['Ectomorfism']
        X = df[['Weight', 'Height']]

        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.10)

        accKNN.append(self.kNearModel(XTrain, XTest, YTrain, YTest, 'Ectomorfism'))
        accDT.append(self.decisionTreeModel(XTrain, XTest, YTrain, YTest, 'Ectomorfism'))
        accRF.append(self.randomForestModel(XTrain, XTest, YTrain, YTest, 'Ectomorfism'))
        accLR.append(self.logiticRegressionModel(XTrain, XTest, YTrain, YTest, 'Ectomorfism'))

        # plot Compare Accuracy Ectomorfism
        plt.plot(range(20), accKNN[2][0], 'o-', label='KNN')
        plt.plot(range(20), accDT[2], 'o-', label='Decision Tree')
        plt.plot(range(20), accRF[2], 'o-', label='Random Forest')
        plt.plot(range(20), accLR[2], 'o-', label='Logistic Regresor')
        plt.legend()
        plt.title('\nEctomorfims Accuracy Compared')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

        '''# plot Accuracy KNN
        plt.plot(range(20), accKNN[0][0], 'o-', label=f'Endomorfism con K. {accKNN[0][1]}')
        plt.plot(range(20), accKNN[1][0], 'o-', label=f'Mesomorfism con K. {accKNN[1][1]}')
        plt.plot(range(20), accKNN[2][0], 'o-', label=f'Ectomorfism con K. {accKNN[2][1]}')
        plt.legend()
        plt.title('\n KNN Accuracy')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()'''

        '''# plot Accuracy Decision Tree
        plt.plot(range(20), accDT[0], 'o-', label='Endomorfism')
        plt.plot(range(20), accDT[1], 'o-', label='Mesomorfism')
        plt.plot(range(20), accDT[2], 'o-', label='Ectomorfism')
        plt.legend()
        plt.title('\nDecision Tree Accuracy')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()'''

        ''' # plot Accuracy Random Forest
        plt.plot(range(20), accRF[0], 'o-', label='Endomorfism')
        plt.plot(range(20), accRF[1], 'o-', label='Mesomorfism')
        plt.plot(range(20), accRF[2], 'o-', label='Ectomorfism')
        plt.legend()
        plt.title('\nRandom Forest Accuracy')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()'''

        '''# plot Accuracy Logistic Regressor
        plt.plot(range(20), accLR[0], 'o-', label='Endomorfism')
        plt.plot(range(20), accLR[1], 'o-', label='Mesomorfism')
        plt.plot(range(20), accLR[2], 'o-', label='Ectomorfism')
        plt.legend()
        plt.title('\nLogic Regressor Accuracy')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()'''

    @staticmethod
    def kNearModel(XTrain, XTest, YTrain, YTest, title):
        global YPred
        errors = []
        greatestError = 100
        greatestN = 0

        for i in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(XTrain, YTrain)
            YPred = knn.predict(XTest)
            error = np.mean(YPred != YTest)
            errors.append(error)

            if greatestError > error:
                greatestError = error
                greatestN = i

        '''plt.figure(figsize=(12, 8))
        plt.plot(range(1, 40), errors, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title(title + '\nKNN Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()'''

        accArray = []
        for i in range(20):
            knn = KNeighborsClassifier(n_neighbors=greatestN)
            knn.fit(XTrain, YTrain)
            YPred = knn.predict(XTest)
            accArray.append(accuracy_score(YTest, YPred))

        '''plt.subplots(figsize=(10, 8))
        plt.plot(range(20), accArray, 'o-')
        plt.title(title + f'\n KNN Accuracy con K={greatestN}')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()'''

        '''plt.figure(figsize=(10, 7))
        plt.title(title + f'\n KNN Confusion Matrix con K={greatestN}')
        sns.heatmap(confusion_matrix(YTest, YPred), annot=True, cmap='Blues')
        plt.show()'''
        return [accArray, greatestN]

    @staticmethod
    def decisionTreeModel(XTrain, XTest, YTrain, YTest, title):
        accArray = []
        for i in range(20):
            tree = DecisionTreeClassifier()
            tree.fit(XTrain, YTrain)

            Ypred = tree.predict(XTest)
            accArray.append(accuracy_score(YTest, Ypred))

        '''plt.subplots(figsize=(10, 8))
        plt.plot(range(20), accArray, 'o-')
        plt.title(title + f'\nDecision Tree')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()'''

        '''plt.figure(figsize=(10, 7))
        plt.title(title + f'\n Decision Tree Confusion Matrix')
        sns.heatmap(confusion_matrix(YTest, Ypred), annot=True, cmap='Blues')
        plt.show()'''

        return accArray

    @staticmethod
    def randomForestModel(XTrain, XTest, YTrain, YTest, title):
        accArray = []
        for i in range(20):
            forest = RandomForestRegressor(n_estimators=20, random_state=0)
            forest.fit(XTrain, YTrain)

            Ypred = forest.predict(XTest)
            Ypred = [np.around(x) for x in Ypred]
            accArray.append(accuracy_score(YTest, Ypred))

        '''plt.subplots(figsize=(10, 8))
        plt.plot(range(20), accArray, 'o-')
        plt.title(title + f'\nDecision Tree')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()'''

        '''plt.figure(figsize=(10, 7))
        plt.title(title + f'\n Random Forest Confusion Matrix')
        sns.heatmap(confusion_matrix(YTest, Ypred), annot=True, cmap='Blues')
        plt.show()'''

        return accArray

    @staticmethod
    def logiticRegressionModel(XTrain, XTest, YTrain, YTest, title):
        accArray = []
        for i in range(20):
            logistic = LogisticRegression(random_state=0, solver='newton-cg', max_iter=1000)
            logistic.fit(XTrain, YTrain)

            Ypred = logistic.predict(XTest)
            Ypred = [np.around(x) for x in Ypred]
            accArray.append(accuracy_score(YTest, Ypred))

        '''plt.subplots(figsize=(10, 8))
        plt.plot(range(20), accArray, 'o-')
        plt.title(title + f'\nDecision Tree')
        plt.xlabel('Number of iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()'''

        '''plt.figure(figsize=(10, 7))
        plt.title(title + f'\n Logistic Regressor Confusion Matrix')
        sns.heatmap(confusion_matrix(YTest, Ypred), annot=True, cmap='Blues')
        plt.show()'''
        return accArray


SomatotypeClassifier()
