from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from sklearn.cluster import KMeans


class Cluster:

    def __init__(self):
        self.df = pd.read_csv('SomaDataset.csv')
        print(self.df.columns)
        self.df = self.df[['Weight', 'Height', 'Gender', 'BodyFat', 'BMI']]
        dfman = DataFrame(self.df.query("Gender == 1"))
        dfwoman = DataFrame(self.df.query("Gender == 0"))

        self.kMeans(self.df[['BMI']], 'General')
        # self.kMeans(dfman, 'Man Cluster')
        # self.kMeans(dfwoman, 'Woman Cluster')

    @staticmethod
    def optimizer_k_means(data, max_k, title):
        means = []
        inertias = []

        for k in range(1, max_k):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            means.append(k)
            inertias.append(kmeans.inertia_)

        plt.subplots(figsize=(10, 5))
        plt.plot(means, inertias, 'o-')
        plt.title(title)
        plt.xlabel = 'Number of clusters'
        plt.ylabel = 'Iterazioni'
        plt.grid(True)
        plt.show()

    def kMeans(self, dfKmeans, title):
        self.optimizer_k_means(dfKmeans, 30, title)
        kmeans = KMeans(n_clusters=6)
        kmeans.fit(dfKmeans)
        dfKmeans['kmeans_9'] = kmeans.labels_

        clusters = DataFrame(kmeans.cluster_centers_)
        array = list(dfKmeans.columns.values.copy())
        array.pop()
        clusters.columns = array
        plt.subplots(figsize=(10, 7))
        plt.scatter(x=self.df['Height'], y=self.df['Weight'], c=dfKmeans['kmeans_9'])
        plt.title('General analisys 2D')
        plt.xlabel = 'Height'
        plt.ylabel = 'BMI'
        plt.xlim(145, 200)
        plt.ylim(40, 140)
        plt.show()

        plt.subplots(figsize=(10, 7))
        ax = plt.axes(projection='3d')
        ax.scatter3D(xs=self.df['Height'], ys=self.df['Weight'], zs=self.df['BMI'], c=dfKmeans['kmeans_9'])
        plt.xlim(145, 200)
        plt.ylim(40, 117)
        plt.title('General analisys 3D')
        plt.xlabel = 'Height'
        plt.ylabel = 'Weight'
        plt.show()
        # 3d scatterplot using plotly
        Scene = dict(xaxis=dict(title='Height -->'),
                     yaxis=dict(title='Weight--->'),
                     zaxis=dict(title='BMI-->'))

        # model.labels_ is nothing but the predicted clusters i.e y_clusters

        trace = go.Scatter3d(x=self.df['Height'], y=self.df['Weight'], z=self.df['BodyFat'], mode='markers',
                             marker=dict(color=dfKmeans['kmeans_9'], size=15, line=dict(color='black', width=10)))

        layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=800, width=800)
        data = [trace]
        fig = go.Figure(data=data, layout=layout)
        fig.show()


Cluster()
