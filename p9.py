from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = load_iris()
X = pd.DataFrame(dataset.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(dataset.target)
y.columns = ['Targets']
print(X)

plt.figure(figsize=(7, 7))
colormap = np.array(['red', 'lime', 'black'])

scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
XS = pd.DataFrame(xsa, columns=X.columns)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(XS)
y_cluster_gmm = gmm.predict(XS)

plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('GMM Classification')
plt.show()
