# mlWithSci-KitLearn
 Iris morphometrics ---simple classification supervised learning with Sci-Kit learn
from sklearn.datasets import load_iris

iris=load_iris()

iris.keys()
Out[3]: dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

n_samples,n_features=iris.data.shape

n_samples,n_features
Out[5]: (150, 4)

iris.data[0]
Out[6]: array([5.1, 3.5, 1.4, 0.2])

iris.feature_names
Out[7]: 
['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']

iris.target
Out[8]: 
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

iris.target_names
Out[9]: array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

import pandas as pd

iris_df=pd.DataFrame(iris.data,columns=iris.feature_names).assign(species=iris.target_names[iris.target])

import seaborn as sns

sns.pairplot(iris_df,hue='species',size=1.5);
from sklearn.decomposition import PCA

pca=PCA(n_components=2,whiten=True).fit(iris.data)

X_pca=pca.transform(iris.data)

pca.explained_variance_ratio_
Out[17]: array([0.92461621, 0.05301557])

iris_df['First Component']=X_pca[:,0]

iris_df['Second Component']=X_pca[:,1]

sns.lmplot('First Component','Second Component',data=iris_df,fit_reg=False,hue="species")
Out[20]: <seaborn.axisgrid.FacetGrid at 0x2587cd02f98>
