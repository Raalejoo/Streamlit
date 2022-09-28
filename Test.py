import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.title('Clasificadores')

st.write('''
## Explora diferentes clasificadores
### Cual es el mejor?
''')

dataset_name = st.sidebar.selectbox('Selecciona Data: ', ('Iris', 'Breast Cancer', 'Wine'))
classifier_name = st.sidebar.selectbox('Selecciona Clasificador: ', ('KNN', 'SVM', 'Ramdon Forest'))

def get_dataset(dataset_name):

    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    
    X = data.data
    y = data.target

    return X, y

X, y = get_dataset(dataset_name)

st.write('Forma de los datos', X.shape)
st.write('Numero de Clases', len(np.unique(y)))


def add_parameter_ui(clf_name):

    params = dict()

    if clf_name == 'KNN':

        K = st.sidebar.slider('K', 1, 15)

        params['K'] = K
    elif clf_name == 'SVM':

        SVM = st.sidebar.slider('C', 0.01, 10.0)

        params['C'] = SVM
    else:

        max_depth = st.sidebar.slider('Max_Depth: ', 2, 15)
        n_estimators = st.sidebar.slider('N Estimador: ', 1, 100)

        params['n_estimators'] = n_estimators
        params['max_depth'] = max_depth

    return params

params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):

    if clf_name == 'KNN':

        clf = KNeighborsClassifier(n_neighbors=params['K'])

    elif clf_name == 'SVM':

        clf = SVC(C=params['C'])
    else:

        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                     random_state=1234)

    return clf


clf = get_classifier(classifier_name, params)

# Clasificacion

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


acc = accuracy_score(y_test, y_pred)
st.write(f'Clasificador = {classifier_name}')
st.write(f'Presicion = {acc}')


# Graficas

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Componente 1')
plt.ylabel('Principal Componente 2')
plt.colorbar()

# Imprime en Dashboard
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
