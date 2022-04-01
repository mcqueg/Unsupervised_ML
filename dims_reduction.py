

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def PCA(data, n):
    '''
    Arguments:
        X -- df to be projected into a lower dimensional space.. shape (num examples, num features)
        n -- number of desired principal components or the number of dimensions to reduce the dataset to
            n < features 
    Returns:
        X_pca -- the new reduced data from of shape (num examples, n)
    '''
    X = data.copy()
    # step 1: normalize data
    X = MinMaxScaler().fit_transform(X)
    #X_mean = X - np.mean(X, axis=0)

    # step 2: calcualte covariance
    cov_mat = np.cov(X, rowvar=False) # False b/c observations are row-wise & features are column-wise

    # step 3: compute eigenvals and eigenvectors from covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

    # step 4: sort eigenvals index w/ argsort() and flip() to get the largest egienvals on top
    # apply the sorted indices to the eigenvectors in order to sort them in descending order
    eigenvecs_sorted = eigen_vecs[:, np.flip(np.argsort(eigen_vals))]

    # step 5: create the eigenvector subset by slicing from the first element to the specified nth element
    eigenvec_subset = eigenvecs_sorted[:,0:n]

    #step 6: project the dataset by using the dot product and eigenvector subset
    X_pca = np.dot(X, eigenvec_subset)


    return X_pca


def LDA(data, labels, n):
    '''
    Arguments:
        data -- df to be projected into a lower dimensional space.. shape (num examples, num features)
        labels -- the labels for each observation in data
        n -- number of desired principal components or the number of dimensions to reduce the dataset to
            n < features 
    Returns:
        X_lda -- the new reduced data from of shape (num examples, n)
    '''
    X = data.copy()
    X = MinMaxScaler().fit_transform(X)
    # compute mean of data
    X_mean = X.mean()
    X = pd.DataFrame(X)
    # initlize empty lists to hold the scatters for each class
    Sb = []
    Sw = []

    # loop through each class and compute the scatters
    for c in labels.unique():
        # compute between class scatter for each class
        c_mean = X.loc[labels==c].mean()
        c_diff = c_mean - X_mean
        c_Sb = c_diff.to_frame() @ c_diff.to_frame().T
        Sb.append(c_Sb)
        # compute within class scatter for each class
        c_vecs = X.loc[labels==c] # seperate data into classes
        c_Sw = c_vecs.cov() #class scatter
        Sw.append(c_Sw)
    # find the summation of both Sb and Sw to compute the between class and within class scatters
    b_scatter = sum(Sb)
    w_scatter = sum(Sw)
    
    #Calculate Sw2 and eigenvectors/eigenvalues
    Sw2 = np.linalg.inv(w_scatter)
    eigenvals, eigenvecs = np.linalg.eigh(Sw2 * b_scatter)
    # get the subset of eigenvecs
    transformation_matrix = eigenvecs[:n]
    column_names = ['LD' + str(num + 1) for num in range(n)]
    #transform original data by applying the eigenvector subset
    X_lda = pd.DataFrame((transformation_matrix.dot(X.T).T), columns=column_names)
    
    return X_lda