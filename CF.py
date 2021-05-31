
# ==========================================================================
# An example for using collaborative filtering in MovieLen100k data
# ==========================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

def CF_cv(df,k=20,cv=10):
    '''Use collaborative filtering with cross-validation 
    
    Args:
        df: numpa.array 
            a user-item or item-user matrix
        k: int
            number of neighbors for KNN algorithms
        cv: int
            number of folds
    Returns:
        a list of RMSE for all folds
    
    '''
    kf = KFold(n_splits=10, shuffle=True,random_state=1)
    k=20
    result=[]
    nonzero=np.where(df!=0)
    for train,test in kf.split(nonzero[0]):
        X=np.copy(df)
        X[(nonzero[0][test],nonzero[1][test])]=0
        sim=cosine_similarity(X,X)
        # mean rating for each item/user, used for guess when knn has no available ratings for a item/user 
        fill_ratings=np.ma.array(X,mask=(X==0)).mean(axis=0)
        fill_ratings=fill_ratings.filled(fill_value=0)
        # item/user with no more than 3 ratings, used for guess when the training set has no available ratings for a item/user 
        few=np.count_nonzero(X,axis=0)
        few=X[:,np.where((few<=3)&(few>0))[0]]
        few_rating=np.ma.array(few,mask=(few==0)).mean()
        seloss=0
        for i in range(X.shape[0]): # item/user
            topred=nonzero[1][test][nonzero[0][test]==i]
            if len(topred)==0:
                continue
            weight=np.sort(sim[i,:])[-(k+1):-1:1]
            neighbors=X[np.argsort(sim[i,:])[-(k+1):-1:1],:][:,topred]
            ratings=np.ma.average(np.ma.array(neighbors,mask=(neighbors==0)),axis=0,weights=weight)
            ratings=ratings.filled(fill_value=0)
            ratings[ratings==0]=fill_ratings[topred][ratings==0]
            ratings[ratings==0]=few_rating
            seloss+=((ratings-df[(np.repeat(i,len(topred)),topred)])**2).sum()
        result.append(np.sqrt(seloss/test.shape[0]))
    return(result)


if __name__=="__main__":
    
    # data-preprocessing
    df=pd.read_csv(Path('...\ml-100k')/'u.data',sep='\t',header=None)
    df.columns=['userId','movieId','rating','timestamp']
    df=df.pivot(index='userId',columns='movieId',values='rating').reset_index() # user-item matrix
    np.unique(df.iloc[:,1:].isnull().sum(),return_counts=True) # number of rates by movies
    np.unique(df.iloc[:,1:].isnull().sum(axis=1),return_counts=True) # number of rates by users
    df=np.array(df.iloc[:,1:])
    df[np.isnan(df)]=0 # fill nan values
    
    # cross-validation
    result=CF_cv(df,k=20,cv=10) # user-based CF
    result=CF_cv(df.T,k=20,cv=10) # item-based CF

# mean RMSE for user-based CF = 1.061
# mean RMSE for item-based CF = 0.983


