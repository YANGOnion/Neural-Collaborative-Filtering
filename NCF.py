
# ==========================================================================
# An example for using Neural Collaborative Filtering model in MovieLen100k data
# ==========================================================================

import torch
from torch import nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path

class NNCF(nn.Module):
    '''pytorch network class for Neural Collaborative Filtering model

    Attributes:
        num_user: int 
            number of users
        num_item: int
            number of items
        dim_embed: int
            dimension of embedding layer
    '''
    def __init__(self,num_user,num_item,dim_embed):
        super(NNCF,self).__init__()
        self.num_user=num_user
        self.num_item=num_item
        self.dim_embed=dim_embed
        self.mlp_user=nn.Embedding(self.num_user,self.dim_embed)
        self.mlp_item=nn.Embedding(self.num_item,self.dim_embed)
        self.mlp=nn.Sequential(
                nn.Linear(2*self.dim_embed,64),
                nn.ReLU(),
                nn.Linear(64,16),
                nn.ReLU(),
                nn.Linear(16,8),
                nn.ReLU()
                )
        self.gmf_user=nn.Embedding(self.num_user,self.dim_embed)
        self.gmf_item=nn.Embedding(self.num_item,self.dim_embed)
        self.last=nn.Linear(self.dim_embed+8,1)
    
    def forward(self,x):
        out_mlp=torch.cat([self.mlp_user(x[:,0]),self.mlp_item(x[:,1])],dim=1)
        out_mlp=self.mlp(out_mlp)
        out_gmf=torch.mul(self.gmf_user(x[:,0]),self.gmf_item(x[:,1]))
        out=self.last(torch.cat([out_gmf,out_mlp],dim=1))
        return out
        
    
def training(train_loader,model,device,loss_fn,optimizer,epochs=5):
    '''Train the Neural Collaborative Filtering model'''
    for i in range(epochs):
        epoch_loss=[]
        for x, y in train_loader:
            x,y=x.to(device),y.to(device)
            pred = model(x)
            loss = loss_fn(pred, torch.tensor(y,dtype=torch.float32).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        print(np.sqrt(np.mean(epoch_loss))) # trace the training loss


if __name__=="__main__":
    
    # data preprocessing: to create user & item dict
    df=pd.read_csv(Path('...\ml-100k')/'u.data',sep='\t',header=None)
    df.columns=['userId','movieId','rating','timestamp']
    d_user={user:i for i,user in enumerate(sorted(df.userId.unique()))}
    d_item={item:i for i,item in enumerate(sorted(df.movieId.unique()))}
    
    # cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    cv_loss=[]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for train, test in kf.split(df):
        train_dataset = Data.TensorDataset(torch.tensor([[d_user[df.loc[i,'userId']],
                                                          d_item[df.loc[i,'movieId']]] for i in train]),
                                       torch.tensor(np.array(df.loc[train,'rating'])[:,np.newaxis]))
        train_loader=Data.DataLoader(
            dataset=train_dataset, 
            batch_size=256, 
            shuffle=True)
        model=NNCF(len(d_user),len(d_item),32)
        model.to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-3)
        training(train_loader,model,device,loss_fn,optimizer,10)
        x_test=torch.tensor([[d_user[df.loc[i,'userId']],
                              d_item[df.loc[i,'movieId']]] for i in test]).to(device)
        y_test=torch.tensor(np.array(df.loc[test,'rating'])[:,np.newaxis]).to(device)
        with torch.no_grad():
            pred = model(x_test)
            loss=loss_fn(pred, torch.tensor(y_test,dtype=torch.float32).to(device))
            cv_loss.append(loss.item())

# mean RMSE = 0.943


