import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt


#Exogenous parameteres
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
in_features=23
lr=1e-03
EPOCHS=1000000

#Prepare data
data=pd.read_csv('pre_data')
X_data=data[data.columns[:-1]]
X=torch.tensor(X_data.to_numpy()).to(torch.float)
print(X.shape)

Y_data=data[data.columns[-1]]
Y=torch.tensor(Y_data.to_numpy()).unsqueeze(dim=1).to(torch.float)
print(Y.shape)

#Creat training/test sets
train_split=int(0.8*len(data))
X_train,y_train=X[:train_split],Y[:train_split]
X_test,y_test=X[train_split:],Y[train_split:]


X_train,y_train=X_train.to(device),y_train.to(device)
X_test,y_test=X_test.to(device),y_test.to(device)

print(X_test.shape, y_test.shape, X_train.shape, y_train.shape)

#Construct Model

class PredictClaims(nn.Module):
    
    def __init__(self, in_features: int, hidden_units:int):
        super().__init__()
        
        self.linear1=nn.Linear(in_features=in_features, out_features=hidden_units)
        self.linear2=nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.linear3=nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.linear4=nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.linear5=nn.Linear(in_features=hidden_units, out_features=1)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x=self.linear1(x)
        x=F.relu(x)
        x=self.linear2(x)
        x=F.relu(x)
        x=self.linear3(x)
        x=F.relu(x)
        x=self.linear4(x)
        x=F.relu(x)
        x=self.linear5(x)
        return x

#Loss function
loss=nn.MSELoss()

#Initializing
pre_comp_model=PredictClaims(in_features=in_features, hidden_units=in_features*4).to(device)
optimizer=torch.optim.Adam(params=pre_comp_model.parameters(), lr=lr)

#Lists for tracking results
train_loss_all=[]
test_loss_all=[]

epoch_n=[]

# print the number of parameters in the model
print(sum(p.numel() for p in pre_comp_model.parameters()), 'parameters')

pre_comp_model.train()

#Train Loop
for epoch in range(EPOCHS):
        logits=pre_comp_model(X_train)
        loss_m=loss(logits, y_train)
            
        optimizer.zero_grad(set_to_none=True)
        loss_m.backward()
        optimizer.step()
        
        if epoch% 1000==0:
            print (f"train losses: {loss_m}")
            train_loss_all.append(loss_m.item())
            epoch_n.append(epoch)
            
        #Evaluation 
        with torch.no_grad():
            test_logits=pre_comp_model(X_test)
            test_loss=loss(test_logits,y_test)
            
            
            if epoch % 1000==0:
                print (f"test losses: {test_loss}")
                test_loss_all.append(test_loss.item())
                

#Plot the result
with torch.no_grad():
    y_pred=pre_comp_model(X.to(device))

plt.plot(y_pred.cpu().numpy(), label='Model')
plt.plot(Y.cpu().numpy(), label='Real Data')
plt.legend()
plt.show()

plt.plot(epoch_n, train_loss_all, label='train loss')
plt.plot(epoch_n,test_loss_all, label='test loss')
plt.legend()
plt.show()
