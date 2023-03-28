import numpy as np
import pandas as pd
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import torch.optim as optim

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
colonne = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
dataSet = pd.read_csv(url, header=None, names=colonne)

#correlazione tra due attributi e preparazione dei dati da visualizzare
correlazione = np.corrcoef(dataSet['age'], dataSet["workclass"])
print('La correlazione tra la colonna age e la colonna workclass e:', correlazione[0][1])
x = dataSet.iloc[:, :-1].values
y = pd.factorize(dataSet['race'])[0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1, random_state = 0)

#MLP con Sklearn 1.0
modello_sklearn = MLPClassifier(hidden_layer_sizes =(10,), max_iter = 1000, solver='sgd', random_state = 0)
modello_sklearn.fit(x_train, y_train)

#DA GUARDARE SE MODIFICARE I NOMI DELLE VARIABILI
#MLP con Pytorch 1.0
size_input = x_train.shape[1]
size_hidden = 10
size_output = len(np.unique(y_train))
num_epochs = 1000
size_batch = 10

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(size_input, size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_output)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
mlp_pt = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp_pt.parameters(), lr = 0.1)
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mlp_pt(inputs.float())
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

#MLP con Sklearn 2.0
y_pred_sklearn = modello_sklearn.predict(x_test)
print('MLP con Sklearn')
print(classification_report(y_test, y_pred_sklearn))
print(confusion_matrix(y_test, y_pred_sklearn))

#MLP con Pytorch
mlp_pt.eval()
with torch.no_grad():
    inputs = torch.from_numpy(x_test)
    outputs = mlp_pt(inputs.float())
    _, y_pred_pt = torch.max(outputs, 1)

print('MLP con PyTorch')
print(classification_report(y_test, y_pred_pt))
print(confusion_matrix(y_test, y_pred_pt))
