import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.nn import PyroModule, PyroSample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from Toolkit import *

importer = DataImporter()
df_test, df_train, df_submission = importer.load_data()

X = df_train.drop(columns=["Risk"]).copy()
y = df_train["Risk"].copy()

encoder = Encode(X)
labeler = Label(y)
X_encoded = encoder.dummying()
y_labeled = labeler.labeling()

for column in X_encoded.columns:
    if X_encoded[column].dtype == 'object':
        X_encoded[column] = pd.to_numeric(X_encoded[column], errors='coerce')

X_encoded_values = X_encoded.to_numpy(dtype='float64')

print(X_encoded_values.ctypes)

splitter = Split(X_encoded_values, y_labeled)
X_train, X_test, y_train, y_test = splitter.splitting(test_size=0.05, random_state=42)

# Convertir les données en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

import torch
from torch.utils.data import DataLoader, TensorDataset

# Créer un DataLoader pour les données d'entraînement
batch_size = 32  # Taille du batch
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Créer un DataLoader pour les données de test
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Exemple d'itération sur les batches
for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
    # batch_X et batch_y contiennent les données pour un batch
    print(f"Batch {batch_idx + 1}:")
    print(f"X batch shape: {batch_X.shape}")
    print(f"y batch shape: {batch_y.shape}")

# Vous pouvez maintenant utiliser train_loader et test_loader pour entraîner et évaluer votre modèle

class BNN(PyroModule):
    def __init__(self):
        super().__init__()
        self.linear1 = PyroModule[nn.Linear](20, 50)
        self.linear2 = PyroModule[nn.Linear](50, 1)

        # Priors pour les poids et les biais
        self.linear1.weight = PyroSample(dist.Normal(0., 1.).expand([50, 20]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., 1.).expand([50]).to_event(1))
        self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([1, 50]).to_event(2))
        self.linear2.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x



# Calculer la précision
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")