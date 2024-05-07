import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

land_data = pd.read_csv('data/Land.csv')
species_data = pd.read_csv('data/Threatened Species.csv')
water_data = pd.read_csv('data/Water and Sanitation Services.csv')

land_data.rename(columns={'Value': 'Land Value'}, inplace=True)
species_data.rename(columns={'Value': 'Species Value'}, inplace=True)
water_data.rename(columns={'Value': 'Water Value'}, inplace=True)

land_yearly = land_data.groupby('Year')['Land Value'].mean().reset_index()
species_yearly = species_data.groupby('Year')['Species Value'].mean().reset_index()
water_yearly = water_data.groupby('Year')['Water Value'].mean().reset_index()

merged_data = pd.merge(land_yearly, species_yearly, on='Year', how='inner')
merged_data = pd.merge(merged_data, water_yearly, on='Year', how='inner')

features = merged_data[['Land Value', 'Species Value']]
target = merged_data['Water Value']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1).astype(np.float32))
X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1).astype(np.float32))

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(2, 50)
        self.layer2 = nn.Linear(50, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        return self.layer2(x)


model = RegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor)
    test_loss = criterion(predicted, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')
