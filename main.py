#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:40:19 2025

@author: fawaz
"""
from csi_dataset_creator import process_csv
from mean_norm import mean_norm
from train_test import train, test
from train_test_loader import train_test_loader
from models import SimpleCNN
import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



#%%
file_path = '/home/fawaz/Desktop/usf/directed_research/projects_on_git/general_purpose/all_macid.csv'  # Provide the actual path to your CSV file
data, labels = process_csv(file_path)
data = data.unsqueeze(1)
dataset = mean_norm(data)
train_loader, test_loader = train_test_loader(dataset, labels)

num_classes = 2
learning_rate = 0.001
num_epochs = 50

model = SimpleCNN(num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Model Training
model.train()
train(model = model, train_loader = train_loader, test_loader=test_loader, criterion = criterion, optimizer = optimizer, num_epochs=num_epochs)

# Model Testing
model.eval()
_ = test(model, test_loader)