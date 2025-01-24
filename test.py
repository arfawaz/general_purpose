#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:22:33 2025

@author: fawaz
"""

import sys
import os
#imports
import torch.nn as nn
import torch
from torchvision.models import ResNet50_Weights
from torchvision import models
# 1) imports
import models
import torch.nn as nn
import torch.optim as optim
from utilities import load_data_from_csv, train_test_loader, train, test, mean_norm
from models import SimpleCNN, CustomResNet50


#%%

# Add the folder to the system path
folder_path = '/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/general_utilities'
sys.path.append(folder_path)
folder_path = '/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/plutos_replay_attack'
sys.path.append(folder_path)

from general_utilities import cnn_builder
from utilities import load_data_from_csv, train_test_loader, train, test, mean_norm

#%%


import csv
import torch

# Function to parse each row and process it into a 64x2 tensor for CSI data
def parse_csi_data(csi_row):
    # Split the CSI row by spaces and check if we have exactly 128 values (64 subcarriers, each with magnitude and angle)
    csi_values = csi_row.split()
    if len(csi_values) != 128:
        return None
    
    # Create a 64x2 tensor for amplitude and angle pairs
    csi_tensor = []
    for i in range(0, 128, 2):  # Step through the csi_values with a step size of 2
        try:
            magnitude = float(csi_values[i])
            angle = float(csi_values[i + 1])
            csi_tensor.append([magnitude, angle])
        except ValueError:
            return None  # Skip rows with invalid data (non-numeric values)
    
    # Return the 64x2 tensor
    return torch.tensor(csi_tensor)

# Function to process the entire CSV file
def process_csv(file_path):
    data = []   # List to store 64x2 tensors
    labels = [] # List to store integer labels associated with each MAC address
    
    mac_id_to_label = {}  # Dictionary to map MAC addresses to integer labels
    label_counter = 0  # Counter to generate unique integer labels for each MAC address
    
    # Open the CSV file and read it row by row
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        mac_id = None
        for row in reader:
            if len(row) != 2:
                continue  # Skip rows that don't have exactly 2 columns
            
            current_mac_id, csi_row = row
            # Check if the CSI row has valid values
            csi_tensor = parse_csi_data(csi_row)
            if csi_tensor is not None:
                # Assign a new label to the MAC address if it's encountered for the first time
                if current_mac_id not in mac_id_to_label:
                    mac_id_to_label[current_mac_id] = label_counter
                    label_counter += 1
                
                # Get the integer label for the current MAC address
                label = mac_id_to_label[current_mac_id]
                
                data.append(csi_tensor)  # Append the 64x2 tensor
                labels.append(label)      # Append the corresponding integer label
    data_ = torch.stack(data)
    labels_ = torch.tensor(labels, dtype=torch.long)
    return data_, labels_

# Example usage of the function
file_path = '/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/esp32/esp32_csi_tool/ESP32-CSI-Tool/active_ap/custom_data_collection/all_macid.csv'  # Provide the actual path to your CSV file
data, labels = process_csv(file_path)

# For demonstration, print the first 3 samples of data and their corresponding labels
for i in range(min(5, len(data))):
    print(f"Sample {i+1} - Data:\n{data[i]}")
    print(f"Label: {labels[i]}\n")



#%% firrst 3000 enttries only

import csv
import torch

# Function to parse each row and process it into a 64x2 tensor for CSI data
def parse_csi_data(csi_row):
    # Split the CSI row by spaces and check if we have exactly 128 values (64 subcarriers, each with magnitude and angle)
    csi_values = csi_row.split()
    if len(csi_values) != 128:
        return None
    
    # Create a 64x2 tensor for amplitude and angle pairs
    csi_tensor = []
    for i in range(0, 128, 2):  # Step through the csi_values with a step size of 2
        try:
            magnitude = float(csi_values[i])
            angle = float(csi_values[i + 1])
            csi_tensor.append([magnitude, angle])
        except ValueError:
            return None  # Skip rows with invalid data (non-numeric values)
    
    # Return the 64x2 tensor
    return torch.tensor(csi_tensor)

# Function to process the entire CSV file
def process_csv(file_path, max_samples_per_mac=9000):
    data = []   # List to store 64x2 tensors
    labels = [] # List to store integer labels associated with each MAC address
    
    mac_id_to_label = {}  # Dictionary to map MAC addresses to integer labels
    mac_id_sample_count = {}  # Dictionary to keep track of the number of samples for each MAC address
    label_counter = 0  # Counter to generate unique integer labels for each MAC address
    
    # Open the CSV file and read it row by row
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        mac_id = None
        for row in reader:
            if len(row) != 2:
                continue  # Skip rows that don't have exactly 2 columns
            
            current_mac_id, csi_row = row
            # Check if the CSI row has valid values
            csi_tensor = parse_csi_data(csi_row)
            if csi_tensor is not None:
                # Assign a new label to the MAC address if it's encountered for the first time
                if current_mac_id not in mac_id_to_label:
                    mac_id_to_label[current_mac_id] = label_counter
                    label_counter += 1
                
                # Get the integer label for the current MAC address
                label = mac_id_to_label[current_mac_id]
                
                # Track the number of samples for each MAC ID
                if current_mac_id not in mac_id_sample_count:
                    mac_id_sample_count[current_mac_id] = 0
                
                if mac_id_sample_count[current_mac_id] < max_samples_per_mac:
                    data.append(csi_tensor)  # Append the 64x2 tensor
                    labels.append(label)      # Append the corresponding integer label
                    mac_id_sample_count[current_mac_id] += 1  # Increment the sample count for this MAC ID
                
                # Stop adding samples for this MAC ID if 3000 samples are reached
                if mac_id_sample_count[current_mac_id] >= max_samples_per_mac:
                    continue

    data_ = torch.stack(data)
    labels_ = torch.tensor(labels, dtype=torch.long)
    return data_, labels_

# Example usage of the function
file_path = '/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/esp32/esp32_csi_tool/ESP32-CSI-Tool/active_ap/custom_data_collection/all_macid.csv'  # Provide the actual path to your CSV file
data, labels = process_csv(file_path)

# For demonstration, print the first 3 samples of data and their corresponding labels
for i in range(min(5, len(data))):
    print(f"Sample {i+1} - Data:\n{data[i]}")
    print(f"Label: {labels[i]}\n")





#%%




class SimpleCNN(nn.Module):
    def __init__(self,num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1),padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1),padding=(1, 0))
        self.fc1 = nn.Linear(4096, 128)  # Adjusted based on the output shape of conv layers
        self.fc2 = nn.Linear(128, num_classes)  # 4 output classes
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))

    def forward(self, x):
        #the shape of the input is : 999by2
        x = torch.relu(self.conv1(x)) #the shape after the first convolution lahyer is : 
        x = torch.relu(self.conv2(x))
        #x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#%%

file_path = '/home/fawaz/Desktop/USF/PHD/COURSES/SPRING25/projects_on_git/esp32/esp32_csi_tool/ESP32-CSI-Tool/active_ap/custom_data_collection/all_macid.csv'  # Provide the actual path to your CSV file
data, labels = process_csv(file_path)
data = data.unsqueeze(1)
dataset = mean_norm(data)
train_loader, test_loader = train_test_loader(dataset, labels)

num_classes = 2
learning_rate = 0.001
num_epochs = 50

model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Model Training
model.train()
train(model = model, train_loader = train_loader, test_loader=test_loader, criterion = criterion, optimizer = optimizer, num_epochs=num_epochs)

# Model Testing
model.eval()
_ = test(model, test_loader)
#%%
cnn_builder()