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

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate phase from I and Q values
def calculate_phase(i, q):
    return np.arctan2(q, i)  # atan2 is used to get the phase in radians

# Load the data into a pandas DataFrame
file_path = "/home/fawaz/Desktop/usf/directed_research/projects_on_git/esp32/esp32_csi_tool/ESP32-CSI-Tool/active_ap/custom_data_collection/all_macid.csv"  # Replace with your actual file path
data = pd.read_csv(file_path, header=None, names=["mac_id", "csi_data"])

# Parse the csi_data column
# Each row's CSI data is a string of 128 space-separated integers
data['csi_data'] = data['csi_data'].apply(lambda x: list(map(int, x.split())))

# Extract phases for the first five subcarriers (I and Q values are at positions 2*n and 2*n+1)
data['phases'] = data['csi_data'].apply(lambda x: [calculate_phase(x[2*n], x[2*n+1]) for n in range(5)])

# Plot the phases for each mac_id
mac_ids = data['mac_id'].unique()

for mac_id in mac_ids:
    # Filter the data for the current mac_id
    mac_data = data[data['mac_id'] == mac_id]

    # Create a new figure for each mac_id
    plt.figure(figsize=(10, 6))

    # Plot the phase of the first five subcarriers
    for i in range(5):
        plt.plot(mac_data.index, mac_data['phases'].apply(lambda x: x[i]), label=f'Subcarrier {i+1}')

    # Add labels and title
    plt.title(f"Phase vs Time for MAC ID: {mac_id}")
    plt.xlabel('Time')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate phase from I and Q values
def calculate_phase(i, q):
    return np.arctan2(q, i)  # atan2 is used to get the phase in radians

# Load the data into a pandas DataFrame
file_path = "/home/fawaz/Desktop/usf/directed_research/projects_on_git/esp32/esp32_csi_tool/ESP32-CSI-Tool/active_ap/custom_data_collection/all_macid.csv"  # Replace with your actual file path
data = pd.read_csv(file_path, header=None, names=["mac_id", "csi_data"])

# Parse the csi_data column
# Each row's CSI data is a string of 128 space-separated integers
data['csi_data'] = data['csi_data'].apply(lambda x: list(map(int, x.split())))

# Extract phases for user-defined subcarriers
def extract_phases(csi_data, subcarrier_numbers):
    phases = []
    for n in subcarrier_numbers:
        i = csi_data[2 * n]     # I value for subcarrier n
        q = csi_data[2 * n + 1] # Q value for subcarrier n
        phases.append(calculate_phase(i, q))
    return phases

# Set the list of subcarriers to plot (example: first 3 subcarriers)
subcarrier_numbers = [10, 11, 12, 13, 14]  # Modify this list as needed

# Apply the phase extraction to each row for the selected subcarriers
data['phases'] = data['csi_data'].apply(lambda x: extract_phases(x, subcarrier_numbers))

# Plot the phases for each mac_id
mac_ids = data['mac_id'].unique()

for mac_id in mac_ids:
    # Filter the data for the current mac_id
    mac_data = data[data['mac_id'] == mac_id]

    # Create a new figure for each mac_id
    plt.figure(figsize=(10, 6))

    # Plot the phase of the selected subcarriers
    for i, subcarrier in enumerate(subcarrier_numbers):
        plt.plot(mac_data.index, mac_data['phases'].apply(lambda x: x[i]), label=f'Subcarrier {subcarrier + 1}')

    # Add labels and title
    plt.title(f"Phase vs Time for MAC ID: {mac_id}")
    plt.xlabel('Time')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate phase from I and Q values
def calculate_phase(i, q):
    return np.arctan2(i, q)  # atan2 is used to get the phase in radians

# Load the data into a pandas DataFrame
file_path = "/home/fawaz/Desktop/usf/directed_research/projects_on_git/esp32/esp32_csi_tool/ESP32-CSI-Tool/active_ap/custom_data_collection/all_macid.csv"  # Replace with your actual file path

data = pd.read_csv(file_path, header=None, names=["mac_id", "csi_data"])

# Parse the csi_data column
# Each row's CSI data is a string of 128 space-separated integers
data['csi_data'] = data['csi_data'].apply(lambda x: list(map(int, x.split())))

# Extract phases for user-defined subcarriers
def extract_phases(csi_data, subcarrier_numbers):
    phases = []
    for n in subcarrier_numbers:
        i = csi_data[2 * n]     # I value for subcarrier n
        q = csi_data[2 * n + 1] # Q value for subcarrier n
        phases.append(calculate_phase(i, q))
    return phases

# Set the list of subcarriers to plot (example: first 3 subcarriers)
subcarrier_numbers = [10,11]  # Modify this list as needed

# Time axis limit (can be None for no limit)
time_limit = (100,200)  # Example: Limit the time axis from 0 to 100 (modify as needed)

# Apply the phase extraction to each row for the selected subcarriers
data['phases'] = data['csi_data'].apply(lambda x: extract_phases(x, subcarrier_numbers))

# Plot the phases for each mac_id
mac_ids = data['mac_id'].unique()

for mac_id in mac_ids:
    # Filter the data for the current mac_id
    mac_data = data[data['mac_id'] == mac_id]

    # Create a new figure for each mac_id
    plt.figure(figsize=(10, 6))

    # Plot the phase of the selected subcarriers
    for i, subcarrier in enumerate(subcarrier_numbers):
        plt.plot(mac_data.index, mac_data['phases'].apply(lambda x: x[i]), label=f'Subcarrier {subcarrier + 1}')

    # Add labels and title
    plt.title(f"Phase vs Time for MAC ID: {mac_id}")
    plt.xlabel('Time')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)

    # Set the time axis limit if specified
    if time_limit is not None:
        plt.xlim(time_limit)

    # Show the plot
    plt.show()


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate phase from I and Q values
def calculate_phase(i, q):
    return np.arctan2(q, i)  # atan2 is used to get the phase in radians

# Convert radians to degrees
def radians_to_degrees(radians):
    return np.degrees(radians)

# Load the data into a pandas DataFrame
file_path = file_path = "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/all_macid_csi_antenna.csv"  # Replace with your actual file path

data = pd.read_csv(file_path, header=None, names=["mac_id", "csi_data"])

# Parse the csi_data column
# Each row's CSI data is a string of 128 space-separated integers
data['csi_data'] = data['csi_data'].apply(lambda x: list(map(int, x.split())))

# Extract phases for user-defined subcarriers
def extract_phases(csi_data, subcarrier_numbers):
    phases = []
    for n in subcarrier_numbers:
        i = csi_data[2 * n]     # I value for subcarrier n
        q = csi_data[2 * n + 1] # Q value for subcarrier n
        phase = calculate_phase(i, q)
        phases.append(radians_to_degrees(phase))  # Convert to degrees
    return phases

# Set the list of subcarriers to plot (example: first 3 subcarriers)
subcarrier_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 
 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 
 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] # Modify this list as needed

# Time axis limit (can be None for no limit)
time_limit = (5000, 5100)  # Example: Limit the time axis from 0 to 100 (modify as needed)

# Apply the phase extraction to each row for the selected subcarriers
data['phases'] = data['csi_data'].apply(lambda x: extract_phases(x, subcarrier_numbers))

# Plot the phases for each mac_id
mac_ids = data['mac_id'].unique()

for mac_id in mac_ids:
    # Filter the data for the current mac_id
    mac_data = data[data['mac_id'] == mac_id]

    # Create a new figure for each mac_id
    plt.figure(figsize=(10, 6))

    # Plot the phase of the selected subcarriers
    for i, subcarrier in enumerate(subcarrier_numbers):
        plt.plot(mac_data.index, mac_data['phases'].apply(lambda x: x[i]), label=f'Subcarrier {subcarrier + 1}')

    # Add labels and title
    plt.title(f"Phase vs Time (in Degrees) for MAC ID: {mac_id}")
    plt.xlabel('Time')
    plt.ylabel('Phase (degrees)')
    plt.legend()
    plt.grid(True)

    # Set the time axis limit if specified
    if time_limit is not None:
        plt.xlim(time_limit)

    # Show the plot
    plt.show()



#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Convert radians to degrees
def radians_to_degrees(radians):
    return np.degrees(radians)

# Load the data into a pandas DataFrame
file_path = "/home/fawaz/Desktop/usf/directed_research/projects_on_git/rff_csi_esp32/csi_data_collected/all_macid_csi_antenna.csv"  # Replace with your actual file path
data = pd.read_csv(file_path, header=None, names=["mac_id", "phase_data"])

# Parse the phase_data column
# Each row's phase data is a string of 64 space-separated radians
data['phase_data'] = data['phase_data'].apply(lambda x: list(map(float, x.split())))

# Convert the phase values to degrees for the selected subcarriers
def extract_phases_in_degrees(phase_data, subcarrier_numbers):
    return [radians_to_degrees(phase_data[n]) for n in subcarrier_numbers]

# Set the list of subcarriers to plot (example: first 3 subcarriers)
subcarrier_numbers = [51]  # Modify this list as needed

# Time axis limit (can be None for no limit)
time_limit = (1000, 1100)  # Example: Limit the time axis from 0 to 200 (modify as needed)

# Apply the degree conversion to each row for the selected subcarriers
data['phases'] = data['phase_data'].apply(lambda x: extract_phases_in_degrees(x, subcarrier_numbers))

# Plot the phases for each mac_id
mac_ids = data['mac_id'].unique()

for mac_id in mac_ids:
    # Filter the data for the current mac_id
    mac_data = data[data['mac_id'] == mac_id]

    # Create a new figure for each mac_id
    plt.figure(figsize=(10, 6))

    # Plot the phase of the selected subcarriers
    for i, subcarrier in enumerate(subcarrier_numbers):
        plt.plot(mac_data.index, mac_data['phases'].apply(lambda x: x[i]), label=f'Subcarrier {subcarrier + 1}')

    # Add labels and title
    plt.title(f"Phase vs Time (in Degrees) for MAC ID: {mac_id}")
    plt.xlabel('Time')
    plt.ylabel('Phase (degrees)')
    plt.legend()
    plt.grid(True)

    # Set the time axis limit if specified
    if time_limit is not None:
        plt.xlim(time_limit)

    # Show the plot
    plt.show()

