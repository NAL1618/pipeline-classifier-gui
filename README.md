# pipeline-classifier-gui

CNN Pipeline Classifier GUI
1. Overview
This application provides a graphical user interface for analyzing pipeline sensor data using a pre-trained Convolutional Neural Network (CNN). It supports both pre-recorded simulation data (.mat files) and live experimental data streams (.csv files), offering real-time visualization, processing, and prediction of pipeline defects.

2. Prerequisites
Before you begin, ensure you have Python 3.8 or newer installed on your system. You can check this by running python --version in your terminal.

3. Installation and Setup
Follow these steps to set up the application on a new computer.

    Step 1: Get the Project Files, download and place all the project files into a single folder. The essential files are: gui.py (The main      application script), predict_backend.py (The code that loads and runs the model), Your pre-trained model file (model.h5)
    
    Step 2: Install Required Libraries

                        import tkinter as tk
                        from tkinter import ttk, filedialog, messagebox, simpledialog
                        import numpy as np
                        import pandas as pd
                        import os
                        import time
                        import csv
                        import threading
                        import traceback
                        import re
                        import queue
                        import scipy.io
                        import matplotlib.pyplot as plt
                        import librosa
                        import librosa.display

4. Required Folder Structure
For the application to find your data correctly, you must create two folders inside your main project folder:

input_folder: This is where you will place your simulation data (.mat files).

exp_input_folder: This is where the application will watch for new experimental data. The data must be organized in subfolders for each sensor.

Your final project directory should look like this:

Your_Project_Folder/
gui.py                      # The main application script
predict_backend.py          # The AI model backend

your_model_file.h5          # Your trained AI model

input_folder/               # For simulation data

simulation_data.mat

exp_input_folder/           # For live experimental data

Sensor_Location_1/

20250805-1/

data_001.csv

data_002.csv

Sensor_Location_2/

20250805-1/

data_001.csv

data_002.csv
            
5. How to Use the Application
Step 1: Run the GUI

python gui.py in terminal or just run python file.

The application window will appear with several tabs.
