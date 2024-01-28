import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import interpolate, signal, stats
from sklearn.decomposition import FastICA
import seaborn as sns  # Making  statistical graphics
import matplotlib.pyplot as plt
import pickle
try:
    with open('communication.txt', 'r') as file:
        data_from_pp = file.read()
    # number = float(data_from_pp)
    # result = number * 2
    # Load EEG data from CSV files
    from io import StringIO
    with open('model.pkl', 'rb') as f:
        clf2 = pickle.load(f)
    # Apply the same scaling used during training
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

   

    # Convert the string representation of the list to a numpy array
    data = np.array(eval(data_from_pp))

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data using the scaler
    # scaled_data =
    scaled_data =data
    # [[-0.55672853,-0.91280347,-1.1533564,-0.7541272,-1.58236466,-0.96557653,-1.03063777,0.90521996,-1.40929721,-1.39877988]]
    #scaled_data =[[-0.53694436,-0.94138794,-1.33146599,-1.09214149,-1.22915931,-0.76901927,-0.97131742,-0.1683458,-0.94979416,-1.45451185,-1.3172248,-0.83306618]]
    # Make predictions
    prediction = clf2.predict(scaled_data)
    print("Prediction2222:", prediction)
        # Write the processed data back to the file
    with open('communication.txt', 'w') as file:
        file.write(str(prediction))
except ValueError:
    result = 'Invalid input'