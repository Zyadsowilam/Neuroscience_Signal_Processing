# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats  # Statistical analysis
from scipy import signal, interpolate  # Additional scientific and signal processing functions
import seaborn as sns  # Making  statistical graphics
from zipfile import ZipFile
import os
from sklearn.preprocessing import StandardScaler
# Load EEG data from CSV files
dfs = pd.read_csv('test.csv')
df = pd.read_csv('EEG_Eye_State_Classification.csv')

# Concatenate the two DataFrames vertically
df = pd.concat([dfs, df], ignore_index=True)
print(df)


Fs = 128  # Sampling rate, which is approximately 128 samples per second
t = np.arange(0, len(df) * 1 / Fs, 1/Fs)  # Create a time vector for the data
cols = df.columns.tolist()[:-1]  # List of column names, excluding the 'eyeDetection' column

# Print the number of null (missing) samples for each column
print('Number of null samples:\n' + str(df.isnull().sum()))

# Display the first few rows of the dataset for visual inspection
df.head()
Y = df['eyeDetection']
X = df.drop(columns='eyeDetection')  # Feature variables containing EEG data
print(X.shape)  # Print the shape (number of samples, number of features) of the feature variables
X.head()  # Display the first few rows of the feature data
df.describe()

# Calculate the mean of each column (feature)
mean_values = np.mean(X, axis=0)

# Calculate the variance of each column (feature)
variance_values = np.var(X, axis=0)



# Create a DataFrame to store the statistical measures
data = {
    'Mean': mean_values.tolist(),
    'Variances': variance_values.tolist()
}
# Create the DataFrame
statistical_table = pd.DataFrame(data)

column_names = X.columns.tolist()
# Set the index of the DataFrame to the column names
statistical_table = statistical_table.set_index([column_names])

# Display the DataFrame
statistical_table
# Standardize the feature variables (X) using z-scores for each column
X = X.apply(stats.zscore, axis=0)

# Replace values in X with NaN if their absolute z-score is greater than 4
X = X.applymap(lambda x: np.nan if (abs(x) > 4) else x)

# Recalculate z-scores for X, ignoring NaN values in the calculations
X = X.apply(stats.zscore, nan_policy='omit', axis=0)

# Replace values in X with NaN if their absolute z-score is greater than 4 (again)
X = X.applymap(lambda x: np.nan if (abs(x) > 4) else x)
# Define a function 'interp' to interpolate missing values using cubic spline
def interp(x):
    # Extract time values 't_temp' corresponding to non-NaN elements
    t_temp = t[x.index[~x.isnull()]]

    # Extract values 'x' corresponding to non-NaN elements
    x = x[x.index[~x.isnull()]]

    # Create a cubic spline interpolation function 'clf' based on 't_temp' and 'x'
    clf = interpolate.interp1d(t_temp, x, kind='cubic')

    # Interpolate the missing values using the cubic spline interpolation
    return clf(t)

# Apply the 'interp' function to interpolate missing values in the feature variables (X)
X_interp = X.apply(interp, axis=0)
# Import the FastICA method from the scikit-learn library
from sklearn.decomposition import FastICA

# Create an ICA object with specified parameters
ica = FastICA(max_iter=2000, random_state=0)

# Apply ICA to the interpolated EEG data (X_interp) to extract independent components
X_pcs = pd.DataFrame(ica.fit_transform(X_interp))

# Rename the columns of X_pcs as 'PC1', 'PC2', 'PC3', etc.
X_pcs.columns = ['PC' + str(ind+1) for ind in range(X_pcs.shape[-1])]

# Drop the first and seventh principal components (component numbering starts from 1)
X_pcs = X_pcs.drop(columns=['PC1', 'PC7'])

# Reconstruct clean EEG data after removing the bad components
# Modify the mixing matrix by removing the first and seventh columns
ica.mixing_ = np.delete(ica.mixing_, [0, 6], axis=1)

# Inverse transform the independent components to obtain cleaned EEG data
X_interp_clean = pd.DataFrame(ica.inverse_transform(X_pcs))

# Rename the columns of X_interp_clean to match the original column names
X_interp_clean.columns = cols

# Define a bandpass filter with a passband between 8-12 Hz
b, a = signal.butter(6, [8 / Fs * 2, 12 / Fs * 2], btype='bandpass')

# Apply the bandpass filter to each column of the cleaned EEG data (X_interp_clean)
# Scale the filtered data to the original scale for visualization purposes
X_interp_clean_alpha = X_interp_clean.apply(
    lambda x: signal.filtfilt(b, a, x) / max(abs(signal.filtfilt(b, a, x))) * max(abs(x)),
    axis=0
)

# Extract the envelope of the alpha waves using the Hilbert transform
X_interp_clean_alpha = X_interp_clean_alpha.apply(lambda x: np.abs(signal.hilbert(x)), axis=0)

# Rename the columns of the data to match the original column names
X_interp_clean_alpha.columns = cols

X_interp_clean.to_csv('original_cleaned_datajj.csv', index=False)
X_interp_clean_alpha.to_csv('filtered_datajj.csv', index=False)
# Set X to the previously processed EEG data containing alpha wave magnitudes
X = X_interp_clean_alpha
# Select specific columns from X
selected_columns = ['AF3', 'F7', 'F3','FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8','F4','F8']
X = X[selected_columns]

# Import PCA from scikit-learn
from sklearn.decomposition import PCA

# Specify the number of components to keep (N)
N = 2

# Create a PCA object with N components
pca = PCA(n_components=N)

# Apply PCA to the EEG data (X) to reduce its dimensionality
X_pca = pd.DataFrame(pca.fit_transform(X), columns=['PC' + str(i+1) for i in range(N)])

# Print the variance ratio explained by the components
print('Variance ratio explained by the components is: ' + str(pca.explained_variance_ratio_))
# Dataset after preprocessing: X
num_columns = X.shape[1]
print("Columns:")
print(num_columns)

num_rows = X.shape[0]
print("Rows:")
print(num_rows)
import numpy as np
from scipy import signal

# EEG data in X (assuming it's a one-dimensional array)
# Define the sampling frequency (Fs) in Hz
Fs = 1000  # Change this to your actual sampling frequency

# Define the frequency range for alpha waves (8-12 Hz)
alpha_band = (8, 12)

# Calculate PSD using the Welch method with corrected noverlap
frequencies, psd = signal.welch(X, fs=Fs, nperseg=1024, noverlap=512, scaling='density', axis=0)

  #If you have a signal with N data points, it is divided into segments of length nperseg with noverlap data points
  #overlapping between adjacent segments.
  #The overlapping segments are used to compute the PSD, and the results are averaged across these segments to
  #produce a more stable and less noisy estimate of the spectral density.
  #You should choose appropriate values for nperseg and noverlap based on your specific data and analysis requirements.
  #The choice depends on the balance you want to strike between time and frequency resolution, and it may
  #require some experimentation and domain knowledge.

# Find the indices corresponding to the alpha band
alpha_indices = np.where((frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1]))

# Extract the PSD values in the alpha band
alpha_psd = psd[alpha_indices, :]
# EEG data in X (with 14980 rows and 10 columns)
# Define the sampling frequency (Fs) in Hz
Fs = 1000  # Change this to your actual sampling frequency

# Define the frequency bands (in Hz)
alpha_band = (8, 12)
beta_band = (12, 30)
theta_band = (4, 8)
delta_band = (1, 4)


# Calculate band power for each frequency band
alpha_power = np.trapz(psd[(frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1]), :], dx=(1 / Fs), axis=0)
beta_power = np.trapz(psd[(frequencies >= beta_band[0]) & (frequencies <= beta_band[1]), :], dx=(1 / Fs), axis=0)
theta_power = np.trapz(psd[(frequencies >= theta_band[0]) & (frequencies <= theta_band[1]), :], dx=(1 / Fs), axis=0)
delta_power = np.trapz(psd[(frequencies >= delta_band[0]) & (frequencies <= delta_band[1]), :], dx=(1 / Fs), axis=0)

import numpy as np
from scipy import signal

# EEG data in X (with 14980 rows and 10 columns)
# Define the sampling frequency (Fs) in Hz
Fs = 1000  # Change this to your actual sampling frequency

# Define the frequency bands (in Hz)
alpha_band = (8, 12)
beta_band = (12, 30)
theta_band = (4, 8)
delta_band = (1, 4)

# Calculate PSD using the Welch method
frequencies, psd = signal.welch(X, fs=Fs, nperseg=1024, noverlap=512, scaling='density', axis=0)

# Calculate band power for each frequency band
alpha_power = np.trapz(psd[(frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1]), :], dx=(1 / Fs), axis=0)
beta_power = np.trapz(psd[(frequencies >= beta_band[0]) & (frequencies <= beta_band[1]), :], dx=(1 / Fs), axis=0)
theta_power = np.trapz(psd[(frequencies >= theta_band[0]) & (frequencies <= theta_band[1]), :], dx=(1 / Fs), axis=0)
delta_power = np.trapz(psd[(frequencies >= delta_band[0]) & (frequencies <= delta_band[1]), :], dx=(1 / Fs), axis=0)

# Calculate band power ratios
alpha_beta_ratio = alpha_power / beta_power
alpha_theta_ratio = alpha_power / theta_power
alpha_delta_ratio = alpha_power / delta_power
beta_alpha_ratio = beta_power / alpha_power
beta_theta_ratio = beta_power / theta_power
beta_delta_ratio = beta_power / delta_power
theta_alpha_ratio = theta_power / alpha_power
theta_beta_ratio = theta_power / beta_power
theta_delta_ratio = theta_power / delta_power
delta_alpha_ratio = delta_power / alpha_power
delta_beta_ratio = delta_power / beta_power
delta_theta_ratio = delta_power / theta_power

# Visualize the band power ratios
import matplotlib.pyplot as plt

ratios = ['Alpha/Beta', 'Alpha/Theta', 'Alpha/Delta', 'Beta/Alpha', 'Beta/Theta', 'Beta/Delta',
          'Theta/Alpha', 'Theta/Beta', 'Theta/Delta', 'Delta/Alpha', 'Delta/Beta', 'Delta/Theta']
values = [alpha_beta_ratio.mean(), alpha_theta_ratio.mean(), alpha_delta_ratio.mean(),
          beta_alpha_ratio.mean(), beta_theta_ratio.mean(), beta_delta_ratio.mean(),
          theta_alpha_ratio.mean(), theta_beta_ratio.mean(), theta_delta_ratio.mean(),
          delta_alpha_ratio.mean(), delta_beta_ratio.mean(), delta_theta_ratio.mean()]
column_names = X.columns.tolist()
print(column_names)

# Assuming X[0:1] is a valid input data point
# print("teet")
# prediction = clf2.predict(X[0:1])
# print("Prediction:", prediction)
scaler = StandardScaler()
print(X.iloc[1, :])
# train an Support Vector Machine (SVM) to classify
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# split train test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=48, test_size=0.8, stratify=Y, shuffle=True)

# normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.metrics import roc_auc_score

# train with grid search
svc = SVC()
parameters = {'gamma': [0.1, 1, 10], 'C': [0.1, 1, 10]}
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)

# predict labels
y_pred = clf.predict(X_test)

# extract accuracy (r2 score)
results = roc_auc_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix

# confusion matrix estimation
conf = confusion_matrix(y_test, y_pred, normalize='true')

# print score
print( 'Score is: ' + str( results ) )
print( 'Best params for the kernel SVM is: ' + str(clf.best_params_) )

print(X[0:1])
import pickle
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)
prediction = clf2.predict(X.iloc[0:1, :])
print("Prediction:", prediction)
# Save the first row to tt.csv
X[0:1].to_csv('tt.csv', index=False)

# Load the saved data for prediction
prediction_data = pd.read_csv('tt.csv')

# Ensure that the features in `prediction_data` match the features used during training
expected_features = X.columns.tolist()  # Assuming `X` is the DataFrame used during training

# Check if all expected features are present in the prediction DataFrame
missing_features = set(expected_features) - set(prediction_data.columns)
if missing_features:
    print(f"Missing features in prediction data: {missing_features}")
else:
    # Ensure the order of features is the same
    prediction_data = prediction_data[expected_features]

    # Apply the same scaling used during training
    scaled_data = scaler.transform(prediction_data)

    # Make predictions
    prediction = clf2.predict(scaled_data)
    print("Prediction2222:", prediction)
