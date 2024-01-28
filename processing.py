import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import interpolate, signal, stats
from sklearn.decomposition import FastICA
import seaborn as sns  # Making  statistical graphics
import matplotlib.pyplot as plt
# Function to plot data

# Load EEG_Eye_State_Classification.csv
df = pd.read_csv('EEG_Eye_State_Classification.csv')

# Load test.csv
test_df = pd.read_csv('test.csv')
print(test_df)
# Find the matching row in EEG_Eye_State_Classification.csv
matching_row = df[df.isin(test_df.to_dict(orient='list')).all(axis=1)]
print("1")
print(matching_row)
# Remove the matching row from its original position
df = df[df.index != matching_row.index[0]]
# Remove the matching row from its original position
df = df[df.index != matching_row.index[0]]

# Insert the matching row as the first row
df = pd.concat([matching_row, df]).reset_index(drop=True)

print(df)
# df = pd.read_csv('EEG_Eye_State_Classification.csv')


Fs = 128
t = np.arange(0, len(df) * 1 / Fs, 1/Fs)
cols = df.columns.tolist()[:-1]

Y = df['eyeDetection']
X = df.drop(columns='eyeDetection')

X = X.apply(stats.zscore, axis=0)
X = X.applymap(lambda x: np.nan if (abs(x) > 4) else x)
X = X.apply(stats.zscore, nan_policy='omit', axis=0)
X = X.applymap(lambda x: np.nan if (abs(x) > 4) else x)

def interp(x):
    t_temp = t[x.index[~x.isnull()]]
    x = x[x.index[~x.isnull()]]
    clf = interpolate.interp1d(t_temp, x, kind='cubic')
    return clf(t)

X_interp = X.apply(interp, axis=0)

# Apply Independent Component Analysis (ICA) to extract independent components
ica = FastICA(max_iter=2000, random_state=0)
X_pcs = pd.DataFrame(ica.fit_transform(X_interp))

X_pcs.columns = ['PC' + str(ind+1) for ind in range(X_pcs.shape[-1])]
X_pcs = X_pcs.drop(columns=['PC1', 'PC7'])


# Reconstruct clean EEG data after removing bad components
ica.mixing_ = np.delete(ica.mixing_, [0, 6], axis=1)
X_interp_clean = pd.DataFrame(ica.inverse_transform(X_pcs))
X_interp_clean.columns = cols

# Apply bandpass filter to the cleaned EEG data in the alpha range (8-12 Hz)
b, a = signal.butter(6, [8 / Fs * 2, 12 / Fs * 2], btype='bandpass')
X_interp_clean_alpha = X_interp_clean.apply(
    lambda x: signal.filtfilt(b, a, x) / max(abs(signal.filtfilt(b, a, x))) * max(abs(x)),
    axis=0
)

X_interp_clean_alpha = X_interp_clean_alpha.apply(lambda x: np.abs(signal.hilbert(x)), axis=0)
X_interp_clean_alpha.columns = cols

X_interp_clean.to_csv('original_cleaned_data.csv', index=False)
X_interp_clean_alpha.to_csv('filtered_data.csv', index=False)

def plot_dataf(X, xlim=[0, 20]):
    fig = go.Figure()

    for ind_data, data in enumerate(X):
        for ind, col in enumerate(data.columns.tolist()):
            fig.add_trace(go.Scatter(x=t if ind_data == 0 else frequencies_filter_response,
                                     y=5 * ind + stats.zscore(data[col], nan_policy='omit') if ind_data == 0 else stats.zscore(data[col], nan_policy='omit'),
                                     mode='lines', line=dict(width=0.5), name=col))

    fig.update_layout(legend=dict(orientation="h", x=0, y=1.1))
    fig.update_layout(xaxis=dict(range=xlim))
    # fig.show()# Frequency response of the bandpass filter
w, h = signal.freqz(b, a, worN=8000)
frequencies_filter_response = (Fs * 0.5 / np.pi) * w

# Plot the frequency response
fig = go.Figure()
fig.add_trace(go.Scatter(x=frequencies_filter_response, y=np.abs(h), mode='lines', name='Bandpass Filter Response'))
fig.update_layout(xaxis_type="log", yaxis_type="linear", title='Bandpass Filter Frequency Response')
# fig.show()
# Set X to the previously processed EEG data containing alpha wave magnitudes
X = X_interp_clean_alpha

# Calculate the correlation matrix between the columns of X
Cols_corr = X.corr()

# Create a heatmap to visualize the correlations between columns
plt.figure(figsize=(10, 10))
sns.heatmap(Cols_corr, annot=True, annot_kws={'fontsize': 12})

# Initialize a list 'cols_drop_ind' to keep track of columns to be dropped
cols_drop_ind = [0] * len(cols)

# Iterate through pairs of columns in X to find high correlations
for i in range(len(cols)):
    for j in range(len(cols)):
        # Check if the correlation is high (>= 0.8) and make a note to drop one of them
        if (i < j) and abs(Cols_corr.iloc[i, j] >= 0.8):
            cols_drop_ind[j] = 1

# Create a list 'cols_drop' containing column names to be dropped
cols_drop = [cols[ind] for ind in range(len(cols_drop_ind)) if cols_drop_ind[ind]]

# Drop the high-correlation columns from the EEG data (X)
X.drop(columns=cols_drop, inplace=True)

# Create a new heatmap to visualize the correlations after dropping the high-correlation columns
plt.figure(figsize=(10, 10))
sns.heatmap(X.corr(), annot=True, annot_kws={'fontsize': 12})
# plt.show()
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


# plt.show()
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

# Visualize the PSD
import matplotlib.pyplot as plt

# plt.show()
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

# plt.show()
column_names = X.columns.tolist()
print(column_names)
# train an Support Vector Machine (SVM) to classify
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# split train test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=48, test_size=0.2, stratify=Y, shuffle=True)

# normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("2")

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
