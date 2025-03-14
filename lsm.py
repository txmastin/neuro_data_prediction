import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import optuna
import joblib  # For parallel execution


def load_tsv_labels(tsv_path):
    """
    Load subject labels from a TSV file and map group labels to numerical values.
    """
    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    
    # Define mapping for group labels
    group_mapping = {'A': 0, 'C': 1, 'F': 2}  # Assign numerical labels
    
    label_mapping = {row['participant_id']: group_mapping[row['group']] for _, row in df.iterrows() if row['group'] in group_mapping}
    return label_mapping

def load_eeg_data(file_path):
    """
    Load EEG data from a .set file.
    """
    eeg_data = mne.io.read_raw_eeglab(file_path, preload=True)
    return eeg_data

def segment_eeg(eeg_data, window_size=1000, stride=250):
    """
    Segment EEG data into fixed-size overlapping windows.
    - window_size: Number of time points per window (e.g., 500 for 1s at 500Hz)
    - stride: Overlapping step size (e.g., 250 for 50% overlap)
    """
    eeg_array = eeg_data.get_data() * 1e6 # Shape: (n_channels, n_timepoints) scaled up
    n_channels, n_timepoints = eeg_array.shape
    windows = []
    for start in range(0, n_timepoints - window_size, stride):
        window = eeg_array[:, start:start + window_size]
        windows.append(window)

    return np.array(windows)  # Shape: (n_windows, n_channels, window_size)

def load_and_segment_eeg(data_root, tsv_path, window_size=1000, stride=750):
    """
    Load EEG recordings from nested subject directories, segment them, and apply class labels from TSV file.
    - data_root: Root directory containing subject subdirectories with .set EEG files
    - tsv_path: Path to the participants.tsv file containing subject labels
    """
    label_mapping = load_tsv_labels(tsv_path)
    all_windows = []
    all_labels = []
    
    for subject in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject, "eeg")
        if os.path.isdir(subject_path):
            for file_name in os.listdir(subject_path):
                if file_name.endswith(".set"):
                    file_path = os.path.join(subject_path, file_name)
                    eeg_data = load_eeg_data(file_path)
                    windows = segment_eeg(eeg_data, window_size, stride)
                    
                    label = label_mapping.get(subject, None)
                    if label is not None:
                        labels = np.full((windows.shape[0],), label)
                        all_windows.append(windows)
                        all_labels.append(labels)
    
    return np.vstack(all_windows), np.concatenate(all_labels)


class SpikingLiquidStateMachine:
    def __init__(self, 
                 n_reservoir=1000, 
                 connectivity=0.2, 
                 spectral_radius=0.9, 
                 input_scaling=0.5, 
                 leak_rate=0.2, 
                 threshold=0.5, 
                 resting_potential=0.0, 
                 refractory_period=2,
                 n_inputs=19,  # Number of EEG channels
                 n_outputs=3):  # Number of classes
        
        self.n_reservoir = n_reservoir
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.threshold = threshold
        self.resting_potential = resting_potential
        self.refractory_period = refractory_period

        # Initialize reservoir weights
        self.W = np.random.rand(n_reservoir, n_reservoir)
        self.W[np.random.rand(*self.W.shape) > connectivity] = 0
        self.W = self.W - np.diag(np.diag(self.W))  # Remove self-connections
        self.W = self.W / np.max(np.abs(np.linalg.eigvals(self.W))) * spectral_radius

        # Initialize input weights (expanded to match 19 EEG inputs)
        self.W_in = np.random.rand(n_reservoir, n_inputs) 
        
        # Initialize output weights (multi-class)
        self.W_out = np.random.rand(n_outputs, n_reservoir)
        self.W_out[np.random.rand(*self.W_out.shape) > connectivity] = 0
        
        # Initialize neuron states
        self.neuron_states = np.zeros(n_reservoir)
        self.neuron_spikes = np.zeros(n_reservoir)
        self.fired = np.zeros(n_reservoir, dtype=bool)
        self.refractory_counters = np.zeros(n_reservoir, dtype=int)
    
    def step(self, input_signal):
        """
        Process a single time step of multi-channel EEG input.
        """
        self.neuron_spikes = self.fired.astype(int) 
        total_input = np.dot(self.W, self.neuron_spikes) + np.dot(self.W_in, input_signal) * self.input_scaling

        # Refractory handling: block input accumulation for refractory neurons
        refractory_mask = self.refractory_counters > 0
        total_input[refractory_mask] = 0

        # Update neuron states with leak and input
        self.neuron_states = (1 - self.leak_rate) * self.neuron_states + total_input

        # Detect spiking neurons
        self.fired = self.neuron_states > self.threshold
        self.neuron_states[self.fired] = self.resting_potential

        # Set refractory period
        self.refractory_counters[self.fired] = self.refractory_period
        self.refractory_counters[refractory_mask] -= 1

        return self.neuron_states, sum(self.fired)
    
    def predict(self, reservoir_feature):
        """
        Multi-class prediction using softmax.
        """
        logits = np.dot(self.W_out, reservoir_feature)
        exp_logits = np.exp(logits - np.max(logits))  # Avoid overflow
        return exp_logits / np.sum(exp_logits)  # Softmax activation

def train_lsm(lsm, X_train, y_train, learning_rate=0.001, epochs=3, batch_size=64):
    """
    Train the Liquid State Machine (LSM) using mini-batch training.
    """
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)  # Shuffle data
        X_train, y_train = X_train[indices], y_train[indices]
        epoch_loss = 0
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            batch_loss = 0
            for x, y in zip(X_batch, y_batch):
                reservoir_memory = []
                for t in range(x.shape[1]):  # Iterate over time steps
                    reservoir_activations, _ = lsm.step(x[:, t])
                    reservoir_memory.append(reservoir_activations)
                
                reservoir_feature = np.mean(reservoir_memory, axis=0)
                prediction = lsm.predict(reservoir_feature)
                
                error = np.zeros(3)
                error[int(y)] = 1  # One-hot target
                error -= prediction
                
                lsm.W_out += learning_rate * np.outer(error, reservoir_feature)  # Update weights
                batch_loss += np.linalg.norm(error)
                
            epoch_loss += batch_loss / batch_size
        

def evaluate_lsm(lsm, X_test, y_test):
    """
    Evaluate the trained LSM on a test set.
    """
    y_pred = []
    for x in X_test:
        reservoir_memory = []
        for t in range(x.shape[1]):
            reservoir_activations, _ = lsm.step(x[:, t])
            reservoir_memory.append(reservoir_activations)
        
        reservoir_feature = np.mean(reservoir_memory, axis=0)
        prediction = lsm.predict(reservoir_feature)
        y_pred.append(np.argmax(prediction))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


data_root = "../ds004504/derivatives/"
tsv_path = "../ds004504/participants.tsv"

'''
windows, labels = load_and_segment_eeg(data_root, tsv_path)
print("Segmented EEG shape:", windows.shape)  # (n_windows, 19, window_size)
print("Labels shape:", labels.shape)  # (n_windows,)


from collections import Counter

label_counts = Counter(labels)
print("Label distribution:", label_counts)

print("First segmented window shape:", windows[0].shape)
print("First window values (Channel 0):", windows[0, 0, :10])
'''

X, y = load_and_segment_eeg(data_root, tsv_path)
X = X[:, :, :1000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    """
    Objective function for Optuna to optimize LSM hyperparameters.
    """
    # Define the hyperparameters to optimize
    n_reservoir = trial.suggest_int("n_reservoir", 300, 900, step=300)
    connectivity = trial.suggest_float("connectivity", 0.1, 0.4, step=0.05)
    spectral_radius = trial.suggest_float("spectral_radius", 0.1, 1.5)
    input_scaling = trial.suggest_float("input_scaling", 0.1, 1.0)
    leak_rate = trial.suggest_float("leak_rate", 0.05, 0.5)
    threshold = trial.suggest_float("threshold", 0.1, 1.0)
    refractory_period = trial.suggest_int("refractory_period", 1, 5)

    # Create LSM with sampled hyperparameters
    lsm = SpikingLiquidStateMachine(
        n_reservoir=n_reservoir,
        connectivity=connectivity,
        spectral_radius=spectral_radius,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        threshold=threshold,
        refractory_period=refractory_period
    )

    # Train and evaluate the LSM
    train_lsm(lsm, X_train, y_train, learning_rate=0.001, epochs=3, batch_size=64)
    accuracy = evaluate_lsm(lsm, X_test, y_test)

    return accuracy  # Optuna maximizes this value

# Run the Optuna optimization with parallelization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, n_jobs=-1, show_progress_bar=True)  # Use all CPU cores

# Print best parameters found
print("Best Hyperparameters:", study.best_params)

joblib.dump(study, "optuna_lsm_study.pkl")

'''
import numpy as np
import matplotlib.pyplot as plt

# Pick a random sample index
idx = np.random.randint(len(labels))

# Plot EEG signals for that sample
plt.figure(figsize=(10, 6))
for i in range(windows.shape[1]):  # Loop over 19 EEG channels
    plt.plot(windows[idx, i, :] + i * 10, label=f'Ch {i}')  # Offset each channel

plt.title(f"EEG Window #{idx} | Label: {labels[idx]}")
plt.xlabel("Time (samples)")
plt.ylabel("EEG Signal (offset for visibility)")
plt.legend()
plt.show()
'''
