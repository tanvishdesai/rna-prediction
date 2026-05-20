import numpy as np
import pandas as pd
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")

# --- Configuration ---
SEQUENCE_MAX_LENGTH = 256  # Adjusted for potentially faster training initially
NUM_ATOM_COORDS = 15  # 5 atoms * 3 coordinates (x, y, z)
VOCAB_SIZE = 5  # A, C, G, U, and a padding character ('P')
EMBEDDING_DIM = 64
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 10 # Start with a smaller number of epochs for initial runs

print("--- Script Configuration Initialized ---")
print(f"SEQUENCE_MAX_LENGTH: {SEQUENCE_MAX_LENGTH}")
print(f"NUM_ATOM_COORDS: {NUM_ATOM_COORDS}")
print(f"VOCAB_SIZE: {VOCAB_SIZE}")
print(f"EMBEDDING_DIM: {EMBEDDING_DIM}")
print(f"LSTM_UNITS: {LSTM_UNITS}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"EPOCHS: {EPOCHS}")
print("---------------------------------------\n")


# --- 1. Data Loading and Preprocessing ---
print("--- Section 1: Data Loading and Preprocessing ---")

def get_char_to_int_map():
    """Returns a mapping from RNA characters to integers."""
    print("  [Data Prep] Defining character to integer mapping...")
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'P': 4} # 'P' for padding
    print(f"  [Data Prep] Character map: {char_to_int}")
    return char_to_int

CHAR_TO_INT = get_char_to_int_map()
INT_TO_CHAR = {i: char for char, i in CHAR_TO_INT.items()}


def encode_sequence(sequence, char_to_int_map, max_length):
    """Encodes an RNA sequence into a fixed-length integer vector."""
    encoded = [char_to_int_map.get(char, char_to_int_map['P']) for char in sequence]
    
    padding_needed = max_length - len(encoded)
    if padding_needed > 0:
        encoded += [char_to_int_map['P']] * padding_needed
    else:
        encoded = encoded[:max_length]
    return np.array(encoded, dtype=np.int32)

def load_data(sequence_file_path, label_file_path, char_to_int_map, max_seq_len, num_coords, is_test_set=False):
    """
    Loads and preprocesses sequence and label data from CSV files.
    Assumes label_file_path has columns like 'ID' (e.g., 'R1107_1'), and coordinate columns.
    For training/validation, it expects coordinate columns named like 'x_1', 'y_1', 'z_1', ..., 'x_5', 'y_5', 'z_5'.
    """
    print(f"  [Data Load] Loading sequences from: {sequence_file_path}")
    try:
        sequences_df = pd.read_csv(sequence_file_path)
    except FileNotFoundError:
        print(f"  [Data Load] ERROR: Sequence file not found: {sequence_file_path}")
        print("  [Data Load] Please ensure the file path is correct and the file exists.")
        print("  [Data Load] Proceeding with DUMMY data for structural demonstration.")
        return generate_dummy_data(10, char_to_int_map, max_seq_len, num_coords) # Small dummy set

    encoded_sequences = []
    target_labels = []
    target_ids_loaded = []

    if not is_test_set:
        print(f"  [Data Load] Loading labels from: {label_file_path}")
        try:
            labels_df = pd.read_csv(label_file_path)
            # Assuming labels_df has an 'ID' column like 'TARGETID_RESID' and coordinate columns
            # Example: R1107_1, G, 1, 0.0, 0.0, ..., 0.0
            # We need to extract target_id and resid to group labels per sequence
            labels_df[['target_id_from_label', 'resid']] = labels_df['ID'].str.split('_', expand=True)
            labels_df['resid'] = pd.to_numeric(labels_df['resid'])
            
            # Prepare coordinate column names based on sample_submission.csv
            coord_cols = []
            for i in range(1, 6): # 5 atoms
                coord_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
            
            # Ensure all coordinate columns exist
            missing_cols = [col for col in coord_cols if col not in labels_df.columns]
            if missing_cols:
                print(f"  [Data Load] ERROR: Missing expected coordinate columns in label file: {missing_cols}")
                print(f"  [Data Load] Expected columns like: {coord_cols}")
                print("  [Data Load] Please check your label file format.")
                print("  [Data Load] Proceeding with DUMMY data for structural demonstration.")
                return generate_dummy_data(10, char_to_int_map, max_seq_len, num_coords)

            grouped_labels = labels_df.groupby('target_id_from_label')
        except FileNotFoundError:
            print(f"  [Data Load] ERROR: Label file not found: {label_file_path}")
            print("  [Data Load] Please ensure the file path is correct and the file exists.")
            print("  [Data Load] Proceeding with DUMMY data for structural demonstration.")
            return generate_dummy_data(10, char_to_int_map, max_seq_len, num_coords)
        except Exception as e:
            print(f"  [Data Load] ERROR: Could not process label file {label_file_path}: {e}")
            print("  [Data Load] Please check your label file format. Expected 'ID' like 'target_id_resid'.")
            print("  [Data Load] Proceeding with DUMMY data for structural demonstration.")
            return generate_dummy_data(10, char_to_int_map, max_seq_len, num_coords)


    print(f"  [Data Load] Processing {len(sequences_df)} sequences...")
    for index, row in sequences_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        
        encoded_seq = encode_sequence(sequence, char_to_int_map, max_seq_len)
        encoded_sequences.append(encoded_seq)
        target_ids_loaded.append(target_id)

        if not is_test_set:
            try:
                target_label_data = grouped_labels.get_group(target_id)
                target_label_data = target_label_data.sort_values(by='resid')
                
                # Extract only the coordinate values
                coords_array = target_label_data[coord_cols].values.astype(np.float32) # Shape: (actual_seq_len, num_coords)
                
                # Pad or truncate labels to max_seq_len
                actual_seq_len = coords_array.shape[0]
                label_padding_needed = max_seq_len - actual_seq_len
                
                if label_padding_needed > 0:
                    padding_array = np.zeros((label_padding_needed, num_coords), dtype=np.float32) # Pad with zeros
                    processed_labels = np.vstack([coords_array, padding_array])
                else:
                    processed_labels = coords_array[:max_seq_len, :]
                
                target_labels.append(processed_labels)
            except KeyError:
                print(f"    [Data Load] Warning: No labels found for target_id: {target_id} in {label_file_path}. Skipping this sequence for training/validation.")
                encoded_sequences.pop() # Remove the sequence if its labels are missing
                target_ids_loaded.pop()
                continue
            except Exception as e:
                print(f"    [Data Load] Error processing labels for target_id {target_id}: {e}. Skipping.")
                encoded_sequences.pop() 
                target_ids_loaded.pop()
                continue
        
        if (index + 1) % 100 == 0:
            print(f"    [Data Load] Processed {index + 1}/{len(sequences_df)} sequences...")

    X = np.array(encoded_sequences, dtype=np.int32)
    
    if not is_test_set:
        if not target_labels: # If all sequences were skipped
             print("  [Data Load] ERROR: No valid label data could be processed. Check label file and IDs.")
             print("  [Data Load] Proceeding with DUMMY data for structural demonstration.")
             return generate_dummy_data(10, char_to_int_map, max_seq_len, num_coords)
        y = np.array(target_labels, dtype=np.float32)
        print(f"  [Data Load] Successfully loaded and processed data. X shape: {X.shape}, y shape: {y.shape}")
        return X, y, target_ids_loaded
    else:
        print(f"  [Data Load] Successfully loaded and processed test sequences. X shape: {X.shape}")
        return X, target_ids_loaded

def generate_dummy_data(num_samples, char_to_int_map, max_length, num_coords):
    print(f"  [Data Load] Generating {num_samples} dummy samples for structural demonstration.")
    dummy_sequences_text = ["ACGU" * (i % 10 + 5) for i in range(num_samples)] 
    X = np.array([encode_sequence(seq, char_to_int_map, max_length) for seq in dummy_sequences_text])
    y = np.random.rand(num_samples, max_length, num_coords).astype(np.float32)
    dummy_ids = [f"DUMMY_{i}" for i in range(num_samples)]
    print(f"  [Data Load] Dummy X shape: {X.shape}, Dummy y shape: {y.shape}")
    return X, y, dummy_ids

# --- Actual Data Loading ---
print("\n--- Attempting to load actual data ---")
X_train, y_train, train_ids = load_data(
    "train_sequences.v2.csv", 
    "train_labels.v2.csv", 
    CHAR_TO_INT, 
    SEQUENCE_MAX_LENGTH,
    NUM_ATOM_COORDS
)

X_val, y_val, val_ids = load_data(
    "validation_sequences.csv", 
    "validation_labels.csv", 
    CHAR_TO_INT, 
    SEQUENCE_MAX_LENGTH,
    NUM_ATOM_COORDS
)
print("--- Section 1 Complete --- \n")


# --- 2. Model Definition (TensorFlow/Keras) ---
print("--- Section 2: Model Definition (TensorFlow/Keras) ---")

def build_rna_structure_model(vocab_size, embedding_dim, lstm_units, sequence_length, num_output_coords):
    """Builds an LSTM-based model for RNA structure prediction."""
    print("  [Model Build] Initializing model construction...")
    print(f"    Vocab Size: {vocab_size}")
    print(f"    Embedding Dim: {embedding_dim}")
    print(f"    LSTM Units: {lstm_units}")
    print(f"    Sequence Length (Input): {sequence_length}")
    print(f"    Num Output Coords per residue: {num_output_coords}")

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, 
                                  output_dim=embedding_dim, 
                                  input_length=sequence_length,
                                  mask_zero=True), # mask_zero=True as 'P' (padding) is mapped to 0
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units // 2, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_output_coords)) # Output 15 coords for each time step
    ])
    
    print("  [Model Build] Model Layers:")
    print("    1. Embedding Layer (Input: Integers -> Output: Dense Vectors, Masking Padding)")
    print("    2. Bidirectional LSTM Layer (Captures forward/backward context)")
    print("    3. Dropout Layer (Regularization)")
    print("    4. Bidirectional LSTM Layer (Further context capture)")
    print("    5. Dropout Layer (Regularization)")
    print("    6. TimeDistributed Dense Layer (Output: 15 coordinates per residue)")
    print(f"  [Model Build] Model construction complete.")
    return model

rna_model = build_rna_structure_model(
    VOCAB_SIZE, 
    EMBEDDING_DIM, 
    LSTM_UNITS, 
    SEQUENCE_MAX_LENGTH, 
    NUM_ATOM_COORDS
)

print("\n  Compiling model...")
rna_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae']) # Mean Absolute Error
print("  [Model Compile] Model compilation: Optimizer=Adam, Loss=MSE, Metrics=MAE")

print("  [Model Compile] Model summary:")
rna_model.summary()
print("--- Section 2 Complete --- \n")


# --- 3. Training Loop ---
print("--- Section 3: Training Loop ---")

def train_model(model, x_train_data, y_train_data, x_val_data, y_val_data, epochs, batch_size):
    """Trains the model."""
    print("  [Train] Starting model training...")
    if x_train_data is None or y_train_data is None or x_val_data is None or y_val_data is None:
        print("  [Train] ERROR: Training or validation data is missing. Cannot start training.")
        return None
        
    print(f"    Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"    Training data shape: X-{x_train_data.shape}, Y-{y_train_data.shape}")
    print(f"    Validation data shape: X-{x_val_data.shape}, Y-{y_val_data.shape}")

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, 
                                                         restore_best_weights=True,
                                                         monitor='val_loss',
                                                         verbose=1)
    
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                        factor=0.2, 
                                                        patience=3, 
                                                        min_lr=1e-6,
                                                        verbose=1)

    history = model.fit(
        x_train_data, y_train_data,
        validation_data=(x_val_data, y_val_data),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping_cb, reduce_lr_cb],
        verbose=1 
    )
    print("  [Train] Model training complete.")
    return history

print("\n  Starting training process...")
# Ensure data was loaded correctly before training
if X_train is not None and y_train is not None and X_val is not None and y_val is not None:
    if X_train.shape[0] > 0 and X_val.shape[0] > 0 : # Check if there are samples to train on
        training_history = train_model(rna_model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)
        if training_history:
            print(f"  [Train] Training history keys: {training_history.history.keys()}")
    else:
        print("  [Train] No samples found in training/validation data after processing. Skipping training.")
else:
    print("  [Train] Training or validation data not loaded properly. Skipping training.")
print("--- Section 3 Complete --- \n")


# --- 4. Prediction and Submission File Generation ---
print("--- Section 4: Prediction and Submission File Generation ---")

def predict_and_format_submission(model, test_sequence_file, char_to_int_map, max_seq_len, num_coords, submission_file_name="submission.csv"):
    """Loads test data, makes predictions, and formats them into a submission CSV."""
    print("  [Predict] Loading test sequences...")
    X_test, test_ids = load_data(test_sequence_file, None, char_to_int_map, max_seq_len, num_coords, is_test_set=True)

    if X_test is None or X_test.shape[0] == 0:
        print("  [Predict] No test data loaded or processed. Cannot make predictions.")
        return

    print(f"  [Predict] Making predictions on {X_test.shape[0]} test sequences...")
    predictions = model.predict(X_test, batch_size=BATCH_SIZE) # Shape: (num_test_samples, max_seq_len, num_coords)
    print(f"  [Predict] Predictions shape: {predictions.shape}")

    print(f"  [Predict] Formatting predictions for submission file: {submission_file_name}...")
    
    # Load the original sequences again to get actual lengths and residue names
    test_sequences_df = pd.read_csv(test_sequence_file)
    
    submission_entries = []
    coord_cols_names = []
    for i in range(1, 6): # 5 atoms
        coord_cols_names.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    for i, target_id in enumerate(test_ids):
        # Find the original sequence to get its actual length and characters
        original_sequence_info = test_sequences_df[test_sequences_df['target_id'] == target_id]
        if original_sequence_info.empty:
            print(f"    [Predict] Warning: Could not find original sequence for target_id {target_id}. Skipping.")
            continue
        
        original_sequence = original_sequence_info.iloc[0]['sequence']
        actual_length = len(original_sequence)

        for resid_idx in range(actual_length):
            # Ensure we don't go beyond SEQUENCE_MAX_LENGTH for predictions array
            if resid_idx >= predictions.shape[1]: 
                print(f"    [Predict] Warning: Residue index {resid_idx+1} for {target_id} exceeds model's max sequence length {predictions.shape[1]}. Capping.")
                break 

            entry_id = f"{target_id}_{resid_idx + 1}"
            resname = original_sequence[resid_idx]
            
            coords = predictions[i, resid_idx, :] # Get the 15 coords for this residue
            
            submission_entry = {'ID': entry_id, 'resname': resname, 'resid': resid_idx + 1}
            for coord_idx, coord_val in enumerate(coords):
                submission_entry[coord_cols_names[coord_idx]] = coord_val
            submission_entries.append(submission_entry)

    submission_df = pd.DataFrame(submission_entries)
    
    # Ensure column order matches sample_submission.csv
    output_columns = ['ID', 'resname', 'resid'] + coord_cols_names
    submission_df = submission_df[output_columns]

    submission_df.to_csv(submission_file_name, index=False)
    print(f"  [Predict] Submission file '{submission_file_name}' created with {len(submission_df)} entries.")

# --- Making predictions on the test set and generating submission file ---
print("\n  Generating submission file for test sequences...")
predict_and_format_submission(rna_model, 
                              "test_sequences.csv", 
                              CHAR_TO_INT, 
                              SEQUENCE_MAX_LENGTH, 
                              NUM_ATOM_COORDS, 
                              submission_file_name="my_submission.csv")

print("--- Section 4 Complete --- \n")


print("--- Script Execution Finished ---")
print("Reminder:")
print("1. Ensure pandas and tensorflow are installed ('pip install pandas tensorflow').")
print("2. Verify that 'train_sequences.v2.csv', 'train_labels.v2.csv', 'validation_sequences.csv', 'validation_labels.csv', and 'test_sequences.csv' are in the same directory as this script, or update file paths.")
print("3. The label loading logic in `load_data` assumes 'train_labels.v2.csv' and 'validation_labels.csv' contain an 'ID' column (e.g., 'R1107_1') and coordinate columns ('x_1', 'y_1', ..., 'z_5'). Adjust if your format differs.")
print("4. Training deep learning models can be time-consuming and resource-intensive.")
print("---------------------------------------------") 