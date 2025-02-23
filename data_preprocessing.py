import numpy as np
import os
from glob import glob
from scipy.interpolate import interp1d


def read_sequence(folder_path):
    npy_files = sorted(glob(os.path.join(folder_path, "*.npy")))
    sequence = []
    for npy_file in npy_files:
        frame = np.load(npy_file)
        sequence.append(frame)
    return np.array(sequence)


def normalize_sequence_length(sequence, target_length=100):
    current_length = sequence.shape[0]
    if current_length == target_length:
        return sequence

    t = np.linspace(0, 1, current_length)
    t_new = np.linspace(0, 1, target_length)

    original_shape = sequence.shape
    sequence_2d = sequence.reshape(current_length, -1)

    f = interp1d(t, sequence_2d, axis=0)
    new_sequence = f(t_new)

    return new_sequence.reshape(target_length, original_shape[1])


def read_all_sequences(main_folder, target_length=100):
    sequences = []
    for i in range(120):  # 0 to 119
        sequence_folder = os.path.join(main_folder, str(i))
        if os.path.exists(sequence_folder):
            sequence = read_sequence(sequence_folder)
            # Normalize sequence length
            normalized_sequence = normalize_sequence_length(sequence, target_length)
            sequences.append(normalized_sequence)
            print(
                f"Processed folder {i}, Original shape: {sequence.shape}, Normalized shape: {normalized_sequence.shape}")
    return np.array(sequences)  # Now all sequences will have same length


if __name__ == "__main__":
    # Set target length for all sequences
    TARGET_LENGTH = 100  # You can adjust this value

    dataset_path = "dataset"  # Replace with your path
    valid_path = os.path.join(dataset_path, "Valid")
    invalid_path = os.path.join(dataset_path, "Invalid")

    print("Processing valid sequences...")
    valid_sequences = read_all_sequences(valid_path, TARGET_LENGTH)
    print("\nProcessing invalid sequences...")
    invalid_sequences = read_all_sequences(invalid_path, TARGET_LENGTH)

    print("\nFinal shapes:")
    print("Valid sequences shape:", valid_sequences.shape)
    print("Invalid sequences shape:", invalid_sequences.shape)

    # Save preprocessed data
    print("\nSaving preprocessed data...")
    np.save('preprocessed_valid.npy', valid_sequences)
    np.save('preprocessed_invalid.npy', invalid_sequences)
    print("Done!")