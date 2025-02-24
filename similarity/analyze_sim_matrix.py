import pickle
import numpy as np
from loguru import logger
import os
from matplotlib import pyplot as plt

def load(file_path="file_name.pkl"):
    """
    Load data from a pickle file.

    Args:
        file_path (str): Path to the pickle file to load data from.

    Returns:
        list[torch.Tensor]: List of loaded data.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    # logger.info(f"File loaded from {file_path}")
    return data


def load_folder(folder_path):
    """
    Load all pickle files in a folder.

    Args:
        folder_path (str): Path to the folder containing pickle files.

    Returns:
        dict: A dictionary with file names as keys and loaded data as values.
    """
    loaded_data = {}
    logger.info(folder_path)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(folder_path, file_name)
            try:
                loaded_data[file_name] = load(file_path)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
    return loaded_data


def analyze_similarity_matrices(data_dict):
    """
    Analyze similarity matrices in the given dictionary.

    Args:
        data_dict (dict): A dictionary with file names as keys and similarity matrices as values.

    Returns:

    """
    min_values = []
    for file_name, matrix in data_dict.items():
        try:
            shape = matrix.shape
            sum_length = shape[0]
            if not isinstance(matrix, np.ndarray):
                raise ValueError("Input must be a NumPy array.")

            min_value = np.min(np.sort(matrix.flatten())[-sum_length:])
            if min_value:
                logger.info(f"Top min_value {min_value}")
                min_values.append(min_value)
            else:
                logger.warning(f"No valid matrix found. {file_name}")
                break

        except AttributeError:
            logger.info(f"File: {file_name} does not contain a valid matrix.")
    
    # Compute statistics if min_values is not empty
    if min_values:
        mean_val = np.mean(min_values)
        std_val = np.std(min_values)
        median_val = np.median(min_values)

        logger.info(f"Mean: {mean_val}, Std: {std_val}, Median: {median_val}")
        # Plot histogram
        plt.figure(figsize=(10, 6))
        counts, bins, patches = plt.hist(min_values, bins=10, alpha=0.7, color='blue', edgecolor='black', label="Histogram")
        
        # Plot a line connecting bar tops
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate bin centers
        plt.plot(bin_centers, counts, color='red', marker='o', linestyle='-', label="Line overlay")

        # Add mean and std annotations
        plt.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.2f}")
        plt.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f"Mean + Std: {mean_val + std_val:.2f}")
        plt.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1, label=f"Mean - Std: {mean_val - std_val:.2f}")

        # Add labels and title
        plt.title("Histogram of Minimum Values with Line Overlay", fontsize=14)
        plt.xlabel("Minimum Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig("min_histogram.png", dpi=300, bbox_inches="tight")
        plt.show()

        return {
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
        }
    else:
        logger.warning("No valid matrices were processed.")
        return {
            "mean": None,
            "std": None,
            "median": None,
        }    


if __name__ == "__main__":
    # analyze_similarity_matrices(data_dict=load_folder("similarity/output/no_sim_matrix/")) # Norwegian data
    analyze_similarity_matrices(data_dict=load_folder("similarity/output/train_no_all_sim_matrix/")) # Norwegian ALL Lovdata
    # analyze_similarity_matrices(data_dict=load_folder("similarity/output/train_au_sim_matrix/")) # Australian data