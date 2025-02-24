import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the directory where XML files are located
directory = "./data/"

# Create lists to store lengths of sammendrag and premiss text
sammendrag_lengths = []
premiss_lengths = []

for subdir in ['all_lovdata']:
    folder_path = os.path.join(directory, subdir)
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        for file in tqdm(files, desc=f"Processing {subdir}"):  # Use tqdm to display progress
            if file.endswith(".xml"):
                file_path = os.path.join(folder_path, file)
                tree = ET.parse(file_path)
                root = tree.getroot()

                sammendrag = root.find('Sammendrag')
                premiss = root.find('Premiss')

                if sammendrag is not None and sammendrag.text:
                    sammendrag_lengths.append(len(sammendrag.text))
                if premiss is not None and premiss.text:
                    premiss_lengths.append(len(premiss.text))

# Calculate mean and standard deviation for sammendrag and premiss
mean_sammendrag = np.mean(sammendrag_lengths)
std_sammendrag = np.std(sammendrag_lengths)

mean_premiss = np.mean(premiss_lengths)
std_premiss = np.std(premiss_lengths)

# Display the calculated values
print("Mean characters in Sammendrag:", mean_sammendrag)
print("Standard Deviation in Sammendrag:", std_sammendrag)
print("Mean characters in Premiss:", mean_premiss)
print("Standard Deviation in Premiss:", std_premiss)

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(sammendrag_lengths, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of Sammendrag Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
# Annotate statistics for sammendrag
plt.annotate(f'Mean: {mean_sammendrag:.2f}\nStd: {std_sammendrag:.2f}', xy=(0.7, 0.85), xycoords='axes fraction')

plt.subplot(1, 2, 2)
plt.hist(premiss_lengths, bins=100, color='green', alpha=0.7)
plt.title('Histogram of Premiss Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
# Annotate statistics for premiss
plt.annotate(f'Mean: {mean_premiss:.2f}\nStd: {std_premiss:.2f}', xy=(0.7, 0.85), xycoords='axes fraction')

plt.tight_layout()
# plt.show()

# Save the histograms
histogram_path = "./data/all_processed_data"
plt.savefig(histogram_path)

histogram_path
