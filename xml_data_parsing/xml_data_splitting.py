import os
import xml.etree.ElementTree as ET
import shutil
import random

# Specify the directory containing the XML files
directory = "./data/all_lovdata"
output_directory = "./data"

# Define thresholds (number of characters required)
threshold_sammendrag = 200 # summary length
threshold_premiss = 12_000 # text length

# Prepare the list for filenames
qualified_entries = []

# Read all XML files in the directory
for file in os.listdir(directory):
    if file.endswith(".xml"):
        file_path = os.path.join(directory, file)
        tree = ET.parse(file_path)
        root = tree.getroot()

        sammendrag = root.find('Sammendrag')
        premiss = root.find('Premiss')

        if sammendrag is not None and premiss is not None and sammendrag.text and premiss.text:
            len_sammendrag = len(sammendrag.text)
            len_premiss = len(premiss.text)

            if len_sammendrag >= threshold_sammendrag and len_premiss >= threshold_premiss:
                qualified_entries.append(file)

# Set a random seed for reproducibility
random.seed(123)

# Shuffle the list of qualified entries
random.shuffle(qualified_entries)

# Calculate the number of files in each set
total_entries = len(qualified_entries)
print(f"Total entries after preprocessing: {total_entries}")
num_train = int(0.7 * total_entries)
num_val = int(0.1 * total_entries)
num_test = total_entries - num_train - num_val

# Split the entries based on calculated sizes
train_files = qualified_entries[:num_train]
val_files = qualified_entries[num_train:num_train + num_val]
test_files = qualified_entries[num_train + num_val:]

# Ensure the output directories exist
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_directory, folder), exist_ok=True)

# Copy files to their respective directories
for file_list, folder in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
    for file in file_list:
        shutil.copy(os.path.join(directory, file), os.path.join(output_directory, folder, file))

# Output the results
print(f"Files have been copied: {len(train_files)} training files, {len(val_files)} validation files, {len(test_files)} test files.")
