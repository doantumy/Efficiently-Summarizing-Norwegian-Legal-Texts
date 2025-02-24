import os
import xml.etree.ElementTree as ET
import json
import argparse
import re
import spacy 
from tqdm import tqdm
import shutil
import random
from loguru import logger
import glob

def delete_xml_files(directory):
    """
    Deletes all XML files in the specified directory.

    Args:
        directory (str): Path to the directory where XML files are located.
    """
    # Construct the path pattern to match all XML files in the directory
    path_pattern = os.path.join(directory, '*.xml')
    
    # Use glob to find all files that match the pattern
    xml_files = glob.glob(path_pattern)
    
    # Loop through the list of xml file paths and remove each file
    for file_path in xml_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def create_folder(base_path, folder_name):
    """Create a folder if it doesn't exist and return the path."""
    path = os.path.join(base_path, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def split_data(source_folder, destination_folder, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Splits the data in the source folder into train, validation, and test sets.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    train_dir = create_folder(destination_folder, 'train')
    val_dir = create_folder(destination_folder, 'val')
    test_dir = create_folder(destination_folder, 'test')

    files = [f for f in os.listdir(source_folder) if f.endswith('.xml')]
    random.shuffle(files)

    train_end = int(len(files) * train_ratio)
    val_end = train_end + int(len(files) * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    def copy_files(files, directory):
        for file in files:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(directory, file)

            if not os.path.isfile(source_path):
                logger.info(f"File not found: {source_path}")
                continue

            try:
                shutil.move(source_path, destination_path)
                # logger.info(f"Moved {file} to {directory}")
            except Exception as e:
                logger.info(f"Failed to move {file}. Reason: {e}")

    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

    logger.info(f"Files distributed: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def sentenization(text, length_limit=5, spacy_model="nb_core_news_sm", max_length=3_000_000):
        """
        Performs sentence segmentation using SpaCy.

        Args:
            text (str): Document text.
            length_limit (int): Minimum number of characters per sentence.
            spacy_model (str): SpaCy model to use for sentence segmentation.
            max_length (int): Maximum length for SpaCy NLP model.
        Returns:
            List[str]: A list of sentences.
        """
        if not spacy_model:
            raise ValueError("SpaCy NLP model is required for sentenization.")

        nlp = spacy.load(spacy_model)
        nlp.max_length=max_length
        texts = text.split("\n")
        results = []
        for item in texts:
            item = re.sub(r'^\(\d+\)\s+', '', item).strip()
            if len(item.strip()) >= 200:
                doc = nlp(item)
                result = [sent.text for sent in doc.sents if len(sent.text) >= length_limit]
                if result:
                    results.extend(result)
            elif len(item.strip()) > 10: results.append(item)
        return results
        

def process_files(directory, args):
    """
    Process XML files in the given directory.

    Args:
        directory (str): Path to the directory containing XML files.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # for filename in os.listdir(directory):
    for filename in tqdm(os.listdir(directory), desc=f"Processing {directory}"):
        if filename.endswith(".xml"):
            tree = ET.parse(os.path.join(directory, filename))
            root = tree.getroot()

            sammendrag = root.find('Sammendrag')
            premiss = root.find('Premiss')

            if sammendrag is not None and premiss is not None and len(premiss.text) >= 3_000: # text must be at most 3k characters
                summary = sammendrag.text.strip()
                premiss_text = premiss.text.strip()
                result = {}
                if args.do_combined:
                    # Combine text into a long string and add it directly to the text list
                    combined_text = ' '.join(premiss_text.strip().split())
                    yield {
                        "summary": summary,
                        "filename": filename,
                        "text": [combined_text]
                    }
                
                if args.do_sentenization:
                    # Split text into sentences and add them directly to the text list
                    sentences = sentenization(premiss_text)
                    yield {
                        "summary": summary,
                        "filename": filename,
                        "text": sentences
                    }
                
                if args.do_segmentation:
                    # Split the text on empty lines and clean newlines within segments
                    chapters = re.split(r'\n\s*\n', premiss_text.strip())
                    chapter_dict = {}
                    for i, chapter in enumerate(chapters):
                        lines = [
                            re.sub(r'^\(\d+\)\s+', '', line.strip().replace('***', ''))  # Remove specific patterns and trim
                            for line in chapter.split('\n')
                            # if line.strip() and re.sub(r'^\(\d+\)\s+', '', line.strip().replace('***', '')).strip()  # Check for non-empty after cleaning
                        ]
                        if lines:  # Only add chapters that have non-empty lines
                            key = f"KAPITTEL_0_{i}"
                            chapter_dict[key] = lines
                    if chapter_dict:
                        result["KAPITTEL_0"] = chapter_dict

                    yield {
                        "summary": summary,
                        "filename": filename,
                        "text": result
                    }

def main(args):
    directories = ['train', 'val', 'test']
    base_path = args.base_dir
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w') as jsonfile:
        jsonfile.write('{\n')
        for i, dir_name in enumerate(directories):
            full_path = os.path.join(base_path, dir_name)
            jsonfile.write(f'  "{dir_name}": [\n')
            first = True
            for entry in process_files(full_path, args):
                if not first:
                    jsonfile.write(',\n')
                json.dump(entry, jsonfile, indent=4, ensure_ascii=False)
                first = False
            if i < len(directories) - 1:
                jsonfile.write('\n  ],\n')
            else:
                jsonfile.write('\n  ]\n')
        jsonfile.write('}')

if __name__ == "__main__":
    # Randomly move file to train/val/test directory
    # source_folder = './data/all_data'
    # destination_folder = './data/'
    # split_data(source_folder, destination_folder)

    # Processing xml files
    parser = argparse.ArgumentParser(description='Process XML files and output JSON with specific text formats.')
    parser.add_argument('--output_file', type=str, default=None, help='Filename of output JSON file')
    parser.add_argument('--base_dir', type=str, default='data', help='Base directory where train, val, and test folders are located')
    parser.add_argument('--do_combined', action='store_true', help='Include combined text in the output.')
    parser.add_argument('--do_sentenization', action='store_true', help='Include sentence tokenization in the output.')
    parser.add_argument('--do_segmentation', action='store_true', help='Include text segmentation by chapters in the output.')
    args = parser.parse_args()
    main(args)

# SUMM^N python xml_data_preprocessing.py --do_sentenization --output_file ./data/all_processed_data/all_data_wo_segment_w_sent.json
# para python xml_data_preprocessing.py --output_file ./data/all_processed_data/all_data_w_segment_w_sent.json --do_segmentation
# classic python xml_data_preprocessing.py --do_combined --output_file ./data/all_processed_data/all_data_wo_seg_wo_sent.json