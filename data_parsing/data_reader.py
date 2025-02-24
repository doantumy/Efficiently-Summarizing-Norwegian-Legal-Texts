from data_parsing.data_parsing import parse_general_documents
from typing import Dict, List, Union
from tqdm import tqdm
import os
import json


class DataReader(object):
    def __init__(self, train_file: Dict[str, str] = None, 
                val_file: Dict[str, str] = None, 
                test_file: Dict[str, str] = None, 
                do_segmentation: bool = True,
                do_sentenization: bool = False) -> None:
        """Read training, validation and test data, combine them into one dictionary.
        The keys of the dictionary are the names of the splits, and the values are lists of dictionaries.

        Args:
            train_file (Dict[str, str], optional): Path to the training data file in JSON format. Defaults to None.
            val_data (Dict[str, str], optional): Path to the validation data file in JSON format. Defaults to None.
            test_data (Dict[str, str], optional): Path to the test data file in JSON format. Defaults to None.
            do_segmentation (bool, optional): Flag indicating whether to perform segmentation on the data. Defaults to True.
            do_sentenization (bool, optional): Flag indicating whether to perform sentenization on the data. 
            Only applied when do_sentenization=False. Defaults to False.
        """

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.do_segmentation = do_segmentation
        self.do_sentenization = do_sentenization


    def data_loader(self) -> Dict[str, List[Dict[str, Union[str, List[str], Dict[str, Dict[str, List[str]]]]]]]:
        """Read training and validation data, combine them into one dictionary.
        The keys of the dictionary are the names of the splits, and the values are lists of dictionaries.

        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary containing the training and validation data. The dictionary has the following structure:
                {
                    'train': [ 
                        {'reference': 'link to lovdata document', 'text': [list of data], ...}, 
                        ...
                    ],
                    'val': [ 
                        {'reference': 'link to lovdata document', 'text': {json data}, ...}, 
                        ...
                    ]
                }
                JSON data can have format as below:
                {
                    "KAPITTEL_0": {
                        "KAPITTEL_0-0": [
                            "Oslo statsadvokatembeter har ved tiltalebeslutning 10. august 2023 satt A, født [00.00.1988], under tiltale for overtredelse av:",
                            "Vegtrafikkloven § 31 første ledd, jf. § 22 første leddfor å ha ført en motorvogn når han var påvirket av berusende eller bedøvende midler.",
                            "Forsvarer la ned påstand om at tiltalte frifinnes, subsidiært anses på mildeste måte."
                        ]
                    },
                    "KAPITTEL_1": {
                        "KAPITTEL_1-0": [
                            "Lagmannsretten bemerker:"
                        ],
                        "KAPITTEL_1-1": [
                            "1. Oppsummering",
                            "Lagmannsretten har under dissens kommet til at anken over bevisbedømmelsen under skyldspørsmålet ikke fører fram, og at A skal frifinnes."
                        ],
                        "KAPITTEL_1-2": [
                            "2. Skyldspørsmålet",
                            "2.1 InnledningTiltalen gjelder føring av motorvogn i påvirket tilstand.Det er ikke omtvistet at ti.",
                        ]
                    }
                }
            Each entry in the 'train' and 'val' lists is a dictionary representing a data sample, which can include various fields such as 'id', 'text', 'label', etc.
        """
        
        
        def parse_data(ref_link: str, split_name:str, log_path: str) -> Union[Dict, List[str]]:
            """Parse document from the given URL.

            Args:
                ref_link (str): URL to fetch the document from.
                split_name (str): Name of the split (train, val, or test).
                log_path (str): path to save the error log file.

            Returns:
                Union[Dict, List[str]]: A dictionary or a list of strings representing the parsed document.
            """
            try:
                url = f'https://lovdata.no{ref_link}'
                parsed_doc = parse_general_documents(url = url, 
                                                     do_segmentation = self.do_segmentation,
                                                     do_sentenization = self.do_sentenization)    
                # If the document is not available, the function returns "nothing" text.
                # if isinstance(parsed_doc, (list, dict)) or (isinstance(parsed_doc, str) and parsed_doc.strip() != "nothing"):
                if (isinstance(parsed_doc, list)) or \
                    (isinstance(parsed_doc, dict)) or \
                    (isinstance(parsed_doc, str) and parsed_doc.strip() != "nothing"):
                    return parsed_doc
                else: return None
            except AttributeError as ae:
                # Write the error to a file
                with open(f"./{log_path}/error_log_{split_name}.txt", "a") as error_file:
                    error_file.write(url + "\n")
                return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None


        def load_multiple_json(file_path: str) -> List[Dict[str, str]]:
            """Load multiple JSON files into a single list of dictionaries.

            Args:
                file_path (str): Path to the JSON file.
            
                Returns:
                    List[Dict[str, str]]: A list of dictionaries representing the parsed data.
            """
            split_name = file_path.split("/")[-1].split(".")[0]
            # Extract the directory path from file_path
            log_path = os.path.dirname(file_path)

            with open(file_path, "r", encoding="utf-8") as file:
                data = []
                # Get the total number of lines in the file for tqdm
                total_lines = sum(1 for _ in file)
                # Reset file pointer to the beginning
                file.seek(0)
                # Read each line and parse as JSON
                for _, line in tqdm(enumerate(file, start=1), total=total_lines, desc="Loading JSON files...", unit=" lines", dynamic_ncols=True, ncols=80):
                    dict_data = {}
                    line_data = json.loads(line)
                    parsed_doc = parse_data(ref_link = line_data["reference"], split_name = split_name, log_path = log_path)
                    if parsed_doc is not None:
                        dict_data["summary"] = line_data["summary"]
                        dict_data["reference"] = line_data["reference"]
                        dict_data["text"] = parsed_doc
                        data.append(dict_data)
                    else: continue
            return data

        print("Preprocessing training data...")
        train_data = load_multiple_json(self.train_file)
        
        print("Preprocessing validation data...")
        validation_data = load_multiple_json(self.val_file)
        
        print("Preprocessing testing data...")
        test_data = load_multiple_json(self.test_file)
        
        data = {
            "train": train_data,
            "val": validation_data,
            "test": test_data
        }

        return data
    

    def save_to_files(self, input_data, output_path: str, saved_name: str) -> None:
        """Save the data to a JSON file.

        Args:
            input_data (Dict[str, List[Dict[str, Union[str, List[str], Dict[str, Dict[str, List[str]]]]]]): 
                A dictionary containing the data to be saved.
            output_path (str): Path to the output directory.

        """
        file_name = os.path.join(output_path, saved_name)
        # Extract the directory path from file_name
        dir_path = os.path.dirname(file_name)

        # Check if the directory exists, and if not, create it
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(input_data, json_file, ensure_ascii=False, indent=4)