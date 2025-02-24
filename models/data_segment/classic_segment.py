import os
from tqdm import tqdm
import json
import argparse
from loguru import logger

def write_list_to_file(save_dir, data):
    """Write a list of lines to file to match with the training data
        train.source, train.target, and so on. One example per line

    Args:
        save_dir (str): path to write
        data (list): list of data
    """    
    with open(save_dir, 'w', encoding='utf-8') as file:
        for sample in data:
            file.write(sample.strip()+"\n")


class ClassicSegmentor(object):
    """Process rawdata into format that can be read during training
    """
    
    def __init__(self, data_file: str=None, output_dir: str=None, combined_text: bool=False) -> None:
        self.data_file = data_file
        self.output_dir = output_dir
        self.combined_text = combined_text
    

    def read_json(self):
        """Read input json file

        Returns:
            load_data(Dict[str, List[Dict[str, str]]]): returned data has same format as input
        """        
        with open(self.data_file, "r") as file:
            loaded_data = json.load(file)
        return loaded_data
    

    def segment(self, raw_data):
        """Segment raw data into "train, "val" and "test"

        Args:
            raw_data (Dict[str, List[Dict[str, List[str]]]]): raw data (same as input data format)
                `text` field is a list. Check if it contains multiple sentences or just one long string. 
                If multiple sentences: Toggle combine_text = True
                Else set combine_text = False

        Returns:
            processed_data: return processed data
        """        
        processed_data = {"train": {"source": [], "target": []}, 
                          "val": {"source": [], "target": []}, 
                          "test": {"source": [], "target": []}}
    
        for split in processed_data.keys():
            if split in raw_data:
                if self.combined_text:
                    for item in tqdm(raw_data[split], desc=f"Processing {split} data"):
                        summary = item.get("summary", "")
                        lst_text = item.get("text", "") # text has list format
                        text = " ".join(lst_text)
                        if summary.strip() != "" and text.strip() != "":
                            processed_data[split]["source"].append(text)
                            processed_data[split]["target"].append(summary)
                        else: 
                            logger.info(f"Warning: Empty summary or text in item: {item}")
                else:
                    for item in tqdm(raw_data[split], desc=f"Processing {split} data"):
                        summary = item.get("summary", "")
                        text = item.get("text", "")[0] # text has list format with only one string
                        if summary != "" and text != "":
                            processed_data[split]["source"].append(text)
                            processed_data[split]["target"].append(summary)
                        else: 
                            logger.info(f"Warning: Empty summary or text in item: {item}")
            else:
                logger.info(f"Warning: {split} not found in data")
        
        return processed_data


    def save(self, processed_data):
        """Save data into train.source, train.target, and so on.

        Args:
            processed_data (Dict[str, Dict[str, Lisy]]): processed data
        """        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        for split in processed_data.keys():
            source_file = f"{self.output_dir}/{split}.source"
            target_file = f"{self.output_dir}/{split}.target"
            
            with tqdm(total=len(processed_data[split]["source"]), desc=f"Saving {split} data") as pbar:
                write_list_to_file(save_dir = source_file, data = processed_data[split]["source"])
                pbar.update(len(processed_data[split]["source"]))
            
            with tqdm(total=len(processed_data[split]["target"]), desc=f"Saving {split} data") as pbar:
                write_list_to_file(save_dir = target_file, data = processed_data[split]["target"])
                pbar.update(len(processed_data[split]["target"]))

    def process_data(self):
        processed_data = self.segment(self.read_json())
        self.save(processed_data=processed_data)
        logger.info(f"Processing is done. Files saved.")

def main(args):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths to go up one level
    data_file = os.path.abspath(os.path.join(script_dir, '..', args.data_file))
    output_dir = os.path.abspath(os.path.join(script_dir, '..', args.output_dir))
    
    cs = ClassicSegmentor(data_file=data_file, output_dir=output_dir, combined_text=True)
    cs. process_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="../data/processed_data/processed_data_wo_segment.json", help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, default="../data/processed_data", help="Path to the output directory")
    args = parser.parse_args()
    main(args)

# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_threshold_0.58_reformat_false_order_true.json" --output_dir=../data/processed_data/prunned_data/classic_threshold_0.58_reformat_false_order_true
# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_proportion_0.5_reformat_false_order_true.json" --output_dir=../data/processed_data/prunned_data/classic_proportion_0.5_reformat_false_order_true
# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_threshold_0.65_reformat_false_order_true.json" --output_dir=../data/processed_data/prunned_data/classic_threshold_0.65_reformat_false_order_true
# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_proportion_0.3_reformat_false_order_true.json" --output_dir=../data/processed_data/prunned_data/classic_proportion_0.3_reformat_false_order_true
# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_threshold_0.72_reformat_false_order_true.json" --output_dir=../data/processed_data/prunned_data/classic_threshold_0.72_reformat_false_order_true
# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_threshold_local_reformat_false_order_true.json" --output_dir=../data/processed_data/prunned_data/classic_threshold_local_reformat_false_order_true
# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_proportion_0.7_reformat_false_order_true.json" --output_dir=../data/processed_data/prunned_data/classic_proportion_0.7_reformat_false_order_true
# Shuffle
# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_threshold_local_reformat_false_order_false.json" --output_dir=../data/processed_data/prunned_data/classic_threshold_local_reformat_false_order_false
# python3 classic_segment.py --data_file="../data/processed_data/prunned_data/all_splits_proportion_0.5_reformat_false_order_false.json" --output_dir=../data/processed_data/prunned_data/classic_proportion_0.5_reformat_false_order_false

# ALL LOVDATA
# python3 classic_segment.py --data_file="../data/all_processed_data/all_data_wo_seg_wo_sent.json" --output_dir=../data/all_processed_data/classic_all_no_data
