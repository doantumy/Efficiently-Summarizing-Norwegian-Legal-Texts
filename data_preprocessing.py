import os
import argparse
from data_parsing.data_reader import DataReader

"""
Pre-processing raw json data and save into file for training/validation/test
"""


def parse_args():
    """Parse input arguments.

    Returns:
        ArgumentParser: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./data/processed_data", help="path to save processed data")
    parser.add_argument("--saved_name", type=str, default="processed_data_no_seg.json", help="output file name")
    parser.add_argument("--train_file", type=str, default="./data/train.json", help="path to train file")
    parser.add_argument("--val_file", type=str, default="./data/val.json", help="path to val file")
    parser.add_argument("--test_file", type=str, default="./data/test.json", help="path to test file")
    parser.add_argument("--do_segmentation", action="store_true", default=False, help="whether to do segmentation")
    parser.add_argument("--do_sentenization", action="store_true", default=False,
                        help="whether to do sentenization. Only work in non-segmentation mode.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    output_path = args.output_path
    saved_name = args.saved_name
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
    do_segmentation = args.do_segmentation
    do_sentenization = args.do_sentenization

    # Read files and do preprocessing.
    # If do_seegmentation is True, segment the data based on KAPITTEL, AVSNITT, paragraph
    # Otherwise, return a single long sequence of text for each document
    try:
        if do_segmentation and do_sentenization:
            raise ValueError(
                "Both do_segmentation and do_sentenization cannot be True at the same time.\n Sentenization only works in non-segmentation mode.")

        data_reader = DataReader(train_file=train_file,
                                 val_file=val_file,
                                 test_file=test_file,
                                 do_segmentation=do_segmentation,
                                 do_sentenization=do_sentenization)
        processed_data = data_reader.data_loader()
        # Save preprocessed data into files
        data_reader.save_to_files(input_data=processed_data, output_path=output_path, saved_name=saved_name)
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None


if __name__ == "__main__":
    main()

# Example running command: 
### With segmentation
# python3 data_preprocessing.py --do_segmentation --output_path ./data/processed_data --saved_name processed_data_w_segment.json --train_file ./data/lovdataTrainingV3woHTML.json --val_file ./data/lovdataValidationV3woHTML.json --test_file ./data/lovdataTestV3woHTML.json
### Without segmentation - withOUT sentenization
# python3 data_preprocessing.py --output_path ./data/processed_data --saved_name processed_data_wo_segment.json --train_file ./data/lovdataTrainingV3woHTML.json --val_file ./data/lovdataValidationV3woHTML.json --test_file ./data/lovdataTestV3woHTML.json

### Without segmentation - WITH sentenization
# python3 data_preprocessing.py --do_sentenization --output_path ./data/processed_data --saved_name processed_data_wo_segment_w_sentenization.json --train_file ./data/lovdataTrainingV3woHTML.json --val_file ./data/lovdataValidationV3woHTML.json --test_file ./data/lovdataTestV3woHTML.json

## TEST with sample.json
# python3 data_preprocessing.py --do_sentenization --output_path ./data/processed_data --saved_name processed_data_wo_segment.json --train_file ./data/sample1.json --val_file ./data/sample2.json --test_file ./data/sample3.json
# python3 data_preprocessing.py --do_segmentation --output_path ./data/processed_data --saved_name processed_data_w_segment.json --train_file ./data/sample1.json --val_file ./data/sample2.json --test_file ./data/sample3.json
