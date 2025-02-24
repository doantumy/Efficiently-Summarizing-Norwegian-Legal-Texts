import json
import random
from tqdm import tqdm
import spacy
from loguru import logger
import pickle
import os

def extract_identifiers_from_filenames(similarity_matrices_dir):
    """
    Extracts reference identifiers from the filenames in a directory containing similarity matrices.

    Args:
        similarity_matrices_dir (str): Path to the directory containing similarity matrix files.

    Returns:
        set: A set of extracted reference identifiers from the filenames.

    Assumptions:
        - Filenames are structured like 'batch_{batch_idx}_doc_{idx}_ref_{ref}.pkl'.
        - The reference identifier is extracted from the portion of the filename after 'ref_' and before '.pkl'.
    
    Example:
        If the filename is 'batch_0_doc_0_ref_tmha-2021-26652-1.pkl', the extracted identifier will be 'tmha-2021-26652-1'.
    """
    sim_files = os.listdir(similarity_matrices_dir)
    identifiers = []
    for filename in sim_files:
        parts = filename.split('_')
        ref_part = parts[-1]  # Assuming 'ref_{ref}.pkl'
        ref = ref_part.replace('ref_', '').replace('.pkl', '')
        identifiers.append(f"{ref}")
    return set(identifiers)


def load_similarity_matrix(file_path):
    """
    Loads a similarity matrix from a pickle file.

    Args:
        file_path (str): Path to the pickle file containing the similarity matrix.

    Returns:
        np.ndarray: The loaded similarity matrix.

    Example:
        The function reads a .pkl file containing a NumPy array and returns it for further processing.
    """
    with open(file_path, "rb") as f:
        sim_matrix = pickle.load(f)
    return sim_matrix


def load_and_sentenize_texts_in_batches(
    data_file,
    spacy_model,
    identifiers,
    batch_size=64,
    min_text_length=50,
    summary_length_limit=10,
    text_length_limit=50,
    sample_fraction=1.0, 
    split="train", 
    lang="no",
    ref_name="reference"
):
    """
    Loads text and summary data from a JSON file, processes it in batches, tokenizes the texts and summaries into sentences using SpaCy,
    and yields batches of sentences and their corresponding references for items whose
    references are in the provided identifiers set.

    Args:
        data_file (str): Path to the JSON file containing the data.
        spacy_model (spacy.Language): Loaded SpaCy language model for sentence tokenization.
        identifiers (set): A set of reference identifiers to filter the data.
        batch_size (int, optional): Number of records per batch. Defaults to 64.
        min_text_length (int, optional): Minimum words for texts to be included. Defaults to 50.
        summary_length_limit (int, optional): Minimum number of characters per summary sentence. Defaults to 10.
        text_length_limit (int, optional): Minimum number of characters per text sentence. Defaults to 50.
        sample_fraction (float, optional): Fraction of data to randomly sample (default: 1.0, i.e., use all data).
        split (str, optional): Split of data to load ("train", "dev", or "test"). Defaults to "train".
        ref_name (str): Reference field name. Default to "reference"

    Yields:
        Tuple[List[List[str]], List[List[str]], List[str]]: Batches of summaries, text sentences, and their references.
            - summaries_batch: List of lists of sentences for each summary in the batch.
            - texts_batch: List of lists of sentences for each text in the batch.
            - refs_batch: List of reference identifiers corresponding to the texts.
    """
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = data.get(split, [])

    # Randomly sample a fraction of the data if specified
    if sample_fraction < 1.0:
        num_samples = int(len(train_data) * sample_fraction)
        train_data = random.sample(train_data, num_samples)

    # Process data in batches, filtering on-the-fly
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        valid_summaries = []
        valid_texts = []
        valid_refs = []

        for item in batch:
            full_ref = item.get(ref_name, "")
            short_ref = full_ref.split("/")[-1]
            if short_ref not in identifiers:
                continue  # Skip items without valid references

            summary = item.get("summary", "")
            if lang == "au":
                text = item.get("text", "") if item.get("text") else ""
                # logger.info(f"text {(text.split())}")
            else:
                text = item.get("text", [""])[0] if item.get("text") else ""


            # Check for min_text_length
            if (
                summary and len(summary.split()) >= min_text_length and
                text and len(text.split()) >= min_text_length * 3
            ):
                valid_summaries.append(summary)
                valid_texts.append(text)
                valid_refs.append(full_ref)

        # Tokenize summaries and texts into sentences
        if valid_texts:
            summaries_batch, texts_batch, refs_batch = process_summaries_and_texts_with_spacy(
                summaries=valid_summaries,
                texts=valid_texts,
                refs=valid_refs,
                spacy_model=spacy_model,
                summary_length_limit=summary_length_limit,
                text_length_limit=text_length_limit
            )
            yield summaries_batch, texts_batch, refs_batch
        # Clear variables to save memory
        del batch, valid_summaries, valid_texts, valid_refs

def sentenization(text, length_limit=50, spacy_model="nb_core_news_sm", max_length=3_000_000):
    """Sentence segmentation using Spacy

    Args:
        text (str): Document text
        length_limit (int): Min number of characters per sentence
        spacy_model (str): Spacy model to use for sentence segmentation. Defaults to "nb_core_news_sm".

    Returns:
        List[str]: A list of sentences
    """
    if not spacy_model:
        raise ValueError("Spacy NLP model is required for sentenization.")

    nlp = spacy.load(spacy_model)
    nlp.max_length=max_length
    doc = nlp(text)
    return [sent.text for sent in doc.sents if
            len(sent.text) >= length_limit]  # Only keep sentences >= length_limit characters long

def process_summaries_and_texts_with_spacy(summaries, texts, refs, spacy_model, summary_length_limit=10, text_length_limit=50):
    """
    Processes a batch of summaries and texts using the sentenization function for sentence tokenization.

    Args:
        summaries (List[str]): List of summary documents to process.
        texts (List[str]): List of text documents to process.
        refs (List[str]): Corresponding list of reference identifiers.
        spacy_model (spacy.Language): Loaded SpaCy language model.
        summary_length_limit (int): Minimum number of characters per summary sentence.
        text_length_limit (int): Minimum number of characters per text sentence.

    Returns:
        Tuple[List[List[str]], List[List[str]], List[str]]: Processed summaries, text sentences, and corresponding references.
            - processed_summaries: List of lists of sentences for each processed summary.
            - processed_texts: List of lists of sentences for each processed text.
            - processed_refs: List of references corresponding to the processed summaries and texts.
    """
    processed_summaries = []
    processed_texts = []
    processed_refs = []

    for summary, text, ref in zip(summaries, texts, refs):
        # Check if both summary and text meet minimum length criteria
        if (
            summary and len(summary.split()) >= summary_length_limit and
            text and len(text.split()) >= text_length_limit
        ):
            # Tokenize using the sentenization function
            summary_sents = sentenization(summary, length_limit=summary_length_limit, spacy_model=spacy_model)
            text_sents = sentenization(text, length_limit=text_length_limit, spacy_model=spacy_model)
            
            if summary_sents and text_sents:
                processed_summaries.append(summary_sents)
                processed_texts.append(text_sents)
                processed_refs.append(ref)
            else:
                logger.warning(f"Empty sentenization for id {ref}")
        else:
            logger.warning(f"Text or summary too short for id {ref}")

    return processed_summaries, processed_texts, processed_refs

def build_dataset_for_batch(similarity_matrices_dir, summaries_batch, texts_batch, refs_batch):
    """
    Builds a dataset for a batch of summaries, texts, and references by extracting the maximum similarity score
    for each summary-text pair and linking them.

    Args:
        similarity_matrices_dir (str): Path to the directory containing similarity matrix files.
        summaries_batch (list of list of str): List of summaries, where each summary is a list of sentences.
        texts_batch (list of list of str): List of texts, where each text is a list of sentences.
        refs_batch (list of str): List of reference identifiers corresponding to the summaries and texts.

    Returns:
        list of dict: A list where each element is a dictionary containing:
            - 'summary': The summary sentence.
            - 'text': The text sentence.
            - 'label': The maximum similarity score for that sentence pair.
            - 'ref': The reference identifier.
    """
    dataset = []
    for summaries, texts, ref in zip(summaries_batch, texts_batch, refs_batch):
        # Construct the expected filename pattern
        short_ref = ref.split("/")[-1]
        sim_files = [f for f in os.listdir(similarity_matrices_dir) if f"ref_{short_ref}.pkl" in f]
        if not sim_files:
            logger.info(f"No similarity matrix file found for ref {ref}")
            continue

        sim_file = sim_files[0]  # Assuming one file per ref
        sim_matrix_path = os.path.join(similarity_matrices_dir, sim_file)
        sim_matrix = load_similarity_matrix(sim_matrix_path)

        # Ensure the matrix dimensions match the number of sentences
        if sim_matrix.shape[0] != len(summaries) or sim_matrix.shape[1] != len(texts):
            logger.info(f"Mismatch in matrix dimensions for ref {short_ref}")
            logger.info(f"sim_matrix.shape[0] {sim_matrix.shape[0]} len(summaries) {len(summaries)}")
            logger.info(f"sim_matrix.shape[1] {sim_matrix.shape[1]} len(texts) {len(texts)}")
            continue

        # Transpose the similarity matrix to iterate over columns (text sentences)
        sim_matrix_T = sim_matrix.T
        for text_idx, sim_scores in enumerate(sim_matrix_T):
            max_score = sim_scores.max()
            best_summary_idx = sim_scores.argmax()
            text_sentence = texts[text_idx]
            summary_sentence = summaries[best_summary_idx]

            dataset.append({
                'summary': summary_sentence,
                'text': text_sentence,
                'label': max_score,
                'ref': ref
            })
        # Clear similarity matrix to save memory
        del sim_matrix
    return dataset


def save_batch_to_file(dataset, output_file):
    """
    Saves a batch of the dataset to a JSON file. Appends the batch as a list to the file.

    Args:
        dataset (list of dict): The dataset batch to save. Each element should be serializable to JSON.
        output_file (str): Path to the output JSON file.
    """
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')  # Write each record as a line of JSON


if __name__ == "__main__":
    # data_file = "data/processed_data/test.json"
    # Australian LovData
    # spacy_model_name = "en_core_web_sm" # python -m spacy download en_core_web_sm
    # lang="au"
    # data_file = "data/au_data_processed/au_processed_data_wo_segment.json"
    # # data_file = "data/processed_data/au_test_wo_seg.json"
    # output_dir = "data/au_data_processed/similarity_data/"
    
    # Norwegian LovData
    # data_file = "data/processed_data/processed_data_wo_segment.json"
    # spacy_model_name = "nb_core_news_sm"
    # lang="no"
    # output_dir = "data/processed_data/similarity_data/"
    
    # Norwegian ALL LovData
    data_file = "data/all_processed_data/all_data_wo_seg_wo_sent.json"
    spacy_model_name = "nb_core_news_sm"
    lang="no_all"
    ref_name = "filename"
    output_dir = "data/all_processed_data/"

    batch_size = 64
    splits = ["test", "val", "train"]
    matrix_dir = "similarity/output/"
    for split in splits:
        logger.info(f"Processing {split} data")
        output_file = f"{output_dir}{lang}_similarity_{split}_dataset.json"
        logger.info(f"Output file: {output_file}")
        similarity_matrices_dir = f"{matrix_dir}{split}_{lang}_sim_matrix/"
        # Check if the directory contains files
        if os.path.exists(similarity_matrices_dir) and any(os.scandir(similarity_matrices_dir)):
            # Extract identifiers from filenames of existing similarity matrices
            identifiers = extract_identifiers_from_filenames(similarity_matrices_dir)

            # Initialize tqdm progress bar for batch processing
            total_batches = len(identifiers) // batch_size + 1  # Estimate total batches
            progress_bar = tqdm(
                total=total_batches, 
                desc="Processing batches", 
                unit="batch", 
                leave=True
            )

            for summaries_batch, texts_batch, refs_batch in load_and_sentenize_texts_in_batches(
                    data_file,
                    spacy_model=spacy_model_name,
                    identifiers=identifiers,
                    batch_size=batch_size,
                    min_text_length=50,
                    summary_length_limit=10,
                    text_length_limit=50,
                    sample_fraction=1.,
                    split=split,
                    lang=lang,
                    ref_name=ref_name
                ):
                # logger.info(f"Processing batch with {(refs_batch)} references")
                
                # Build dataset for the batch
                dataset = build_dataset_for_batch(similarity_matrices_dir, summaries_batch, texts_batch, refs_batch)

                # Save the batch to file
                save_batch_to_file(dataset, output_file)

                # Update the progress bar
                progress_bar.update(1)

            # Close the progress bar when done
            progress_bar.close()
        else:
            # Directory is empty or does not exist
            logger.warning(f"No files found in {similarity_matrices_dir}.")

        

