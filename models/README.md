# Data segmentation
## Baseline cases
### Case 1: Classic fine-tuning
**Note:** 
- This will be implemented in the end.
- No structural data segmentation, each example is a single sequence of text. Cut the sequence to fit with model sequence length.
- No Target matching procedure. One text sequence will have one target sumary.
- `Input:` None structure data
- `Output:` [{`truncated_train_example_1_text`, `truncated_summary_1`}, {`truncated_train_example_2_text`, `truncated_summary_2`}, ...]

### Case 2: SUMM$^N$ method
- Sentence-level segment splitting as long as the segment fits into model sequence length.
- Perform target matching procedure. One segment will have one target sumary. Multiple segments from the same training example will have the same target summary.
- `Input:` None structure data
- `Output:` [{`segment_1.1_text`,`summary_1`}, {`segment_1.2_text`,`summary_1`},{`segment_2.1_text`,`summary_2`}, {`segment_2.2_text`,`summary_2`}, ...]

```bash
# filename: segmentor_core.py
class SourceSplitterCore # handle processing the segmentation

# source_segment.py
class SourceSegmentor # handle saving processed output into files -> used for training/evaluation
```


## Enhanced-features cases
### Case 1: Paragraph-level segment
- Performs paragraph-level segment splitting as long as the segment fits into model sequence length
- Perform target matching procedure. One segment will have one target summary. Multiple segments from the same training example
- `Input:` structured-parsed documents with respected target summaries
- `Output:` [{`paragraph_1.1_text`,`summary_1`}, {`paragraph_1.2_text`,`summary_1`, ...}]

```bash
# filename: segmentor_core.py
class SourceParagraphSplitter 

# filename: multi_structure_source_segment.py
class MultiStructureSourceSegmentor 
```

### Case 2: Paragraph-level similarity score segment using `ROUGE-1` or `BERTScore`
- Group similar sentences in the same paragraph together as long as the segment fits into model sequence length
- Perform target matching procedure. One segment will have one target summary. Multiple segments from the same training example
- `Input:` structured-parsed documents with respected target summaries
- `Output:` [{`paragraph_1_sentence_1_to_3_text`, `summary_1`}, {`paragraph_1_sentence_4_to_5_text`, `summary_1`}, {`paragraph_2_sentence_1_to_5`, `summary_2`}, ...]

```bash
# filename: segmentor_core.py
class SourceSimilarityScoreSplitter 

# filename: source_similarity_segment.py
class SourceSimilaritySegmentor
```