from functools import partial

from models.data_segment.source_segment import SourceSegmentor
from models.data_segment.multi_structure_source_segment import MultiStructureSourceSegmentor
from models.data_segment.source_similarity_segment import SourceSimilaritySegmentor
from models.data_segment.target_matching import TargetSegmentor

from models.gen_summary.inference import SummaryGenerator
from models.gen_summary.coarse_seg import CoarseSegCombiner

from models.unify.checks import NoStructureCheck, StructureCheck

RougeSourceSimilaritySegmentor = partial(SourceSimilaritySegmentor, metric='rouge1')
BertSourceSimilaritySegmentor = partial(SourceSimilaritySegmentor, metric='bertscore')
