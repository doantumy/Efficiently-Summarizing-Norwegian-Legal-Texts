from loguru import logger
from tqdm import tqdm


class StructureCheck:
    def __init__(self):
        pass

    def __call__(self, source, target):
        try:
            first_sample = source['train'][0]
        except IndexError:
            raise ValueError("Data not supported, missing level of data structure")
        if isinstance(first_sample, list):
            raise ValueError("Data not compatible with experiment, check 'train_config.cfg' and data source."
                             "Expecting intput to be of structure (list check failed)")
        if isinstance(first_sample, dict):
            logger.info("Structured input data found, CHECK: OK")
            return source, target
        raise ValueError("Data not compatible with experiment, check 'train_config.cfg' and data source."
                         "Expecting intput to be of structure (dict check failed)")


class NoStructureCheck:
    def __init__(self):
        pass

    def __call__(self, source, target):
        try:
            first_sample = source['train'][0]
        except IndexError:
            raise ValueError("Data not supported, missing level of data structure")
        if not isinstance(first_sample, list):
            raise ValueError("Data not compatible with experiment, check 'train_config.cfg' and data source. "
                             "Expecting intput to be of no structure (list check failed)")
        if not isinstance(first_sample[0], str):
            raise ValueError("Data not compatible with experiment, check 'train_config.cfg' and data source. "
                             "Expecting intput to be of no structure (str check failed)")
        logger.info("NoStructure input data found, CHECK: OK")

        def _quick_fix_(s):
            s = s[0] if len(s) == 1 else s
            if isinstance(s, str):
                logger.warning("Quick fix applied to data, check data source for errors")
                s = [ss + '.' for ss in s.split('.')]
            return s

        # remove last level of list
        source = {key: [_quick_fix_(sample)
            for sample in tqdm(value, desc=f"PreProcessing {key}")]
            for key, value in source.items()}

        return source, target
