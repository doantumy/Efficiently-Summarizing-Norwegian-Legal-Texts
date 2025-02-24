"""
Order of running

Stage 0 (Preprocessing)
./dataset_loader/{data name}

Stage 1 (coarse)
source_segment.py -> split source in stage 0
target_matching.py -> split target in stage 0
train_summerizer.sh -> training a summarizer
inference.py -> run multiple instance to generate summaries, and combine them
coarse_seg.py -> combine the generated summaries to form the input to the next stage

Stage 2 (fine-grained)
train_summarizer.sh -> train a fine-grained summarizer
inference.py -> run multiple instance to generate summaries, and combine them
AnyROUGE.py -> evaluate the final results

"""
import datetime
import json
import os
from loguru import logger

# print(os.environ.get('PYTHONPATH', '').split(os.pathsep))

from argparse_dataclass import ArgumentParser
import wandb
from models import SourceSegmentor

from utils.gpu import free_memory, Encoders
from utils.training_args import TrainArgs
from utils.configue import Configure, load_run_configuration
from utils.tools import get_dataloader, save_config
from utils.dataset import assertion_statis, write_finegrained_dataset, load_split_aslist
from torchmetrics.text.rouge import ROUGEScore


def run_trainer(data_folder, 
                trainer_output_folder, 
                base_model, 
                stage_arg, 
                run, 
                group_name, 
                mode,
                learning_rate,
                batch_size,
                save_total_limit,
                logging_steps, 
                max_length,
                min_length, 
                max_new_tokens,
                temperature, 
                top_k, 
                top_p, 
                no_repeat_ngram_size, 
                num_beams,):
    if os.getenv('DEBUG', False) == 'True':
        logger.debug(
            f"DEBUG MODE: skipping | zsh models/train_summarizor.sh "
            f"{data_folder} {trainer_output_folder} {base_model}"
            f" {stage_arg.warmup_steps} {stage_arg.fine_tune_epochs} {run} {group_name} {mode}"
            f" {learning_rate} {batch_size} {save_total_limit} {logging_steps} {max_length}"
            f" {min_length} {max_new_tokens} {temperature} {top_k} {top_p} {no_repeat_ngram_size} {num_beams}")
    else:
        logger.info(f"Training model using {base_model}")
        os.system(
            f"zsh models/train_summarizor.sh {data_folder} {trainer_output_folder} {base_model}"
            f" {stage_arg.warmup_steps} {stage_arg.fine_tune_epochs} {run} {group_name} {mode}"
            f" {learning_rate} {batch_size} {save_total_limit} {logging_steps} {max_length}"
            f" {min_length} {max_new_tokens} {temperature} {top_k} {top_p} {no_repeat_ngram_size} {num_beams}")


def main():
    verbose = os.getenv('VERBOSE', False) == 'True'
    # Parse the run configuration
    run_args, instances = load_run_configuration()
    logger.info(f"Load run args: {run_args}")
    # Parse all arguments
    parser = ArgumentParser(TrainArgs)
    # args, unknown = parser.parse_known_args()
    # Get only known arguments [0]
    training_args = parser.parse_known_args()[0]
    args = Configure.Get(training_args.cfg)
    args.train = training_args  # combine shell & file configs
    args.cur_stage = 0  # we name the data-collecting stage as stage_0
    args.run_args = run_args
    logger.info(f"Run args: {run_args}")
    group_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_args.start_time = group_name
    args.train.output_path = f"{args.train.output_path}.{group_name}"

    # save the config
    config = args.as_dict()
    save_config(config, os.path.join(args.train.output_path, "config.json"))

    wandb.init(project=args.run_args.run, group=group_name, config=config, mode=args.run_args.wandb_mode,
               save_code=False, name="META")

    # Load dataset using dataset loader
    # loader_name specifies that the dataset loader is defined in the `loader_name`.py in cfg file
    dataset_loader = get_dataloader(args.dataset.loader_name)(args)
    source_data, target_data = dataset_loader.load()

    # get data checker and check the data
    checker = instances["data_check"]()
    source_data, target_data = checker(source_data, target_data)

    assertion_statis(source_data, target_data, f"Finish loading stage {args.cur_stage} dataset!")
    if args.train.save_intermediate is True:
        dataset_loader.save()

    # Coarse stages
    # args.cur_stage = args.dataset.stage_num - 1
    # if you want to skip coarse stages (or any stage) and the dataset for fine-grained stage is ready,
    # you can uncomment this line of code
    while args.cur_stage < args.dataset.stage_num - 1:
        args.cur_stage += 1

        # Source Segmentation
        source_segmentor = instances["source_segmentor"](args, source_data, target_data, load_from_file=False)
        # Create groups of sentences and duplicate the target for each group
        split_source, duplicated_target, counter = source_segmentor.segment()
        source_segmentor.save()

        # Target Matching
        logger.info(f"Start target matching of Stage {args.cur_stage}. This may take several minutes.")
        # align the target (summary) sentences with each subgroup
        target_segmentor = instances["target_segmentor"](args, split_source, duplicated_target, load_from_file=False)
        target, _ = target_segmentor.segment()
        target_segmentor.save()

        assertion_statis(split_source, target, f"Finish loading stage {args.cur_stage} dataset!")

        stage_data_folder = os.path.join(args.train.output_path, f"stage_{args.cur_stage}")
        stage_trainer_output_folder = os.path.join(stage_data_folder, "trainer_output")
        stage_arg = getattr(args, f"stage{args.cur_stage}")
        stage_arg.trainer_output_folder = stage_trainer_output_folder

        if args.train.mode == "train":
            # Use collected data to run model training
            run_trainer(stage_data_folder, stage_trainer_output_folder, args.run_args.base_model,
                        getattr(args, f"stage{args.cur_stage}"),
                        args.run_args.run, group_name, args.run_args.wandb_mode,
                        args.run_args.learning_rate, args.run_args.batch_size,
                        args.run_args.save_total_limit, args.run_args.logging_steps,
                        args.run_args.max_length,
                        min_length=args.run_args.min_length,
                        max_new_tokens=args.run_args.max_new_tokens, 
                        temperature=args.run_args.temperature,
                        top_k=args.run_args.top_k, 
                        top_p=args.run_args.top_p,
                        no_repeat_ngram_size=args.run_args.no_repeat_ngram_size,
                        num_beams=args.run_args.num_beams,)

            # Inference using the trained checkpoint
            summary_generator = instances["summary_generator"](args, split_source, fine_grained=False)
            split_hypo = summary_generator.inference(bsz=args.run_args.batch_size,
                                                    min_length=args.run_args.min_length,
                                                    max_new_tokens=args.run_args.max_new_tokens, 
                                                    temperature=args.run_args.temperature,
                                                    top_k=args.run_args.top_k, 
                                                    top_p=args.run_args.top_p,
                                                    no_repeat_ngram_size=args.run_args.no_repeat_ngram_size,
                                                    num_beams=args.run_args.num_beams,)
            if args.train.save_intermediate is True:
                summary_generator.save()

        else:
            # Inference using the trained checkpoint
            summary_generator = instances["summary_generator"](args, split_source, fine_grained=False, test_mode=True)
            split_hypo = summary_generator.inference(bsz=args.run_args.batch_size,
                                                    min_length=args.run_args.min_length,
                                                    max_new_tokens=args.run_args.max_new_tokens, 
                                                    temperature=args.run_args.temperature,
                                                    top_k=args.run_args.top_k, 
                                                    top_p=args.run_args.top_p,
                                                    no_repeat_ngram_size=args.run_args.no_repeat_ngram_size,
                                                    num_beams=args.run_args.num_beams)
            if args.train.save_intermediate is True:
                summary_generator.save()

        free_memory([summary_generator], debug=verbose)

        # # Combine coarse segments to form the next stage's input
        combiner = instances["coarse_seg_combiner"](args, split_hypo, counter, load_from_file=False)
        # combiner = CoarseSegCombiner(args, None, counter, load_from_file=True)
        hypo = combiner.combine()
        combiner.save()

        # save the config
        config = args.as_dict()
        save_config(config, os.path.join(args.train.output_path, "config.json"))

        # update the segmentor to the default one for without structure for further stages
        instances["source_segmentor"] = SourceSegmentor
        source_data = hypo

    # # Fine-grained Stage
    source_path = os.path.join(args.train.output_path, f"stage_{args.cur_stage}")
    cur_source = load_split_aslist(source_path, suffix='hypo')
    cur_target = target_data

    args.cur_stage += 1
    stage_data_folder = os.path.join(args.train.output_path, f"stage_{args.cur_stage}")
    write_finegrained_dataset(cur_source, cur_target, stage_data_folder)
    assertion_statis(cur_source, cur_target, f"Finish loading stage {args.cur_stage} dataset!")

    stage_trainer_output_folder = os.path.join(stage_data_folder, "trainer_output")
    stage_arg = getattr(args, f"stage{args.cur_stage}")
    stage_arg.trainer_output_folder = stage_trainer_output_folder

    if args.train.mode == "train":
        # use the pre-trained model to learn a summary from the summaries
        run_trainer(stage_data_folder, stage_trainer_output_folder, args.run_args.base_model,
                        getattr(args, f"stage{args.cur_stage}"),
                        args.run_args.run, group_name, args.run_args.wandb_mode,
                        args.run_args.learning_rate, args.run_args.batch_size,
                        args.run_args.save_total_limit, args.run_args.logging_steps,
                        args.run_args.max_length,
                        min_length=args.run_args.min_length,
                        max_new_tokens=args.run_args.max_new_tokens, 
                        temperature=args.run_args.temperature,
                        top_k=args.run_args.top_k, 
                        top_p=args.run_args.top_p,
                        no_repeat_ngram_size=args.run_args.no_repeat_ngram_size,
                        num_beams=args.run_args.num_beams,)

        # Inference using the trained checkpoint
        # use the summary-summary model
        summary_generator = instances["summary_generator"](args, cur_source, fine_grained=True)
        cur_hypo = summary_generator.inference(bsz=args.run_args.batch_size,
                                                min_length=args.run_args.min_length,
                                                max_new_tokens=args.run_args.max_new_tokens, 
                                                temperature=args.run_args.temperature,
                                                top_k=args.run_args.top_k, 
                                                top_p=args.run_args.top_p,
                                                no_repeat_ngram_size=args.run_args.no_repeat_ngram_size,
                                                num_beams=args.run_args.num_beams)
        summary_generator.save()
    else:
        # Inference using the trained checkpoint
        # use the summary-summary model
        summary_generator = instances["summary_generator"](args, cur_source, fine_grained=True, test_mode=True)
        cur_hypo = summary_generator.inference(bsz=args.run_args.batch_size,
                                                min_length=args.run_args.min_length,
                                                max_new_tokens=args.run_args.max_new_tokens, 
                                                temperature=args.run_args.temperature,
                                                top_k=args.run_args.top_k, 
                                                top_p=args.run_args.top_p,
                                                no_repeat_ngram_size=args.run_args.no_repeat_ngram_size,
                                                num_beams=args.run_args.num_beams)
        summary_generator.save()

    free_memory([summary_generator], debug=verbose)

    # calculate rouge and log
    # rouge without precession and recall

    rouge = ROUGEScore()
    rouge_scores = rouge(preds=cur_hypo['test'],
                         target=cur_target['test'])
    logger.info(f"ROUGE SCORES: {rouge_scores}")
    # save to file
    with open(os.path.join(stage_data_folder, "rouge_scores.json"), 'w') as f:
        json.dump(rouge_scores, f, indent=4, cls=Encoders)

    # save the config
    config = args.as_dict()
    config["rouge_scores"] = rouge_scores
    save_config(config, os.path.join(args.train.output_path, "config.json"))

    wandb.finish()


if __name__ == '__main__':
    main()
