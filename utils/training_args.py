from dataclasses import dataclass, field


@dataclass
class TrainArgs(object):
    cfg: str = field(
        default="",
        metadata={"help": "The path from ./configure to store configure file."})

    dataset_path: str = field(
        default="/data/yfz5488/src/SummScreen/TVMegaSite/",
        metadata={"help": "The absolute path to the dataset folder."}
    )

    output_path: str = field(
        default="./output/SummScreenMG",
        metadata={"help": "The path to the output folder."}
    )

    save_intermediate: bool = field(
        default=True,
        metadata={"help": "Store or not the intermediate files, such as original dataset."}
    )

    model_path: str = field(
        default="./bart.large.cnn/models.pt",
        metadata={"help": "The path to store the models .pt checkpoint. The models is loaded before training"})

    # cuda_devices: str = field(
    #     default="0,1",
    #     metadata={'help': "The index of GPUs used to train model, seperated by , ."}
    # )

    mode: str = field(
        default="train",
        metadata={"help": "Train the whole dataset or test on test set."}
    )

    # checkpoint_dir: str = field(
    #     default="",
    #     metadata={"help": "The directory to save the checkpoints"}
    # )
