import argparse
import configparser
import os
from loguru import logger

import models as build_bricks
from utils.training_args import TrainArgs
from utils.parameters import parse_parameters

DEFAULT_CONFIGURE_DIR = "configure"
DEFAULT_DATASET_DIR = "data"
DEFAULT_MODEL_DIR = "models"


class Args(object):
    def __init__(self, contain=None):
        self.__self__ = contain
        self.__default__ = None
        self.__default__ = set(dir(self))

    def __call__(self):
        return self.__self__

    def __getattribute__(self, name):
        if name[:2] == "__" and name[-2:] == "__":
            return super().__getattribute__(name)
        if name not in dir(self):
            return None
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if not (value is None) or (name[:2] == "__" and name[-2:] == "__"):
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in dir(self) and name not in self.__default__:
            super().__delattr__(name)

    def __iter__(self):
        return list((arg, getattr(self, arg)) for arg in set(dir(self)) - self.__default__).__iter__()

    def __len__(self):
        return len(set(dir(self)) - self.__default__)

    def as_dict(self):
        kv = {}
        for arg in set(dir(self)) - self.__default__:
            argv = getattr(self, arg)
            if isinstance(argv, Args):
                kv[arg] = argv.as_dict()
            elif isinstance(argv, TrainArgs):
                kv[arg] = argv.__dict__
            elif isinstance(argv, argparse.Namespace):
                kv[arg] = argv.__dict__
            else:
                kv[arg] = argv

        return kv


class String(object):
    @staticmethod
    def to_basic(string):
        """
        Convert the String to what it really means.
        For example, "true" --> True as a bool value
        :param string:
        :return:
        """
        try:
            return int(string)
        except ValueError:
            try:
                return float(string)
            except ValueError:
                pass
        if string in ["True", "true"]:
            return True
        elif string in ["False", "false"]:
            return False
        else:
            return string.strip("\"'")  # for those we want to add space before and after the string


class Configure(object):
    @staticmethod
    def get_file_cfg(file):
        """
        get configurations in file.
        :param file:
        :return: configure args
        """
        cfgargs = Args()
        parser = configparser.ConfigParser()
        parser.read(file)
        for section in parser.sections():
            setattr(cfgargs, section, Args())
            for item in parser.items(section):
                setattr(getattr(cfgargs, section), item[0], String.to_basic(item[1]))
        return cfgargs

    @staticmethod
    def refresh_args_by_file_cfg(file, prev_args):
        args = Configure.get_file_cfg(file)
        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        for arg_name, arg in prev_args:
            if arg is None:
                continue
            if arg_name != "cfg":
                names = arg_name.split(".")
                cur = args
                for name in names[: -1]:
                    if getattr(cur, name) is None:
                        setattr(cur, name, Args())
                    cur = getattr(cur, name)
                if getattr(cur, names[-1]) is None:
                    setattr(cur, names[-1], arg)
        return args

    @staticmethod
    def get_console_cfg(default_file):
        """
        get configurations from console.
        :param default_file:
        :return:
        """
        conargs = Args()
        parser = argparse.ArgumentParser()
        types = {"bool": bool, "int": int, "float": float}
        args_label = Configure.get_file_cfg(default_file)
        for arg_name, arg in args_label:
            argw = {}
            if arg.help:
                argw["help"] = arg.help
            if arg.type == "implicit_bool" or arg.type == "imp_bool":
                argw["action"] = "store_true"
            if arg.type == "string" or arg.type == "str" or arg.type is None:
                if arg.default:
                    if arg.default == "None" or "none":
                        argw["default"] = None
                    else:
                        argw["default"] = arg.default
            if arg.type in types:
                argw["type"] = types[arg.type]
                if arg.default:
                    if arg.default == "None" or "none":
                        argw["default"] = None
                    else:
                        argw["default"] = types[arg.type](arg.default)
            parser.add_argument("--" + arg_name, **argw)
        tmpargs = parser.parse_args()
        for arg_name, arg in args_label:
            setattr(conargs, arg_name, getattr(tmpargs, arg_name))
        return conargs

    @staticmethod
    def Get(cfg):
        args = Configure.get_file_cfg(os.path.join(DEFAULT_CONFIGURE_DIR, cfg))

        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        return args


def load_bricks(cfg, run):
    config = configparser.ConfigParser()
    config.read(cfg)

    run_config = config[f"{run}.BRICKS"]
    instances = {}
    additional_config = {}
    for element in run_config:
        value = run_config[element]

        impl = getattr(build_bricks, value, None)
        if impl is not None:
            instances[element] = impl
        if value.isnumeric():
            value = int(value)
        additional_config[element] = value

    return additional_config, instances


def load_run_configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tc", "--run-cfg", type=str, default="run_config.cfg",
                        help="Configuration file name")
    parser.add_argument("-r", "--run", type=str, default="summ.n",
                        help="Run mode, default is summ.n")
    parser.add_argument("-bm", "--base-model", type=str, default="ltg/nort5-small",
                        help="T5 base model name")
    parser.add_argument("-uq", "--use-quantize", type=bool, default=False,
                        help="Use quantized model for inference")
    parser.add_argument("-d", "--debug", type=bool, default=False)
    parser.add_argument("-wm", "--wandb-mode", type=str, default="offline")
    
    logger.info(f"Loading geneneration settings parameters")
    # Add the parameters from utils/parameters.py
    parser = parse_parameters(parser)

    args = parser.parse_known_args()[0]
    add_config, instances = load_bricks(args.run_cfg, args.run)
    # add add_config to args
    for element in add_config:
        setattr(args, element, add_config[element])

    return args, instances


if __name__ == '__main__':
    args = load_run_configuration()
    # logger.info(f"Load run args: {args}")
