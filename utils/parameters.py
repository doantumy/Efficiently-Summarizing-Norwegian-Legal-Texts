import argparse

def parse_parameters(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("-lr", "--learning-rate", type=float, default=3e-5)
    parser.add_argument("-bs", "--batch-size", type=int, default=8)
    parser.add_argument("-st", "--save-total-limit", type=int, default=1)
    parser.add_argument("-ls", "--logging-steps", type=int, default=500)
    parser.add_argument("-ml", "--max-length", type=int, default=256, help="Max length of input sequence")
    parser.add_argument("-mnl", "--min-length", type=int, default=128, help="Geneneration settings: Min length of generated output.")
    parser.add_argument("-mnt", "--max-new-tokens", type=int, default=150, help="Geneneration settings: Max number of new tokens")
    parser.add_argument("-tem", "--temperature", type=float, default=0.7, help="Geneneration settings: Temperature < 1 means more deterministic output")
    parser.add_argument("-tk", "--top-k", type=int, default=50, help="Geneneration settings: Top k value, >50 means more creative output")
    parser.add_argument("-tp", "--top-p", type=float, default=0.8, help="Geneneration settings: Top p value, <=0.8 for deterministic output")
    parser.add_argument("-nrng", "--no-repeat-ngram-size", type=int, default=2, help="Geneneration settings: No repeat ngram size")
    parser.add_argument("-nb", "--num-beams", type=int, default=2, help="Geneneration settings: =1: greedy search, >1: beam search multiple hypotheses at each step")

    return parser