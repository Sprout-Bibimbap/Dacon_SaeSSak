import argparse
import torch
import json
from dotenv import load_dotenv
from sources.utils import *
from sources.trainer import Trainer
from sources.data_loader import *


def main(args):
    print("This script is running directly")
    model, tokenizer = get_model_tokenizer(args)
    train_loader, valid_loader = get_loader(args.config, tokenizer)
    trainer = Trainer(args, model, train_loader, valid_loader)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Development Score")
    parser.add_argument(
        "--config",
        type=str,
        default="./sources/base_config.json",
        help="Path to the training configuration file",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    load_dotenv()
    with open(args.config, "r", encoding="utf-8") as f:
        args.config = json.load(f)
    main(args)
