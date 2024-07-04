import argparse
import torch
import json
from dotenv import load_dotenv
from sources.utils import *
from sources.trainer import Trainer
from sources.data_loader import *
import wandb
from datetime import datetime
import pytz


def main(args):
    model, tokenizer = get_model_tokenizer(args)
    train_loader, valid_loader = get_loader(args.config, tokenizer)
    trainer = Trainer(args, model, train_loader, valid_loader)
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Development Score")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./sources/base_config.json",
        help="Path to the training configuration file",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    with open(args.config_path, "r", encoding="utf-8") as f:
        args.config = json.load(f)
    korea_timezone = pytz.timezone("Asia/Seoul")
    args.time = datetime.now(korea_timezone).strftime("%Y-%m-%d_%H-%M-%S")
    load_dotenv(args.config["env_path"])
    seed_everything(args.config["seed"])
    args.model = model_identification(args)
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="SaeSSac-Score", name=f"{args.model}_{args.time}")
    start_message(args)
    main(args)
