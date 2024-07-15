import os
import torch
from tqdm import tqdm
from sources.utils import get_criterion, get_optimizer, get_scheduler
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, args, model, train_loader, valid_loader):
        self.model = model
        self.device = args.device
        self.model_name = args.model
        self.model_ver = args.version
        self.start_time = args.time

        self.criterion = get_criterion(args.config)
        self.optimizer = get_optimizer(args.config, self.model)
        self.scheduler = get_scheduler(args.config, self.optimizer)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.epochs = args.config["epochs"]
        self.save_interval = args.config["save_interval"]

        self.best_loss = float("inf")

    def train(self):
        for epoch in range(self.epochs):
            print("=" * 50)
            self.model.train()
            running_loss = 0.0
            for inputs, targets in tqdm(self.train_loader, ncols=100):
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            validation_loss = self.valid()

            print(
                f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {running_loss/len(self.train_loader):.4f}, Valid Loss: {validation_loss:.4f}"
            )
            wandb.log(
                {
                    "Train loss": running_loss / len(self.train_loader),
                    "Valid Loss": validation_loss,
                },
                step=epoch,
            )
            # 스케줄러 업데이트
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(validation_loss)
            else:
                self.scheduler.step()

            if validation_loss < self.best_loss:
                print("**Best Model**")
                self.best_loss = validation_loss
                self.save(epoch + 1)

    def valid(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(self.valid_loader, ncols=100):
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                targets = targets.to(self.device)
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.valid_loader)
        return avg_loss

    def save(self, epoch):
        base_dir = os.path.dirname(__file__)
        checkpoint_dir = os.path.join(base_dir, "..", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{self.model_name}{self.model_ver}_{self.start_time}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")
