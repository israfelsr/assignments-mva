from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class TrainingArguments:
    seed: int = -1
    run_name: str = MISSING
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 10
    learning_rate: int = 0.01


@dataclass
class ModelArguments:
    pretrained_model_key: Optional[str] = None


@dataclass
class DatasetArguments:
    num_classes: int = MISSING
    train_dir: str = MISSING
    val: str = MISSING


@dataclass
class BirdClassifierArguments:
    datasets: DatasetArguments = DatasetArguments()
    training: TrainingArguments = TrainingArguments()
    model: ModelArguments = ModelArguments()