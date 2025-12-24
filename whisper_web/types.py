from dataclasses import dataclass
from datetime import date
import torch


@dataclass
class AudioChunk:
    data: torch.Tensor
    start_time: int


@dataclass
class Transcription:
    text: str
    timestamp: date
