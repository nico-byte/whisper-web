from dataclasses import dataclass
from datetime import date
import torch


@dataclass
class AudioChunk:
    data: torch.Tensor
    timestamp: date


@dataclass
class Transcription:
    text: str
    timestamp: date
