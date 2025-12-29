### Stolen from https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/diarization/msdd/msdd.py
import os
import tempfile
from typing import List, Tuple, Union, Dict

import torch
import torchaudio

from pyannote.audio import Pipeline  # type: ignore
import pyannote

import logging

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])  # type: ignore
torch.serialization.add_safe_globals([pyannote.audio.core.task.Specifications])  # type: ignore
torch.serialization.add_safe_globals([pyannote.audio.core.task.Problem])  # type: ignore
torch.serialization.add_safe_globals([pyannote.audio.core.task.Resolution])  # type: ignore


class MSDDDiarizer:
    def __init__(self, device="mps"):
        self.device = device
        self.hf_token = os.getenv("HF_TOKEN", None)
        self.min_duration = 0.1  # seconds

        if not self.hf_token:
            logger.warning("HF_TOKEN not found in environment variables")

        try:
            logger.info("Loading pyannote speaker-diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=self.hf_token)
            logger.info("Pyannote pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pyannote pipeline: {e}", exc_info=True)
            raise e

    def _save_tensor_to_wav(self, audio: torch.Tensor, sample_rate: int = 16000) -> str:
        """
        Save a torch tensor to a temp WAV file for pyannote pipeline consumption.
        Expects audio in one of:
          - 1D tensor (N,) => saved as mono
          - 2D tensor (channels, N)
          - 3D tensor (batch, channels, N) where batch==1
        Returns path to wav file.
        """
        if not isinstance(audio, torch.Tensor):
            raise ValueError("audio must be a torch.Tensor")

        # Normalize shapes
        if audio.ndim == 3:
            # assume (batch, channels, N)
            if audio.size(0) == 1:
                audio = audio.squeeze(0)
            else:
                # collapse batch by averaging (unlikely)
                audio = audio.mean(dim=0)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, N)

        # Ensure float32 and cpu
        audio = audio.detach().cpu()
        if audio.dtype != torch.float32:
            audio = audio.to(torch.float32)

        tmp_dir = tempfile.mkdtemp(prefix="pyannote_wav_")
        out_path = os.path.join(tmp_dir, "input.wav")
        # torchaudio expects shape (channels, time)
        torchaudio.save(out_path, audio, sample_rate)
        return out_path

    def diarize(self, audio: Union[torch.Tensor, str]) -> List[Tuple[int, int, int]]:
        """
        Run diarization and return list of (start_ms, end_ms, speaker_idx).
        If pipeline not available or an error occurs, returns empty list.
        """
        if self.pipeline is None:
            logger.warning("Diarization pipeline is None")
            return []

        tmp_path = None
        input_path = None
        try:
            if isinstance(audio, (str, os.PathLike)):
                input_path = str(audio)
                logger.info(f"Using audio file: {input_path}")
            else:
                # assume tensor-like
                input_path = self._save_tensor_to_wav(audio)
                tmp_path = os.path.dirname(input_path)
                logger.info(f"Saved tensor to temporary file: {input_path}")

            # Verify file exists and has content
            if not os.path.exists(input_path):
                logger.error(f"Audio file does not exist: {input_path}")
                return []

            # Check audio duration
            info = torchaudio.info(input_path)
            duration = info.num_frames / info.sample_rate

            if duration < self.min_duration:
                logger.warning(f"Audio too short ({duration:.2f}s < {self.min_duration}s), skipping diarization")
                return []

            file_size = os.path.getsize(input_path)
            logger.info(f"Audio file size: {file_size} bytes, duration: {duration:.2f}s")

            # Run pyannote pipeline - PASS THE FILE PATH DIRECTLY
            logger.info("Running diarization pipeline...")
            diarization_output = self.pipeline(input_path)

            # Extract the annotation from the output
            if hasattr(diarization_output, "speaker_diarization"):
                diarization = diarization_output.speaker_diarization
            elif hasattr(diarization_output, "itertracks"):
                diarization = diarization_output
            else:
                logger.error(f"Unexpected diarization output type: {type(diarization_output)}")
                return []

            # Map speaker labels (strings) to integer indices
            speaker_map: Dict[str, int] = {}
            next_idx = 0
            labels: List[Tuple[int, int, int]] = []

            # Iterate over segments with speaker label
            segment_count = 0
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment_count += 1
                logger.debug(f"Segment {segment_count}: turn={turn}, speaker={speaker}")
                start_s = float(turn.start)
                end_s = float(turn.end)

                if speaker not in speaker_map:
                    speaker_map[speaker] = next_idx
                    logger.info(f"New speaker detected: {speaker} -> idx {next_idx}")
                    next_idx += 1

                speaker_idx = speaker_map[speaker]
                # Add the start_time offset to align with original audio timeline
                start_ms = int(start_s * 1000)
                end_ms = int(end_s * 1000)

                labels.append((start_ms, end_ms, speaker_idx))

            logger.info(f"Total segments found: {segment_count}")
            logger.info(f"Total speakers: {len(speaker_map)}")
            logger.info(f"Speaker mapping: {speaker_map}")

            if not labels:
                logger.warning("No diarization segments found!")

            labels = sorted(labels, key=lambda x: x[0])
            return labels

        except Exception as e:
            logger.error(f"pyannote diarization failed: {e}", exc_info=True)
            return []
        finally:
            # cleanup temp dir if we created one
            if tmp_path and os.path.exists(tmp_path):
                try:
                    import shutil

                    shutil.rmtree(tmp_path, ignore_errors=True)
                    logger.debug(f"Cleaned up temporary directory: {tmp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")
