### Stolen from https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/diarization/msdd/msdd.py
import os
import tempfile
import logging
from typing import List, Tuple, Union, Dict

import torch
import torchaudio
from omegaconf import OmegaConf

from pyannote.audio import Pipeline  # type: ignore


class MSDDDiarizer:
    def __init__(self, device: Union[str, torch.device] = "cpu", hf_token: str | None = None):
        """
        Adapter constructor. Loads a pyannote Pipeline when available.

        Params:
            device: device string or torch.device (informational; pipeline uses internal device selection)
            hf_token: optional HuggingFace token if the pipeline model is gated
        """
        self.device = device
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        try:
            # load pretrained diarization pipeline (may require auth token)
            if self.hf_token:
                self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token)
            else:
                # try without token; may raise if model requires auth
                self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        except Exception as e:
            logging.warning(f"Failed to initialize pyannote Pipeline: {e}")
            self.pipeline = None

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
            return []

        tmp_path = None
        input_path = None
        try:
            if isinstance(audio, (str, os.PathLike)):
                input_path = str(audio)
            else:
                # assume tensor-like
                input_path = self._save_tensor_to_wav(audio)
                tmp_path = os.path.dirname(input_path)

            # Run pyannote pipeline
            diarization = self.pipeline(input_path)

            # Map speaker labels (strings) to integer indices
            speaker_map: Dict[str, int] = {}
            next_idx = 0
            labels: List[Tuple[int, int, int]] = []

            # Iterate over segments with speaker label
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_s = float(turn.start)
                end_s = float(turn.end)
                if speaker not in speaker_map:
                    speaker_map[speaker] = next_idx
                    next_idx += 1
                speaker_idx = speaker_map[speaker]
                labels.append((int(start_s * 1000), int(end_s * 1000), speaker_idx))

            labels = sorted(labels, key=lambda x: x[0])
            return labels

        except Exception as e:
            logging.warning(f"pyannote diarization failed: {e}")
            return []
        finally:
            # cleanup temp dir if we created one
            if tmp_path and os.path.exists(tmp_path):
                try:
                    # remove the temporary directory and its contents
                    import shutil

                    shutil.rmtree(tmp_path, ignore_errors=True)
                except Exception:
                    pass


def create_config():
    config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "diar_infer_telephonic.yaml"))
    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.diarizer.out_dir = None
    config.diarizer.manifest_filepath = None
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = False  # compute VAD provided with model_path to vad config
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"  # Telephonic speaker diarization model

    return config
