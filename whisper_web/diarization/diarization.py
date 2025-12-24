### Stolen from https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/diarize.py
import torch
import re
from typing import Any, Dict, Optional

from deepmultilingualpunctuation import PunctuationModel
from whisper_web.diarization.lib import MSDDDiarizer
from packaging import version
from transformers import AutoModelForCTC, AutoTokenizer, Wav2Vec2ForCTC
from transformers import __version__ as transformers_version
from transformers.utils import is_flash_attn_2_available

import asyncio
from whisper_web.events import DiarizationCompleted, EventBus

import logging
import numpy
import math

from whisper_web.diarization.utils import (
    langs_to_iso,
    get_words_speaker_mapping,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
)


from ctc_forced_aligner import (
    get_alignments,
    get_spans,
    postprocess_results,
    preprocess_text,
)

logger = logging.getLogger(__name__)

SAMPLING_FREQ = 16000


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


def load_alignment_model(
    device: str,
    model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    attn_implementation: str = None,
    dtype: torch.dtype = torch.float32,
):
    if attn_implementation is None:
        if version.parse(transformers_version) < version.parse("4.41.0"):
            attn_implementation = "eager"
        elif is_flash_attn_2_available() and device == "cuda" and dtype in [torch.float16, torch.bfloat16]:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
    model = (
        AutoModelForCTC.from_pretrained(
            model_path,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate_emissions(model: Wav2Vec2ForCTC, audio_waveform, window_length=30, context_length=2, batch_size=1):
    # Ensure input is 1D numpy array
    assert audio_waveform.ndim == 1, "audio_waveform must be a 1D array"
    audio_waveform = audio_waveform.cpu().numpy()

    context = context_length * SAMPLING_FREQ
    window = window_length * SAMPLING_FREQ
    extension = math.ceil(audio_waveform.shape[0] / window) * window - audio_waveform.shape[0]
    padded_waveform = numpy.pad(audio_waveform, (context, context + extension), mode="constant")

    num_windows = (padded_waveform.shape[0] - 2 * context) // window
    if num_windows <= 0:
        input_windows = numpy.array([padded_waveform])
    else:
        input_windows = numpy.array([padded_waveform[i * window : i * window + window + 2 * context] for i in range(num_windows)])

    emissions_list = []
    model.eval()

    # detect device and helper cache cleanup
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device("cpu")

    def _empty_accelerator_cache():
        try:
            if str(model_device).startswith("cuda") and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
            if str(model_device).startswith("mps") and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    # process in small batches and free memory after each
    with torch.inference_mode():
        for i in range(0, len(input_windows), max(1, batch_size)):
            batch = input_windows[i : i + batch_size].astype(numpy.float32)
            input_tensor = torch.from_numpy(batch)  # (B, T)
            # move to model device if possible
            try:
                input_tensor = input_tensor.to(model_device)
            except Exception:
                input_tensor = input_tensor.cpu()

            # run model
            outputs = model(input_tensor)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    logits = outputs[0]
                else:
                    raise RuntimeError("Unexpected model output format when generating emissions")

            # convert to numpy log-probs for consistency
            if logits.dtype.is_floating_point:
                # compute log_softmax on-device for stability
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
            else:
                log_probs = logits.cpu().numpy()

            emissions_list.append(log_probs)

            # free intermediate tensors & accelerator caches aggressively
            try:
                del input_tensor, outputs, logits, log_probs
            except Exception:
                pass
            _empty_accelerator_cache()

    # concatenate and post-process
    emissions = numpy.concatenate(emissions_list, axis=0)

    # Free the list to save memory
    del emissions_list
    _empty_accelerator_cache()

    if emissions.ndim == 3:
        # Flatten batch dimension: (B, T, V) -> (B*T, V)
        emissions = emissions.reshape(-1, emissions.shape[-1])

    start_frame = time_to_frame(context_length)
    end_frame_offset = time_to_frame(context_length)

    if end_frame_offset > 0:
        emissions = emissions[start_frame:-end_frame_offset, :]  # Correct axis
    else:
        emissions = emissions[start_frame:, :]

    # Remove extension padding frames
    ext_frames = time_to_frame(extension / SAMPLING_FREQ) if extension > 0 else 0
    if ext_frames > 0:
        emissions = emissions[:-ext_frames, :]

    # Heuristic: if values look like logits (many positives), convert to log-probs
    if numpy.max(emissions) > 20:
        a_max = numpy.max(emissions, axis=-1, keepdims=True)
        a_exp = numpy.exp(emissions - a_max)
        emissions = numpy.log(a_exp / numpy.sum(a_exp, axis=-1, keepdims=True))
        del a_max, a_exp  # Free intermediate arrays

    # Add blank token dimension
    emissions = numpy.concatenate([emissions, numpy.zeros((emissions.shape[0], 1), dtype=numpy.float32)], axis=1)
    emissions = emissions.astype(numpy.float32)

    stride = float(audio_waveform.shape[0] * 1000 / emissions.shape[0] / SAMPLING_FREQ)
    return emissions, math.ceil(stride)


def run_forced_alignment(
    alignment_model,
    alignment_tokenizer,
    audio_waveform,
    transcript,
    language,
    batch_size,
):
    try:
        emissions, stride = generate_emissions(
            alignment_model,
            audio_waveform.to(alignment_model.dtype).to(alignment_model.device),
            batch_size=batch_size,
        )
    except Exception as e:
        logger.error(f"Failed to generate emissions: {e}", exc_info=True)
        raise e

    try:
        tokens_starred, text_starred = preprocess_text(
            transcript,
            romanize=True,
            language=language,
        )
    except Exception as e:
        logger.error(f"Failed to preprocess text: {e}", exc_info=True)
        raise e

    try:
        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )
    except Exception as e:
        logger.error(f"Failed to get alignments: {e}", exc_info=True)
        raise e

    spans = get_spans(tokens_starred, segments, blank_token)
    return postprocess_results(text_starred, spans, stride, scores)


class DiarizationEngine:
    """
    DiarizationEngine that can be driven by events.
    If an EventBus is provided, the engine will subscribe to DiarizationRequest
    events, run diarization in a background thread, and publish DiarizationCompleted.
    """

    def __init__(
        self,
        event_bus: EventBus,
        device: str = "mps",
        diarizer_type: str = "msdd",
    ):
        self.device = device
        self.diarizer_type = diarizer_type
        self.punct_model = None
        self.event_bus = event_bus
        self._impl = None
        self._init_diarizer()

    def _init_diarizer(self):
        if self.diarizer_type == "msdd":
            try:
                self.diarizer_model = MSDDDiarizer(device=self.device)
            except ImportError:
                self.diarizer_model = None
                logger.warning("MSDDDiarizer not available. Diarization will be disabled.")
        else:
            self.diarizer_model = None
            logger.warning(f"Unknown diarizer type: {self.diarizer_type}")

        # Load alignment model
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            self.device,
            dtype=torch.float32,
        )

    async def __call__(
        self,
        audio_waveform: Any,
        start_time: int,
        transcript: str,
        language: str,
        session_id: Optional[str] = None,
        batch_size: int = 1,
        suppress_numerals: bool = False,
        punctuate: bool = True,
        alignment_helpers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform diarization and return speaker-attributed transcript and word/segment mapping.

        Args:
                audio_waveform: np.ndarray or torch.Tensor, mono audio waveform
                transcript: str, full transcript text
                language: str, ISO language code
                batch_size: int, batch size for alignment
                suppress_numerals: bool, whether to suppress numerals
                punctuate: bool, whether to restore punctuation
                alignment_helpers: dict, helpers for forced alignment (see below)

        Returns:
                dict with keys: 'speaker_segments', 'word_speaker_mapping', 'sentence_speaker_mapping', 'transcript_with_speakers'
        """
        # --- Forced Alignment ---
        start_time = (start_time / SAMPLING_FREQ) * 1000  # convert to ms

        word_timestamps = await asyncio.to_thread(
            run_forced_alignment,
            self.alignment_model,
            self.alignment_tokenizer,
            audio_waveform,
            transcript,
            langs_to_iso[language],
            batch_size,
        )

        # --- Diarization ---
        try:
            if self.diarizer_model is not None:
                speaker_ts = self.diarizer_model.diarize(audio_waveform.unsqueeze(0))
            else:
                speaker_ts = []
        except Exception as e:
            logger.error(f"Failed to obtain speaker timestamps: {e}", exc_info=True)
            speaker_ts = []

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        # --- Punctuation Restoration ---
        if punctuate:
            try:
                if self.punct_model is None:
                    self.punct_model = PunctuationModel(model="kredor/punctuate-all")
                words_list = list(map(lambda x: x["word"], wsm))
                labeled_words = self.punct_model.predict(words_list)
                ending_puncts = ".?!"
                model_puncts = ".,;:!?"

                def is_acronym(x):
                    return re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

                for word_dict, labeled_tuple in zip(wsm, labeled_words):
                    word = word_dict["word"]
                    if word and labeled_tuple[1] in ending_puncts and (word[-1] not in model_puncts or is_acronym(word)):
                        word += labeled_tuple[1]
                        if word.endswith(".."):
                            word = word.rstrip(".")
                        word_dict["word"] = word
            except Exception as e:
                logger.warning(f"Punctuation restoration failed: {e}")

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # Compose speaker-aware transcript as string
        from io import StringIO

        transcript_buf = StringIO()
        get_speaker_aware_transcript(ssm, transcript_buf)
        transcript_with_speakers = transcript_buf.getvalue()

        try:
            for word_info in wsm:
                word_info["start_time"] += start_time
                word_info["end_time"] += start_time
        except Exception as e:
            logger.error(f"Error adjusting word timestamps: {e}", exc_info=True)

        try:
            for sentence_info in ssm:
                sentence_info["start_time"] += start_time
                sentence_info["end_time"] += start_time
        except Exception as e:
            logger.error(f"Error adjusting sentence timestamps: {e}", exc_info=True)

        result = {
            "speaker_segments": speaker_ts,
            "word_speaker_mapping": wsm,
            "sentence_speaker_mapping": ssm,
            "transcript_with_speakers": transcript_with_speakers,
        }

        await self.event_bus.publish(DiarizationCompleted(result=result, session_id=session_id))
