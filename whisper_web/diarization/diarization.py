### Stolen from https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/diarize.py
import torch
import logging
import re
from typing import Any, Dict, Optional

from deepmultilingualpunctuation import PunctuationModel
from whisper_web.diarization.lib import MSDDDiarizer
from packaging import version
from transformers import AutoModelForCTC, AutoTokenizer
from transformers import __version__ as transformers_version
from transformers.utils import is_flash_attn_2_available


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
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


class DiarizationEngine:
    """
    Modular diarization engine for speaker-attributed transcription.
    Wraps diarization, forced alignment, and punctuation restoration.

    Usage:
            diarizer = DiarizationEngine(device="cuda", diarizer_type="msdd")
            diarization_result = diarizer.diarize(audio_waveform, transcript, language, batch_size=8)
    """

    def __init__(self, device: str = "cuda", diarizer_type: str = "msdd"):
        self.device = device
        self.diarizer_type = diarizer_type
        self.punct_model = None
        self._init_diarizer()

    def _init_diarizer(self):
        if self.diarizer_type == "msdd":
            try:
                self.diarizer_model = MSDDDiarizer(device=self.device)
            except ImportError:
                self.diarizer_model = None
                logging.warning("MSDDDiarizer not available. Diarization will be disabled.")
        else:
            self.diarizer_model = None
            logging.warning(f"Unknown diarizer type: {self.diarizer_type}")

    def diarize(
        self,
        audio_waveform: Any,
        transcript: str,
        language: str,
        batch_size: int = 8,
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
        # Import alignment helpers here to avoid circular imports
        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            postprocess_results,
            preprocess_text,
        )
        from whisper_web.diarization.utils import (
            langs_to_iso,
            get_words_speaker_mapping,
            get_realigned_ws_mapping_with_punctuation,
            get_sentences_speaker_mapping,
            get_speaker_aware_transcript,
        )

        # Load alignment model
        alignment_model, alignment_tokenizer = load_alignment_model(
            self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        emissions, stride = generate_emissions(
            alignment_model,
            torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device),
            batch_size=batch_size,
        )

        tokens_starred, text_starred = preprocess_text(
            transcript,
            romanize=True,
            language=langs_to_iso[language],
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # --- Diarization ---
        if self.diarizer_model is not None:
            speaker_ts = self.diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
        else:
            speaker_ts = []

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        # --- Punctuation Restoration ---
        if punctuate:
            try:
                if self.punct_model is None:
                    self.punct_model = PunctuationModel(model="kredor/punctuate-all")
                words_list = list(map(lambda x: x["word"], wsm))
                labeled_words = self.punct_model.predict(words_list, chunk_size=230)
                ending_puncts = ".?!"
                model_puncts = ".,;:!?"
                is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
                for word_dict, labeled_tuple in zip(wsm, labeled_words):
                    word = word_dict["word"]
                    if word and labeled_tuple[1] in ending_puncts and (word[-1] not in model_puncts or is_acronym(word)):
                        word += labeled_tuple[1]
                        if word.endswith(".."):
                            word = word.rstrip(".")
                        word_dict["word"] = word
            except Exception as e:
                logging.warning(f"Punctuation restoration failed: {e}")

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # Compose speaker-aware transcript as string
        from io import StringIO

        transcript_buf = StringIO()
        get_speaker_aware_transcript(ssm, transcript_buf)
        transcript_with_speakers = transcript_buf.getvalue()

        # Clean up alignment model to free VRAM
        del alignment_model
        torch.cuda.empty_cache()

        return {
            "speaker_segments": speaker_ts,
            "word_speaker_mapping": wsm,
            "sentence_speaker_mapping": ssm,
            "transcript_with_speakers": transcript_with_speakers,
        }
