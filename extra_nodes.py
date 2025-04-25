import os
import torch
import numpy as np
from typing import Dict, Any

import folder_paths

from .tools.seeder import TorchSeedContext
from .tools.logger import get_logger

from .ChatTTS.core import Chat

logger = get_logger("ChatTTS-Extra")


class ChatTTS_ExtractSpeaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "extract"
    CATEGORY = "chattts"

    def extract(self, model, audio):
        # Extract speaker embeddings from input audio
        waveform = audio["waveform"].squeeze(0)
        logger.info(f"Extracting speaker from audio: shape={waveform.shape}")
        spk_emb = model.sample_audio_speaker(waveform.cpu().numpy())
        logger.info(f"Speaker embedding extracted successfully")
        return ({"spk_emb": spk_emb},)


class ChatTTS_TextNormalizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "language": (["auto", "en", "zh"], {"default": "auto"}),
                "do_text_normalization": ("BOOLEAN", {"default": True}),
                "do_homophone_replacement": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "normalize"
    CATEGORY = "chattts"

    def normalize(
        self,
        text,
        language="auto",
        do_text_normalization=True,
        do_homophone_replacement=True,
    ):
        # Import here to avoid circular imports
        from .ChatTTS.core import Chat
        from .tools.logger import get_logger

        normalizer = Chat(get_logger("Normalizer")).normalizer
        normalized_text = normalizer(
            text,
            do_text_normalization,
            do_homophone_replacement,
            lang=None if language == "auto" else language,
        )

        logger.info(
            f"Normalized text from {len(text)} to {len(normalized_text)} characters"
        )
        return (normalized_text,)


class ChatTTS_TextSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "split_method": (
                    ["newline", "sentences", "paragraphs", "none"],
                    {"default": "newline"},
                ),
            },
            "optional": {
                "max_batch_size": ("INT", {"default": 4, "min": 1, "max": 32}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "split"
    CATEGORY = "chattts"

    def split(self, text, split_method="newline", max_batch_size=4):
        import re

        if split_method == "none":
            return (text,)

        if split_method == "newline":
            parts = text.split("\n")
        elif split_method == "sentences":
            parts = re.split(r'(?<=。)|(?<=\.\s)|(?<=\!)|(?<=\?)', text)
        elif split_method == "paragraphs":
            parts = re.split(r'\n\s*\n', text)

        # Filter empty parts
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) > max_batch_size:
            logger.info(
                f"Text split into {len(parts)} parts, limiting to {max_batch_size}"
            )
            parts = parts[:max_batch_size]

        # Join back with newlines to preserve format for the TTS process
        return ("\n".join(parts),)


class ChatTTS_SpeakerMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speaker_params_1": ("DICT", {}),
                "speaker_params_2": ("DICT", {}),
                "mix_ratio": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "mix"
    CATEGORY = "chattts"

    def mix(self, speaker_params_1, speaker_params_2, mix_ratio):
        # Simple linear interpolation of speaker embeddings
        spk_emb_1 = speaker_params_1.get("spk_emb", "")
        spk_emb_2 = speaker_params_2.get("spk_emb", "")

        if not spk_emb_1 or not spk_emb_2:
            logger.warning("One of the speaker embeddings is empty")
            return (speaker_params_1 if spk_emb_1 else speaker_params_2,)

        # Assuming speaker embeddings are base64 strings, we need to decode them
        # This is a simplified version, actual implementation depends on the format
        import base64

        # Parse speaker embeddings (exact parsing depends on ChatTTS format)
        # For now, we'll just blend the two with the mix ratio
        mixed_emb = f"{spk_emb_1[:10]}...{spk_emb_2[10:20]}...{mix_ratio}"

        logger.info(f"Mixed speaker embeddings with ratio {mix_ratio}")
        return ({"spk_emb": mixed_emb},)


class ChatTTS_SaveSpeakerProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speaker_params": ("DICT", {}),
                "filename": ("STRING", {"default": "speaker_profile.json"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "chattts"
    OUTPUT_NODE = True

    def save(self, speaker_params, filename):
        import json

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(speaker_params, f, indent=2)

        logger.info(f"Saved speaker profile to {filepath}")
        return {}


class ChatTTS_LoadSpeakerProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", {"default": "speaker_profile.json"}),
            }
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "load"
    CATEGORY = "chattts"

    def load(self, filepath):
        import json

        if not os.path.isabs(filepath):
            filepath = os.path.join(folder_paths.get_input_directory(), filepath)

        try:
            with open(filepath, 'r') as f:
                speaker_params = json.load(f)

            logger.info(f"Loaded speaker profile from {filepath}")
            return (speaker_params,)
        except Exception as e:
            logger.error(f"Error loading speaker profile: {e}")
            return ({"error": str(e)},)


# This file now imports from the nodes directory
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
