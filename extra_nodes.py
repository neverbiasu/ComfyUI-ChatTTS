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
        try:
            # Get waveform and convert to appropriate format
            waveform = audio["waveform"]
            logger.info(
                f"Original audio shape: {waveform.shape}, dimensions: {waveform.dim()}"
            )

            # 1. Format audio to the correct format: should be a float 1D array
            if waveform.dim() == 4:  # [batch, batch, channels, samples]
                waveform = waveform[0, 0]
            elif waveform.dim() == 3:  # [batch, channels, samples]
                waveform = waveform[0, 0]
            elif waveform.dim() == 2:  # [channels, samples]
                waveform = waveform[0]

            # Ensure it is continuous 1D data
            if waveform.dim() > 1:
                waveform = waveform.reshape(-1)

            # 2. Convert to numpy, ensure type is float32
            # Key point 1: Use the correct data type
            wav_numpy = waveform.cpu().float().numpy().astype(np.float32)

            # Key point 2: Ensure audio length is sufficient (not caused by filter parameters)
            n_fft = model.config.vocos.feature_extractor.init_args.n_fft
            if len(wav_numpy) < n_fft:
                padded = np.zeros(n_fft, dtype=np.float32)
                padded[: len(wav_numpy)] = wav_numpy
                wav_numpy = padded

            # Key point 3: Normalize audio amplitude (Important!)
            if np.abs(wav_numpy).max() > 1.0:
                wav_numpy = wav_numpy / np.abs(wav_numpy).max()

            # Directly call the model method for processing
            try:
                # Use copy to ensure data continuity
                spk_emb = model.sample_audio_speaker(wav_numpy.copy())
                logger.info(
                    f"Speaker embedding extracted successfully, length: {len(spk_emb)}"
                )
                return ({"spk_emb": spk_emb},)
            except Exception as inner_e:
                logger.error(f"Error in speaker extraction: {inner_e}", exc_info=True)
                # Here you can choose to return an error or use a random speaker
                raise  # Let the outer try-except catch this error

        except Exception as e:
            logger.error(f"Fatal error in extract_speaker: {e}", exc_info=True)
            return ({"error": str(e)},)


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
            # Split by periods, exclamation marks, question marks followed by space or end of string
            parts = re.split(r'(?<=[.!?])\s*', text)
        elif split_method == "paragraphs":
            # Split by one or more empty lines
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
        spk_emb_1 = speaker_params_1.get("spk_emb")
        spk_emb_2 = speaker_params_2.get("spk_emb")

        if spk_emb_1 is None or spk_emb_2 is None:
            logger.warning("One or both speaker embeddings are missing or None.")
            # Return the one that is not None, or the first one if both are None (or an error dict)
            return (speaker_params_1 if spk_emb_1 is not None else speaker_params_2,)

        # Assuming spk_emb are numpy arrays or tensors that support arithmetic operations
        try:
            # Ensure they are tensors for interpolation
            if isinstance(spk_emb_1, np.ndarray):
                spk_emb_1 = torch.from_numpy(spk_emb_1)
            if isinstance(spk_emb_2, np.ndarray):
                spk_emb_2 = torch.from_numpy(spk_emb_2)

            if not isinstance(spk_emb_1, torch.Tensor) or not isinstance(spk_emb_2, torch.Tensor):
                 raise TypeError("Speaker embeddings must be NumPy arrays or PyTorch tensors.")

            if spk_emb_1.shape != spk_emb_2.shape:
                logger.error(f"Speaker embeddings have different shapes: {spk_emb_1.shape} vs {spk_emb_2.shape}")
                # Handle shape mismatch, e.g., return an error or one of the inputs
                return ({"error": "Speaker embeddings shape mismatch"},)

            # Perform linear interpolation
            mixed_emb = (1 - mix_ratio) * spk_emb_1 + mix_ratio * spk_emb_2

            logger.info(f"Mixed speaker embeddings with ratio {mix_ratio}")
            # Return the mixed embedding in the expected dictionary format
            return ({"spk_emb": mixed_emb},)

        except Exception as e:
            logger.error(f"Error mixing speaker embeddings: {e}", exc_info=True)
            return ({"error": f"Mixing error: {e}"},)


class ChatTTS_SaveSpeakerProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speaker_params": ("DICT", {}),
                "filename": ("STRING", {"default": "speaker_profile.pt"}), # Changed default extension
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "chattts"
    OUTPUT_NODE = True

    def save(self, speaker_params, filename):
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Ensure filename has a .pt extension for clarity
        if not filename.lower().endswith(".pt"):
            filename += ".pt"

        filepath = os.path.join(output_dir, filename)

        try:
            # Save the dictionary containing the tensor using torch.save
            # This handles tensors correctly
            torch.save(speaker_params, filepath)
            logger.info(f"Saved speaker profile to {filepath}")
        except Exception as e:
             logger.error(f"Error saving speaker profile: {e}", exc_info=True)
             # Optionally, re-raise or handle the error appropriately
             # raise e

        return {} # Return empty tuple for OUTPUT_NODE = True


class ChatTTS_LoadSpeakerProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Allow selecting file from input directory
                "filepath": (folder_paths.get_filename_list("speaker_profiles"), ),
            }
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "load"
    CATEGORY = "chattts"

    def load(self, filepath):
        # Construct full path using the input directory
        input_dir = folder_paths.get_input_directory()
        full_filepath = os.path.join(input_dir, filepath)

        if not os.path.exists(full_filepath):
             logger.error(f"Speaker profile file not found: {full_filepath}")
             return ({"error": f"File not found: {filepath}"},)

        try:
            # Load the dictionary using torch.load
            # map_location='cpu' ensures it loads correctly even if saved on GPU
            speaker_params = torch.load(full_filepath, map_location='cpu')

            # Basic validation if it's a dictionary and contains 'spk_emb'
            if not isinstance(speaker_params, dict) or "spk_emb" not in speaker_params:
                logger.error(f"Invalid speaker profile format in {filepath}. Expected a dict with 'spk_emb'.")
                return ({"error": "Invalid profile format"},)

            # Ensure the embedding is on the correct device if needed later, though CPU is often fine for parameters
            # if isinstance(speaker_params.get("spk_emb"), torch.Tensor):
            #     speaker_params["spk_emb"] = speaker_params["spk_emb"].to(torch.device("cpu")) # Or your target device

            logger.info(f"Loaded speaker profile from {filepath}")
            return (speaker_params,)
        except Exception as e:
            logger.error(f"Error loading speaker profile from {filepath}: {e}", exc_info=True)
            return ({"error": str(e)},)
