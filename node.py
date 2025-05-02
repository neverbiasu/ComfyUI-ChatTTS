import os
import shutil
import torch
import tempfile
import numpy as np

import folder_paths

from .tools.seeder import TorchSeedContext
from .tools.logger import get_logger

from .ChatTTS.core import Chat
from .ChatTTS.utils import download_all_assets

logger = get_logger("ChatTTS")


def clean_corrupted_cache():
    """Clear corrupted HuggingFace cache"""
    try:
        cache_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--2Noise--ChatTTS"
        )
        if os.path.exists(cache_path):
            logger.info(f"Cleaning corrupted cache: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
        return True
    except Exception as e:
        logger.error(f"Failed to clean cache: {e}")
        return False


class ChatTTSLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # No inputs are needed as the path is determined automatically.
        return {"required": {}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "chattts"

    def load(self):
        # Get model directory path
        chattts_dir = os.path.join(folder_paths.models_dir, "chattts")
        os.makedirs(chattts_dir, exist_ok=True)  # Ensure the directory exists

        model_path = None
        files = [
            f
            for f in os.listdir(chattts_dir)
            if f.endswith(('.pth', '.pt', '.ckpt', '.safetensors'))
        ]

        if files:
            # If there are model files in the directory, use the first one
            model_path = os.path.join(chattts_dir, files[0])
            logger.info(f"Using local model: {model_path}")
            source = "custom"  # Treat local files as custom source
        else:
            # No local model found, use ChatTTS download mechanism
            logger.info(
                "No local model found in models/chattts, attempting download..."
            )
            source = "huggingface"  # Default to huggingface download
            model_path = chattts_dir  # Pass the directory for download

            try:
                # Attempt download using ChatTTS built-in downloader first
                # Note: download_all_assets might not be the intended function here,
                # as Chat.load handles download internally based on source.
                # We rely on Chat.load('huggingface') to handle the download.
                logger.info(f"Attempting download to: {chattts_dir}")
                # The actual download happens within chat.load if source is 'huggingface'
            except Exception as e:
                logger.error(f"Initial download check failed: {e}")
                raise RuntimeError(
                    "Failed to prepare for download. Check permissions or network."
                )

        # Load model using the determined source and path
        chat = Chat(logger)
        logger.info(f"Loading model using source='{source}' and path='{model_path}'")

        # Use custom_path for both local files and download directory
        loaded = chat.load(source=source, custom_path=model_path)

        if not loaded:
            # Check if download failed or local file loading failed
            if source == "huggingface":
                logger.error(
                    f"Model download/load failed from HuggingFace to {model_path}"
                )
                # Clean potentially corrupted cache if download failed
                clean_corrupted_cache()
                raise RuntimeError(
                    f"Download/load failed. Please check network or manually place model files in {chattts_dir}"
                )
            else:  # source == "custom"
                logger.error(f"Failed to load local model file: {model_path}")
                raise RuntimeError(f"Failed to load ChatTTS model from {model_path}")

        logger.info(
            f"ChatTTS model loaded successfully from: {model_path if source == 'custom' else chattts_dir}"
        )
        return (chat,)


class ChatTTS_SeedBasedSpeaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "generate"
    CATEGORY = "chattts"

    def generate(self, model, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        spk_emb = model.sample_random_speaker()

        return ({"spk_emb": spk_emb, "seed": seed},)


class ChatTTS_Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "speaker_params": ("DICT", {}),
                "seed": ("INT", {"default": 2, "min": 1, "max": 4294967295}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_P": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_K": ("INT", {"default": 20, "min": 1, "max": 100}),
                "split_batch": ("INT", {"default": 0, "min": 0, "max": 32}),
            },
        }

    RETURN_TYPES = ("AUDIO", "DICT")
    FUNCTION = "synthesize"
    CATEGORY = "chattts"

    def synthesize(
        self,
        model,
        text,
        speaker_params=None,
        seed=2,
        temperature=0.3,
        top_P=0.7,
        top_K=20,
        split_batch=0,
    ):
        # Implementation following WebUI funcs.py
        spk_emb = speaker_params.get("spk_emb", "") if speaker_params else ""
        logger.info(
            f"ChatTTS audio synthesis: text length={len(text)}, speaker encoding length={len(spk_emb) if spk_emb else 0}"
        )

        # Construct parameters using same approach as WebUI
        params_infer_code = Chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K,
            manual_seed=seed,
        )

        try:
            logger.info(f"Starting inference: split_batch={split_batch}")
            with TorchSeedContext(seed):
                # Call infer method as in WebUI, without streaming
                wav = model.infer(
                    text,
                    skip_refine_text=True,
                    params_infer_code=params_infer_code,
                    stream=False,
                    split_text=split_batch > 0,
                    max_split_batch=split_batch,
                )

            logger.info(
                f"Inference completed: wav type={type(wav)}, length={len(wav) if wav is not None else 'None'}"
            )

            # Defensive check - ensure wav is not None and has at least one element
            if wav is None or len(wav) == 0:
                logger.error("Model returned empty wav or None")
                # Create default empty audio
                empty_audio = {
                    "waveform": torch.zeros((1, 100)),
                    "sample_rate": 24000,
                }
                meta = {
                    "seed": seed,
                    "spk_emb": spk_emb,
                    "error": "Model returned empty audio",
                }
                return (empty_audio, meta)

            # Ensure wav[0] exists and has correct format
            if not isinstance(wav[0], (list, tuple, np.ndarray)):
                logger.error(f"wav[0] has incorrect type: {type(wav[0])}")
                empty_audio = {
                    "waveform": torch.zeros((1, 100)),
                    "sample_rate": 24000,
                }
                meta = {
                    "seed": seed,
                    "spk_emb": spk_emb,
                    "error": f"Audio data format error: {type(wav[0])}",
                }
                return (empty_audio, meta)

            # Process return value - ensure tensor has correct format
            tensor = torch.tensor(wav[0], dtype=torch.float32)
            logger.info(
                f"Audio tensor: shape={tensor.shape}, dimensions={tensor.dim()}"
            )

            # Ensure audio data is in the format required by torchaudio.save [batch_size, channels, samples]
            # PreviewAudio node expects an iterable waveform to iterate over each batch
            if tensor.dim() == 1:
                # 1D tensor [samples] -> 3D tensor [1, 1, samples]
                tensor = tensor.reshape(1, 1, -1).contiguous()
                logger.info(
                    f"Adjusted 1D tensor to 3D [1, 1, samples], new shape={tensor.shape}"
                )
            elif tensor.dim() == 2:
                # 2D tensor [channels, samples] -> 3D tensor [1, channels, samples]
                tensor = tensor.unsqueeze(0).contiguous()
                logger.info(
                    f"Adjusted 2D tensor to 3D [1, channels, samples], new shape={tensor.shape}"
                )
            else:
                # Ensure high-dimension tensors are correctly formatted
                tensor = tensor.view(1, 1, -1).contiguous()
                logger.info(
                    f"Adjusted high-dimension tensor to 3D, new shape={tensor.shape}"
                )

            # Verify tensor integrity
            if not tensor.is_contiguous():
                logger.warning("Tensor not contiguous, forcing contiguous")
                tensor = tensor.contiguous()

            # Print final info
            logger.info(
                f"Final audio tensor: shape={tensor.shape}, dtype={tensor.dtype}, is_contiguous={tensor.is_contiguous()}"
            )

            audio = {
                "waveform": tensor.detach().clone(),  # Create deep copy
                "sample_rate": 24000,
            }

            # Ensure correct return data
            meta = {"seed": seed, "spk_emb": spk_emb}
            logger.info(f"Successfully generated audio: shape={tensor.shape}")
            return (audio, meta)

        except Exception as e:
            logger.error(
                f"Error during audio synthesis: {e}", exc_info=True
            )  # Add full stack trace
            # Return empty but correctly formatted audio data
            empty_audio = {
                "waveform": torch.zeros((1, 100)),  # Create valid 2D empty audio
                "sample_rate": 24000,
            }
            meta = {"seed": seed, "spk_emb": spk_emb, "error": str(e)}
            return (empty_audio, meta)
