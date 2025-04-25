import os
import tempfile
import folder_paths

from .tools.logger import get_logger
from .ChatTTS.core import Chat
from .ChatTTS.utils import download_all_assets
from .utils import logger


class ChatTTSLoader:
    @classmethod
    def INPUT_TYPES(cls):
        chattts_dir = os.path.join(folder_paths.models_dir, "chattts")
        if not os.path.exists(chattts_dir):
            os.makedirs(chattts_dir, exist_ok=True)
        files = [
            f
            for f in os.listdir(chattts_dir)
            if f.endswith(('.pth', '.pt', '.ckpt', '.safetensors'))
        ]
        return {
            "required": {
                "model_path": (["(auto)"] + sorted(files), {"default": "(auto)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "chattts"

    def load(self, model_path):
        # Get model directory path
        chattts_dir = os.path.join(folder_paths.models_dir, "chattts")

        if model_path == "(auto)":
            files = [
                f
                for f in os.listdir(chattts_dir)
                if f.endswith(('.pth', '.pt', '.ckpt', '.safetensors'))
            ]

            if files:
                # If there are model files in the directory, use the first one
                model_path = os.path.join(chattts_dir, files[0])
                logger.info(f"Using local model: {model_path}")
            else:
                # No local model found, use ChatTTS download mechanism
                logger.info(
                    "No local model found, downloading with ChatTTS downloader..."
                )

                try:
                    # Use ChatTTS built-in downloader to download to model directory
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        logger.info(f"Downloading ChatTTS model to {chattts_dir}...")
                        download_all_assets(tmpdir=tmp_dir, homedir=chattts_dir)

                    # Check directory after download
                    files = [
                        f
                        for f in os.listdir(chattts_dir)
                        if f.endswith(('.pth', '.pt', '.ckpt', '.safetensors'))
                    ]

                    if not files:
                        raise RuntimeError("No model file found after download")

                    model_path = os.path.join(chattts_dir, files[0])
                    logger.info(f"Model downloaded successfully: {model_path}")

                except Exception as e:
                    # If above method fails, try using huggingface method
                    logger.warning(
                        f"ChatTTS downloader failed: {e}, trying HuggingFace method..."
                    )

                    chat = Chat(logger)
                    # Use custom_path parameter to specify models/chattts directory
                    loaded = chat.load("huggingface", custom_path=chattts_dir)

                    if not loaded:
                        logger.error("Model download failed")
                        raise RuntimeError(
                            "Download failed, please try manually downloading the model files and place them in models/chattts directory"
                        )

                    # If successful, return the loaded model directly
                    logger.info("Model loaded successfully")
                    return (chat,)
        else:
            # Use user-specified model
            model_path = os.path.join(chattts_dir, model_path)
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        chat = Chat(logger)
        loaded = chat.load("custom", custom_path=model_path)

        if not loaded:
            raise RuntimeError("Failed to load ChatTTS model")

        logger.info(f"ChatTTS model loaded successfully: {model_path}")
        return (chat,)
