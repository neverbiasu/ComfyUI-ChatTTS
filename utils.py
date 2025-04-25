import os
import shutil
from .tools.logger import get_logger

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
