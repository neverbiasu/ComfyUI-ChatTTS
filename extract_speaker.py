from .tools.logger import get_logger

logger = get_logger("ChatTTS-Extract")


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
            # Extract speaker embeddings from input audio
            waveform = audio["waveform"].squeeze(0)
            logger.info(f"Extracting speaker from audio: shape={waveform.shape}")
            spk_emb = model.sample_audio_speaker(waveform.cpu().numpy())
            logger.info(f"Speaker embedding extracted successfully")
            return ({"spk_emb": spk_emb},)
        except Exception as e:
            logger.error(f"Error extracting speaker: {e}")
            return ({"error": str(e)},)
