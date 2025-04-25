from .tools.logger import get_logger

logger = get_logger("ChatTTS-Normalizer")


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
        try:
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
        except Exception as e:
            logger.error(f"Error normalizing text: {e}")
            return (text,)  # Return original text on error
