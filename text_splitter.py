from .tools.logger import get_logger

logger = get_logger("ChatTTS-Splitter")


class ChatTTS_TextSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "split_method": (
                    ["newline", "sentences", "paragraphs", "none", "tokens"],
                    {"default": "newline"},
                ),
            },
            "optional": {
                "max_batch_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                "max_tokens_per_part": (
                    "INT",
                    {"default": 200, "min": 10, "max": 1000},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "split"
    CATEGORY = "chattts"

    def split(
        self, text, split_method="newline", max_batch_size=4, max_tokens_per_part=200
    ):
        try:
            import re

            if split_method == "none":
                return (text,)

            if split_method == "newline":
                parts = text.split("\n")
            elif split_method == "sentences":
                parts = re.split(r'(?<=。)|(?<=\.\s)|(?<=\!)|(?<=\?)', text)
            elif split_method == "paragraphs":
                parts = re.split(r'\n\s*\n', text)
            elif split_method == "tokens":
                # Split by approximate token count
                words = text.split()
                parts = []
                current_part = []
                current_count = 0

                for word in words:
                    if current_count + len(word) > max_tokens_per_part:
                        parts.append(" ".join(current_part))
                        current_part = [word]
                        current_count = len(word)
                    else:
                        current_part.append(word)
                        current_count += len(word)

                if current_part:
                    parts.append(" ".join(current_part))

            # Filter empty parts
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) > max_batch_size:
                logger.info(
                    f"Text split into {len(parts)} parts, limiting to {max_batch_size}"
                )
                parts = parts[:max_batch_size]

            # Join back with newlines to preserve format for the TTS process
            return ("\n".join(parts),)
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            return (text,)  # Return original text on error
