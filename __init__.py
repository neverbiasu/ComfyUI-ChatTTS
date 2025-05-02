from .node import ChatTTSLoader, ChatTTS_SeedBasedSpeaker, ChatTTS_Sampler
from .extra_nodes import (
    ChatTTS_ExtractSpeaker,
    ChatTTS_TextNormalizer,
    ChatTTS_TextSplitter,
    ChatTTS_SaveSpeakerProfile,
    ChatTTS_LoadSpeakerProfile,
)

NODE_CLASS_MAPPINGS = {
    "ChatTTSLoader": ChatTTSLoader,
    "ChatTTS_SeedBasedSpeaker": ChatTTS_SeedBasedSpeaker,
    "ChatTTS_Sampler": ChatTTS_Sampler,
    "ChatTTS_ExtractSpeaker": ChatTTS_ExtractSpeaker,
    "ChatTTS_TextNormalizer": ChatTTS_TextNormalizer,
    "ChatTTS_TextSplitter": ChatTTS_TextSplitter,
    "ChatTTS_SaveSpeakerProfile": ChatTTS_SaveSpeakerProfile,
    "ChatTTS_LoadSpeakerProfile": ChatTTS_LoadSpeakerProfile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTSLoader": "ChatTTS Model Loader",
    "ChatTTS_SeedBasedSpeaker": "ChatTTS Seed-Based Speaker",
    "ChatTTS_Sampler": "ChatTTS Sampler",
    "ChatTTS_ExtractSpeaker": "ChatTTS Voice Extractor",
    "ChatTTS_TextNormalizer": "ChatTTS Text Normalizer",
    "ChatTTS_TextSplitter": "ChatTTS Text Splitter",
    "ChatTTS_SaveSpeakerProfile": "ChatTTS Save Speaker Profile",
    "ChatTTS_LoadSpeakerProfile": "ChatTTS Load Speaker Profile",
}
