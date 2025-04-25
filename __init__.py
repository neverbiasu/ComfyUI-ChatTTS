from .loader import ChatTTSLoader
from .speaker_sample import ChatTTS_SpeakerSample
from .sampler import ChatTTS_Sampler
from .extract_speaker import ChatTTS_ExtractSpeaker
from .text_normalizer import ChatTTS_TextNormalizer
from .text_splitter import ChatTTS_TextSplitter
from .speaker_profile import ChatTTS_SaveSpeakerProfile, ChatTTS_LoadSpeakerProfile

NODE_CLASS_MAPPINGS = {
    "ChatTTSLoader": ChatTTSLoader,
    "ChatTTS_SpeakerSample": ChatTTS_SpeakerSample,
    "ChatTTS_Sampler": ChatTTS_Sampler,
    "ChatTTS_ExtractSpeaker": ChatTTS_ExtractSpeaker,
    "ChatTTS_TextNormalizer": ChatTTS_TextNormalizer,
    "ChatTTS_TextSplitter": ChatTTS_TextSplitter,
    "ChatTTS_SaveSpeakerProfile": ChatTTS_SaveSpeakerProfile,
    "ChatTTS_LoadSpeakerProfile": ChatTTS_LoadSpeakerProfile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTSLoader": "ChatTTS Model Loader",
    "ChatTTS_SpeakerSample": "ChatTTS Speaker Sampler",
    "ChatTTS_Sampler": "ChatTTS Sampler",
    "ChatTTS_ExtractSpeaker": "ChatTTS Voice Extractor",
    "ChatTTS_TextNormalizer": "ChatTTS Text Normalizer",
    "ChatTTS_TextSplitter": "ChatTTS Text Splitter",
    "ChatTTS_SaveSpeakerProfile": "ChatTTS Save Speaker Profile",
    "ChatTTS_LoadSpeakerProfile": "ChatTTS Load Speaker Profile",
}
