from .node import ChatTTSLoader, ChatTTS_SpeakerSample, ChatTTS_Sampler

NODE_CLASS_MAPPINGS = {
    "ChatTTSLoader": ChatTTSLoader,
    "ChatTTS_SpeakerSample": ChatTTS_SpeakerSample,
    "ChatTTS_Sampler": ChatTTS_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTSLoader": "ChatTTS Loader",
    "ChatTTS_SpeakerSample": "ChatTTS Speaker Sample",
    "ChatTTS_Sampler": "ChatTTS Sampler",
}
