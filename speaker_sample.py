from .tools.seeder import TorchSeedContext
from .utils import logger


class ChatTTS_SpeakerSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 2, "min": 1, "max": 4294967295}),
            }
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "sample"
    CATEGORY = "chattts"

    def sample(self, model, seed):
        # Sample speaker using random seed, following WebUI implementation
        with TorchSeedContext(seed):
            spk_emb = model.sample_random_speaker()
        logger.info(f"Sampled speaker with seed: {seed}")
        return ({"spk_emb": spk_emb},)
