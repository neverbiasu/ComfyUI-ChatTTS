import os
import torch
from typing import Any, Dict

import folder_paths


from .tools.seeder import TorchSeedContext
from .tools.logger import get_logger

from .ChatTTS.core import Chat

logger = get_logger("ChatTTS")


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
        # 获取模型路径
        chattts_dir = os.path.join(folder_paths.models_dir, "chattts")
        local_model_path = None

        if model_path == "(auto)":
            # 检查本地模型
            files = [
                f
                for f in os.listdir(chattts_dir)
                if f.endswith(('.pth', '.pt', '.ckpt', '.safetensors'))
            ]

            if files:
                # 使用本地模型
                local_model_path = os.path.join(chattts_dir, files[0])
                logger.info(f"Using local model: {local_model_path}")
            else:
                # 本地没有模型，尝试下载
                logger.info("No local model found, attempting to download...")
        else:
            local_model_path = os.path.join(chattts_dir, model_path)
            if not os.path.exists(local_model_path):
                logger.error(f"Model file not found: {local_model_path}")
                raise FileNotFoundError(f"Model file not found: {local_model_path}")

        # 初始化ChatTTS并加载模型
        chat = Chat(logger)

        if local_model_path:
            # 使用本地模型文件
            loaded = chat.load("custom", custom_path=local_model_path)
        else:
            # 从huggingface下载模型
            logger.info("Downloading model from huggingface...")
            loaded = chat.load("huggingface", force_redownload=False)

        if not loaded:
            raise RuntimeError("Failed to load ChatTTS model")

        logger.info(f"ChatTTS model loaded successfully")
        return (chat,)


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
        # 严格按照 WebUI 实现，使用随机种子采样
        with TorchSeedContext(seed):
            spk_emb = model.sample_random_speaker()
        logger.info(f"Sampled speaker with seed: {seed}")
        return ({"spk_emb": spk_emb},)


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
                "stream": ("BOOLEAN", {"default": False}),
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
        stream=False,
        split_batch=0,
    ):
        # 严格按照 WebUI funcs.py 实现
        spk_emb = speaker_params.get("spk_emb", "") if speaker_params else ""

        # 使用WebUI相同的参数构造
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K,
            manual_seed=seed,
        )

        with TorchSeedContext(seed):
            # 按照 WebUI 的方式调用 infer 方法
            wav = model.infer(
                text,
                skip_refine_text=True,
                params_infer_code=params_infer_code,
                stream=stream,
                split_text=split_batch > 0,
                max_split_batch=split_batch,
            )

        # 处理返回值
        if stream:
            audio = None
            for sr, arr in wav:
                audio = {"waveform": torch.tensor(arr).unsqueeze(0), "sample_rate": sr}
                break
        else:
            audio = {
                "waveform": torch.tensor(wav[0]).unsqueeze(0),
                "sample_rate": 24000,
            }

        meta = {"seed": seed, "spk_emb": spk_emb}
        logger.info(f"Synthesized audio: text={text[:20]}..., seed={seed}")
        return (audio, meta)


NODE_CLASS_MAPPINGS = {
    "ChatTTSLoader": ChatTTSLoader,
    "ChatTTS_SpeakerSample": ChatTTS_SpeakerSample,
    "ChatTTS_Sampler": ChatTTS_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTSLoader": "ChatTTS 模型加载",
    "ChatTTS_SpeakerSample": "ChatTTS 说话人采样",
    "ChatTTS_Sampler": "ChatTTS 文本转语音",
}
