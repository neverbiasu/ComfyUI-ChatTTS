import os
import json
import folder_paths
from .tools.logger import get_logger

logger = get_logger("ChatTTS-Profile")


class ChatTTS_SaveSpeakerProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speaker_params": ("DICT", {}),
                "filename": ("STRING", {"default": "speaker_profile.json"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "chattts"
    OUTPUT_NODE = True

    def save(self, speaker_params, filename):
        try:
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)

            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(speaker_params, f, indent=2)

            logger.info(f"Saved speaker profile to {filepath}")
            return {}
        except Exception as e:
            logger.error(f"Error saving speaker profile: {e}")
            return {}


class ChatTTS_LoadSpeakerProfile:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        return {
            "required": {
                "profile_file": (
                    sorted(files),
                    {"default": files[0] if files else "speaker_profile.json"},
                ),
            }
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "load"
    CATEGORY = "chattts"

    def load(self, profile_file):
        try:
            filepath = os.path.join(folder_paths.get_input_directory(), profile_file)

            if not os.path.isfile(filepath):
                logger.error(f"Profile file not found: {filepath}")
                return ({"error": f"File not found: {profile_file}"},)

            with open(filepath, 'r') as f:
                speaker_params = json.load(f)

            logger.info(f"Loaded speaker profile from {filepath}")
            return (speaker_params,)
        except Exception as e:
            logger.error(f"Error loading speaker profile: {e}")
            return ({"error": str(e)},)
