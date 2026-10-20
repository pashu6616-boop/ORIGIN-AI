from dataclasses import dataclass
from enum import Enum

class CapabilityStatus(str, Enum):
    UNSUPPORTED = "unsupported"

@dataclass
class ModalityResult:
    status: CapabilityStatus
    message: str

def image_input(*_args, **_kwargs):
    return ModalityResult(CapabilityStatus.UNSUPPORTED, "Image understanding is not implemented in ORIGIN v0.1.")
def audio_input(*_args, **_kwargs):
    return ModalityResult(CapabilityStatus.UNSUPPORTED, "Audio and speech recognition are not implemented in ORIGIN v0.1.")
def text_to_speech(*_args, **_kwargs):
    return ModalityResult(CapabilityStatus.UNSUPPORTED, "Text-to-speech is not implemented in ORIGIN v0.1.")
