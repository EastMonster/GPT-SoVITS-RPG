"""The tauri-app."""

from io import BytesIO
from typing import Optional

import numpy as np
import soundfile as sf
from anyio.from_thread import start_blocking_portal
from pydantic import BaseModel


from pytauri import (
    BuilderArgs,
    Commands,
    builder_factory,
    context_factory,
)

from .GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

commands: Commands = Commands()


class Info(BaseModel):
    """Info model."""

    name: str


counter = 0


@commands.command()
async def greet() -> Info:
    global counter
    counter += 1
    return Info(name=f"Hello, World! counter = {counter}")


class TtsParam(BaseModel):
    """TTS parameters."""

    text: Optional[str] = None
    text_lang: Optional[str] = None
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[list] = None
    prompt_lang: Optional[str] = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


class TtsAudio(BaseModel):
    """TTS audio."""

    audio: list[int]


import os

print(os.getcwd())
tts_config = TTS_Config("python/GPT_SoVITS/GPT_SoVITS/configs/tts_infer.yaml")
print(tts_config)
tts_pipeline = TTS(tts_config)


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


@commands.command()
async def tts(body: TtsParam) -> TtsAudio:
    try:
        tts_generator = tts_pipeline.run(body.model_dump())

        sr, audio_data = next(tts_generator)
        audio_data = pack_wav(BytesIO(), audio_data, sr).getvalue()
        data = list(audio_data)
        print(len(data))
        return TtsAudio(audio=data)
    except Exception as e:
        print(e)
        return TtsAudio(audio=[])


def main() -> None:
    """Run the tauri-app."""
    with start_blocking_portal("asyncio") as portal:
        app = builder_factory().build(
            BuilderArgs(
                context=context_factory(),
                invoke_handler=commands.generate_handler(portal),
            )
        )
        app.run()
