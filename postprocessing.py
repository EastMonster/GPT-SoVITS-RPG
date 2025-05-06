from io import BytesIO
from pydub import AudioSegment
from pydub.generators import WhiteNoise


def post_process(audio_bytes: bytes, sample_rate: int, noise_volume: float) -> bytes:
    audio = AudioSegment.from_raw(
        BytesIO(audio_bytes), frame_rate=32000, channels=1, sample_width=2
    )
    low_quality = audio.set_frame_rate(sample_rate)

    noise = WhiteNoise(sample_rate=32000).to_audio_segment(
        duration=len(low_quality), volume=noise_volume
    )
    final_audio = low_quality.overlay(noise)

    return final_audio.raw_data
