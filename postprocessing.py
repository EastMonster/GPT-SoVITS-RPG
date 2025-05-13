import math
from io import BytesIO

import numpy as np
from pydub import AudioSegment
from pydub.effects import low_pass_filter
from pydub.generators import WhiteNoise


def noise_fluctuation(audio: AudioSegment, base_volume: float) -> AudioSegment:
    volume_fluctuation = 2  # dB, 音量将在 base_volume ± volume_fluctuation 之间浮动
    chunk_duration_ms = 100  # ms, 每小段的持续时间

    total_duration_ms = len(audio)
    num_chunks = math.ceil(total_duration_ms / chunk_duration_ms)
    fluctuating_noise = AudioSegment.empty()

    for i in range(num_chunks):
        # 计算当前块的随机音量
        random_offset = np.random.uniform(-volume_fluctuation, volume_fluctuation)
        current_volume = base_volume + random_offset

        # 确定当前块的持续时间
        current_chunk_duration = min(
            chunk_duration_ms, total_duration_ms - len(fluctuating_noise)
        )
        if current_chunk_duration <= 0:
            break

        # 生成一小段白噪音
        noise_chunk = WhiteNoise(sample_rate=32000).to_audio_segment(
            duration=current_chunk_duration, volume=current_volume
        )
        fluctuating_noise += noise_chunk

    # 应用低通滤波器使噪音更低沉
    # 调整 cutoff_frequency 来改变低沉的程度，值越小声音越低沉 (例如 500, 1000, 2000 Hz)
    cutoff_frequency = 1000  # Hz
    filtered_noise = low_pass_filter(fluctuating_noise, cutoff_frequency)

    final_audio = audio.overlay(filtered_noise)
    return final_audio


def voice_stutter(audio: AudioSegment) -> AudioSegment:
    stutter_segment_duration_ms = 100
    silence_probability = 0.2

    processed_segments = []
    cursor_ms = 0
    while cursor_ms < len(audio):
        segment_end_ms = cursor_ms + stutter_segment_duration_ms
        # 获取当前片段，确保不超出音频总长度
        current_segment = audio[cursor_ms : min(segment_end_ms, len(audio))]

        # 如果当前片段长度为0（例如，当光标到达音频末尾时），则停止处理
        if len(current_segment) == 0:
            break

        # 决定是否在当前片段前插入静音
        if np.random.rand() < silence_probability:
            # 插入一段与当前片段等长的静音
            silence_to_insert = AudioSegment.silent(
                duration=len(current_segment), frame_rate=audio.frame_rate
            )
            processed_segments.append(silence_to_insert)
        
        # 总是添加原始的当前片段
        processed_segments.append(current_segment)

        cursor_ms = segment_end_ms

    if processed_segments:
        return sum(processed_segments, AudioSegment.empty())
    else:
        return audio


def post_process(
    audio_bytes: bytes, sample_rate: int, noise_volume: float, stutter: bool
) -> bytes:
    audio = AudioSegment.from_raw(
        BytesIO(audio_bytes), frame_rate=32000, channels=1, sample_width=2
    )
    low_quality = audio.set_frame_rate(sample_rate)

    if stutter is True:
        low_quality = voice_stutter(low_quality)

    final_audio = noise_fluctuation(low_quality, noise_volume)

    return final_audio.raw_data
