import os

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
import gradio as gr
import torch
import torchaudio
import librosa
from modules.commons import str2bool
from models_loading import load_models
import numpy as np
from pydub import AudioSegment
import argparse
# Load model and configuration

fp16 = False
device = None


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


# streaming and chunk processing related params
# max_context_window = sr // hop_length * 30
# overlap_frame_len = 16
# overlap_wave_len = overlap_frame_len * hop_length
bitrate = "320k"

model_f0, semantic_fn, vocoder_fn, campplus_model, to_mel_f0, mel_fn_args = (
    None,
    None,
    None,
    None,
    None,
    None,
)
f0_fn = None
overlap_wave_len = None
max_context_window = None
sr = None
hop_length = None
overlap_frame_len = 16


@torch.no_grad()
@torch.inference_mode()
def voice_conversion(
    source,
    target,
    diffusion_steps,
    length_adjust,
    inference_cfg_rate,
    auto_f0_adjust,
    pitch_shift,
):
    inference_module = model_f0
    mel_fn = to_mel_f0
    # Load audio
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    # Resample
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    # if source audio less than 30 seconds, whisper can handle in one forward
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[
                    :, traversed_time : traversed_time + 16000 * 30
                ]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        converted_waves_16k[
                            :,
                            traversed_time : traversed_time
                            + 16000 * (30 - overlapping_time),
                        ],
                    ],
                    dim=-1,
                )
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += (
                30 * 16000
                if traversed_time == 0
                else chunk.size(-1) - 16000 * overlapping_time
            )
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    F0_ori = f0_fn(ref_waves_16k[0], thred=0.03)
    F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

    if device.type == "mps":
        F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]
    else:
        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

    voiced_F0_ori = F0_ori[F0_ori > 1]
    voiced_F0_alt = F0_alt[F0_alt > 1]

    log_f0_alt = torch.log(F0_alt + 1e-5)
    voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
    voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
    median_log_f0_ori = torch.median(voiced_log_f0_ori)
    median_log_f0_alt = torch.median(voiced_log_f0_alt)

    # shift alt log f0 level to ori log f0 level
    shifted_log_f0_alt = log_f0_alt.clone()
    if auto_f0_adjust:
        shifted_log_f0_alt[F0_alt > 1] = (
            log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        )
    shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    if pitch_shift != 0:
        shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(
            shifted_f0_alt[F0_alt > 1], pitch_shift
        )

    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    prompt_condition, _, codes, commitment_loss, codebook_loss = (
        inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )
    )
    interpolated_shifted_f0_alt = torch.nn.functional.interpolate(
        shifted_f0_alt.unsqueeze(1), size=cond.size(1), mode="nearest"
    ).squeeze(1)
    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        chunk_f0 = interpolated_shifted_f0_alt[
            :, processed_frames : processed_frames + max_source_window
        ]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16 if fp16 else torch.float32
        ):
            # Voice Conversion
            vc_target = inference_module.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = vocoder_fn(vc_target.float()).squeeze().cpu()
        if vc_wave.ndim == 1:
            vc_wave = vc_wave.unsqueeze(0)
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                output_wave = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = (
                    AudioSegment(
                        output_wave.tobytes(),
                        frame_rate=sr,
                        sample_width=output_wave.dtype.itemsize,
                        channels=1,
                    )
                    .export(format="mp3", bitrate=bitrate)
                    .read()
                )
                yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = (
                AudioSegment(
                    output_wave.tobytes(),
                    frame_rate=sr,
                    sample_width=output_wave.dtype.itemsize,
                    channels=1,
                )
                .export(format="mp3", bitrate=bitrate)
                .read()
            )
            yield mp3_bytes, None
        elif is_last_chunk:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
            )
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = (
                AudioSegment(
                    output_wave.tobytes(),
                    frame_rate=sr,
                    sample_width=output_wave.dtype.itemsize,
                    channels=1,
                )
                .export(format="mp3", bitrate=bitrate)
                .read()
            )
            yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
            break
        else:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = (
                AudioSegment(
                    output_wave.tobytes(),
                    frame_rate=sr,
                    sample_width=output_wave.dtype.itemsize,
                    channels=1,
                )
                .export(format="mp3", bitrate=bitrate)
                .read()
            )
            yield mp3_bytes, None


def main(args):
    global \
        model_f0, \
        semantic_fn, \
        vocoder_fn, \
        campplus_model, \
        to_mel_f0, \
        mel_fn_args, \
        f0_fn
    global overlap_wave_len, max_context_window, sr, hop_length
    model_f0, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel_f0, mel_fn_args = (
        load_models(args.fp16, True, args.checkpoint, args.config)
    )
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]
    # streaming and chunk processing related params
    max_context_window = sr // hop_length * 30
    overlap_wave_len = overlap_frame_len * hop_length
    description = (
        "Zero-shot voice conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
        "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
        "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> "
    )
    inputs = [
        gr.Audio(type="filepath", label="Source Audio"),
        gr.Audio(type="filepath", label="Reference Audio"),
        gr.Slider(
            minimum=1,
            maximum=200,
            value=10,
            step=1,
            label="Diffusion Steps",
            info="10 by default, 50~100 for best quality",
        ),
        gr.Slider(
            minimum=0.5,
            maximum=2.0,
            step=0.1,
            value=1.0,
            label="Length Adjust",
            info="<1.0 for speed-up speech, >1.0 for slow-down speech",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            value=0.7,
            label="Inference CFG Rate",
            info="has subtle influence",
        ),
        gr.Checkbox(
            label="Auto F0 adjust",
            value=True,
            info="Roughly adjust F0 to match target voice. Only works when F0 conditioned model is used.",
        ),
        gr.Slider(
            label="Pitch shift",
            minimum=-24,
            maximum=24,
            step=1,
            value=0,
            info="Pitch shift in semitones, only works when F0 conditioned model is used.",
        ),
    ]

    outputs = [
        gr.Audio(label="Stream Output Audio", streaming=True, format="mp3"),
        gr.Audio(label="Full Output Audio", streaming=False, format="wav"),
    ]

    gr.Interface(
        fn=voice_conversion,
        description=description,
        inputs=inputs,
        outputs=outputs,
        title="Seed Voice Conversion",
        cache_examples=False,
    ).launch(
        share=args.share,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the checkpoint file", default=None
    )
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default=None
    )
    parser.add_argument(
        "--share",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to share the app",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use fp16",
        default=True,
    )
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"

    if torch.cuda.is_available():
        device = torch.device(cuda_target)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    main(args)
