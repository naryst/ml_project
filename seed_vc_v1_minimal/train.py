import os
import sys
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import torch
import torch.multiprocessing as mp
import random
import librosa
import yaml
import argparse
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import glob
from tqdm import tqdm
import shutil
import logging
import time
from datetime import datetime
import numpy as np

from modules.commons import recursive_munch, build_model, load_checkpoint
from optimizers import build_optimizer
from data.ft_dataset import build_ft_dataloader
from hf_utils import load_custom_model_from_hf

class Trainer:
    def __init__(self,
                 config_path,
                 pretrained_ckpt_path,
                 data_dir,
                 run_name,
                 batch_size=0,
                 num_workers=0,
                 steps=1000,
                 save_interval=500,
                 max_epochs=1000,
                 device="cuda:0",
                 ):
        self.device = device
        config = yaml.safe_load(open(config_path))
        self.log_dir = os.path.join(config['log_dir'], run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        self.logger.info(f"Starting training session: {run_name}")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Device: {device}")
        
        # copy config file to log dir
        shutil.copyfile(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))
        batch_size = config.get('batch_size', 10) if batch_size == 0 else batch_size
        self.max_steps = steps

        self.n_epochs = max_epochs
        self.log_interval = config.get('log_interval', 10)
        self.save_interval = save_interval

        self.sr = config['preprocess_params'].get('sr', 22050)
        self.hop_length = config['preprocess_params']['spect_params'].get('hop_length', 256)
        self.win_length = config['preprocess_params']['spect_params'].get('win_length', 1024)
        self.n_fft = config['preprocess_params']['spect_params'].get('n_fft', 1024)
        preprocess_params = config['preprocess_params']

        self.train_dataloader = build_ft_dataloader(
            data_dir,
            preprocess_params['spect_params'],
            self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.f0_condition = config['model_params']['DiT'].get('f0_condition', False)
        self.build_sv_model(device, config)
        self.build_semantic_fn(device, config)
        if self.f0_condition:
            self.build_f0_fn(device, config)
        self.build_converter(device, config)
        self.build_vocoder(device, config)

        scheduler_params = {
            "warmup_steps": 0,
            "base_lr": 0.00001,
        }

        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, stage='DiT')

        _ = [self.model[key].to(device) for key in self.model]
        self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)

        # initialize optimizers after preparing models for compatibility with FSDP
        self.optimizer = build_optimizer({key: self.model[key] for key in self.model},
                                         lr=float(scheduler_params['base_lr']))

        if pretrained_ckpt_path is None:
            # find latest checkpoint
            available_checkpoints = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*_step_*.pth"))
            if len(available_checkpoints) > 0:
                latest_checkpoint = max(
                    available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                earliest_checkpoint = min(
                    available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                # delete the earliest checkpoint if we have more than 2
                if (
                    earliest_checkpoint != latest_checkpoint
                    and len(available_checkpoints) > 2
                ):
                    os.remove(earliest_checkpoint)
                    print(f"Removed {earliest_checkpoint}")
            elif config.get('pretrained_model', ''):
                latest_checkpoint = load_custom_model_from_hf("Plachta/Seed-VC", config['pretrained_model'], None)
            else:
                latest_checkpoint = ""
        else:
            assert os.path.exists(pretrained_ckpt_path), f"Pretrained checkpoint {pretrained_ckpt_path} not found"
            latest_checkpoint = pretrained_ckpt_path

        if os.path.exists(latest_checkpoint):
            self.model, self.optimizer, self.epoch, self.iters = load_checkpoint(
                self.model, self.optimizer, latest_checkpoint,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False
            )
            print(f"Loaded checkpoint from {latest_checkpoint}")
        else:
            self.epoch, self.iters = 0, 0
            print("Failed to load any checkpoint, training from scratch.")

        from modules.audio import mel_spectrogram
        self.mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": self.hop_length,
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None
            if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
            else 8000,
            "center": False,
        }
        self.mel_function = lambda x: mel_spectrogram(x, **self.mel_fn_args)  # noqa: E731


        self.eval_interval = 1000

    def setup_logging(self):
        """Setup logging to file and console"""
        self.logger = logging.getLogger('seed_vc_trainer')
        self.logger.setLevel(logging.INFO)
        
        # Create a file handler for logging to a file
        log_file = os.path.join(self.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging to {log_file}")

    def build_sv_model(self, device, config):
        from modules.campplus.DTDNN import CAMPPlus
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_sd_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus_sd = torch.load(campplus_sd_path, map_location='cpu')
        self.campplus_model.load_state_dict(campplus_sd)
        self.campplus_model.eval()
        self.campplus_model.to(device)
        self.sv_fn = self.campplus_model

    def build_f0_fn(self, device, config):
        from modules.rmvpe import RMVPE
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=device)
        self.f0_fn = self.rmvpe

    def build_converter(self, device, config):
        from modules.openvoice.api import ToneColorConverter
        ckpt_converter, config_converter = load_custom_model_from_hf("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json")
        self.tone_color_converter = ToneColorConverter(config_converter, device=device)
        self.tone_color_converter.load_ckpt(ckpt_converter)
        self.tone_color_converter.model.eval()
        se_db_path = load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt", None)
        self.se_db = torch.load(se_db_path, map_location='cpu')

    def build_vocoder(self, device, config):
        vocoder_type = config['model_params']['vocoder']['type']
        vocoder_name = config['model_params']['vocoder'].get('name', None)
        if vocoder_type == 'bigvgan':
            from modules.bigvgan import bigvgan
            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(vocoder_name, use_cuda_kernel=False)
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval().to(device)
            vocoder_fn = self.bigvgan_model
        elif vocoder_type == 'hifigan':
            from modules.hifigan.generator import HiFTGenerator
            from modules.hifigan.f0_predictor import ConvRNNF0Predictor
            hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            self.hift_gen = HiFTGenerator(**hift_config['hift'],
                                          f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            self.hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
            self.hift_gen.eval()
            self.hift_gen.to(device)
            vocoder_fn = self.hift_gen
        else:
            raise ValueError(f"Unsupported vocoder type: {vocoder_type}")
        self.vocoder_fn = vocoder_fn

    def build_semantic_fn(self, device, config):
        speech_tokenizer_type = config['model_params']['speech_tokenizer'].get('type', 'cosyvoice')
        if speech_tokenizer_type == 'whisper':
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_model_name = config['model_params']['speech_tokenizer']['name']
            self.whisper_model = WhisperModel.from_pretrained(whisper_model_name).to(device)
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_model_name)
            # remove decoder to save memory
            del self.whisper_model.decoder

            def semantic_fn(waves_16k):
                ori_inputs = self.whisper_feature_extractor(
                    [w16k.cpu().numpy() for w16k in waves_16k],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000,
                )
                ori_input_features = self.whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
                ).to(device)
                with torch.no_grad():
                    ori_outputs = self.whisper_model.encoder(
                        ori_input_features.to(self.whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori

        elif speech_tokenizer_type == "gigaam":
            from transformers import AutoModel, AutoProcessor

            model_name = config['model_params']['speech_tokenizer']['name']

            gigaam_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            gigaam_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

            gigaam_model = gigaam_model.model.encoder
            gigaam_model.eval()

            def semantic_fn(waves_16k):
                ori_features = gigaam_processor(
                    waves_16k.squeeze(0).cpu().numpy(), return_tensors="pt", padding=True, sampling_rate=16000
                )
                ori_features = ori_features.to(device)
                with torch.no_grad():
                    ori_outputs = gigaam_model(ori_features['input_features'], length=ori_features['input_lengths'])
                S_ori = ori_outputs[0].to(torch.float32) # bs x hidden_dim x seq_len
                S_ori = S_ori.transpose(1, 2) # bs x seq_len x hidden_dim
                return S_ori

        elif speech_tokenizer_type == 'xlsr':
            from transformers import (
                Wav2Vec2FeatureExtractor,
                Wav2Vec2Model,
            )
            model_name = config['model_params']['speech_tokenizer']['name']
            output_layer = config['model_params']['speech_tokenizer']['output_layer']
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
            self.wav2vec_model.encoder.layers = self.wav2vec_model.encoder.layers[:output_layer]
            self.wav2vec_model = self.wav2vec_model.to(device)
            self.wav2vec_model = self.wav2vec_model.eval()
            self.wav2vec_model = self.wav2vec_model.half()

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
                ori_inputs = self.wav2vec_feature_extractor(
                    ori_waves_16k_input_list,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    sampling_rate=16000
                ).to(device)
                with torch.no_grad():
                    ori_outputs = self.wav2vec_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        else:
            raise ValueError(f"Unsupported speech tokenizer type: {speech_tokenizer_type}")
        self.semantic_fn = semantic_fn

    def train_one_step(self, batch):
        waves, mels, wave_lengths, mel_input_length = batch

        B = waves.size(0)
        target_size = mels.size(2)
        target = mels
        target_lengths = mel_input_length

        # get speaker embedding
        if self.sr != 22050:
            waves_22k = torchaudio.functional.resample(waves, self.sr, 22050)
            wave_lengths_22k = (wave_lengths.float() * 22050 / self.sr).long()
        else:
            waves_22k = waves
            wave_lengths_22k = wave_lengths
        se_batch = self.tone_color_converter.extract_se(waves_22k, wave_lengths_22k)

        ref_se_idx = torch.randint(0, len(self.se_db), (B,))
        ref_se = self.se_db[ref_se_idx].to(self.device)

        # convert
        converted_waves_22k = self.tone_color_converter.convert(
            waves_22k, wave_lengths_22k, se_batch, ref_se
        ).squeeze(1)

        if self.sr != 22050:
            converted_waves = torchaudio.functional.resample(converted_waves_22k, 22050, self.sr)
        else:
            converted_waves = converted_waves_22k

        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lengths_16k = (wave_lengths.float() * 16000 / self.sr).long()
        converted_waves_16k = torchaudio.functional.resample(converted_waves, self.sr, 16000)

        # extract S_alt (perturbed speech tokens)
        S_ori = self.semantic_fn(waves_16k)
        S_alt = self.semantic_fn(converted_waves_16k)

        if self.f0_condition:
            F0_ori = self.rmvpe.infer_from_audio_batch(waves_16k)
        else:
            F0_ori = None

        # interpolate speech token to match acoustic feature length
        alt_cond, _, alt_codes, alt_commitment_loss, alt_codebook_loss = (
            self.model.length_regulator(S_alt, ylens=target_lengths, f0=F0_ori)
        )
        ori_cond, _, ori_codes, ori_commitment_loss, ori_codebook_loss = (
            self.model.length_regulator(S_ori, ylens=target_lengths, f0=F0_ori)
        )
        if alt_commitment_loss is None:
            alt_commitment_loss = 0
            alt_codebook_loss = 0
            ori_commitment_loss = 0
            ori_codebook_loss = 0

        # randomly set a length as prompt
        prompt_len_max = target_lengths - 1
        prompt_len = (torch.rand([B], device=alt_cond.device) * prompt_len_max).floor().long()
        prompt_len[torch.rand([B], device=alt_cond.device) < 0.1] = 0

        # for prompt cond token, use ori_cond instead of alt_cond
        cond = alt_cond.clone()
        for bib in range(B):
            cond[bib, :prompt_len[bib]] = ori_cond[bib, :prompt_len[bib]]

        # diffusion target
        common_min_len = min(target_size, cond.size(1))
        target = target[:, :, :common_min_len]
        cond = cond[:, :common_min_len]
        target_lengths = torch.clamp(target_lengths, max=common_min_len)
        x = target

        # style vectors are extracted from the prompt only
        feat_list = []
        for bib in range(B):
            feat = kaldi.fbank(
                waves_16k[bib:bib + 1, :wave_lengths_16k[bib]],
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        y_list = []
        with torch.no_grad():
            for feat in feat_list:
                y = self.sv_fn(feat.unsqueeze(0))
                y_list.append(y)
        y = torch.cat(y_list, dim=0)

        loss, _ = self.model.cfm(x, target_lengths, prompt_len, cond, y)

        loss_total = (
            loss +
            (alt_commitment_loss + ori_commitment_loss) * 0.05 +
            (ori_codebook_loss + alt_codebook_loss) * 0.15
        )

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.cfm.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.length_regulator.parameters(), 10.0)
        self.optimizer.step('cfm')
        self.optimizer.step('length_regulator')
        self.optimizer.scheduler(key='cfm')
        self.optimizer.scheduler(key='length_regulator')

        return loss.detach().item()
    

    @torch.no_grad()
    def inference(self, source_path, target_path, output_dir):
        """Run voice conversion inference.
        
        Args:
            source_path (str): Path to source audio file
            target_path (str): Path to target audio file
            output_dir (str): Directory to save output files
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file not found: {target_path}")
            
        fp16 = True
        def crossfade(chunk1, chunk2, overlap):
            fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
            fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
            if len(chunk2) < overlap:
                chunk2[:overlap] = (
                    chunk2[:overlap] * fade_in[: len(chunk2)]
                    + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
                )
            else:
                chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
            return chunk2

        # Load and validate audio files
        try:
            source_audio, source_sr = librosa.load(source_path, sr=self.sr)
            if source_sr != self.sr:
                self.logger.warning(f"Source audio sample rate {source_sr}Hz resampled to {self.sr}Hz")
        except Exception as e:
            raise RuntimeError(f"Error loading source audio: {str(e)}")

        try:
            ref_audio, ref_sr = librosa.load(target_path, sr=self.sr)
            if ref_sr != self.sr:
                self.logger.warning(f"Reference audio sample rate {ref_sr}Hz resampled to {self.sr}Hz")
        except Exception as e:
            raise RuntimeError(f"Error loading reference audio: {str(e)}")

        # Check audio lengths
        max_duration = 30  # 30 seconds
        if len(source_audio) > max_duration * self.sr:
            self.logger.warning(f"Source audio truncated from {len(source_audio)/self.sr:.1f}s to {max_duration}s")
            source_audio = source_audio[:max_duration * self.sr]
        if len(ref_audio) > max_duration * self.sr:
            self.logger.warning(f"Reference audio truncated from {len(ref_audio)/self.sr:.1f}s to {max_duration}s")
            ref_audio = ref_audio[:max_duration * self.sr]

        diffusion_steps = 30
        length_adjust = 1.0
        inference_cfg_rate = 0.7

        max_context_window = self.sr // self.hop_length * 30
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * self.hop_length

        # Process audio
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref_audio[: self.sr * 25]).unsqueeze(0).float().to(self.device)

        time_vc_start = time.time()
        try:
            # Resample
            converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)
            # if source audio less than 30 seconds, whisper can handle in one forward
            if converted_waves_16k.size(-1) <= 16000 * 30:
                S_alt = self.semantic_fn(converted_waves_16k)
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
                    S_alt = self.semantic_fn(chunk)
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

            ori_waves_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
            S_ori = self.semantic_fn(ori_waves_16k)

            mel = self.mel_function(source_audio.to(self.device).float())
            mel2 = self.mel_function(ref_audio.to(self.device).float())

            target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

            feat2 = torchaudio.compliance.kaldi.fbank(
                ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = self.campplus_model(feat2.unsqueeze(0))

            F0_ori = None
            F0_alt = None
            shifted_f0_alt = None

            # Length regulation
            cond, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
            )
            prompt_condition, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
            )

            max_source_window = max_context_window - mel2.size(2)
            # split source condition (cond) into chunks
            processed_frames = 0
            generated_wave_chunks = []
            # generate chunk by chunk and stream the output
            while processed_frames < cond.size(1):
                chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
                is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                with torch.autocast(
                    device_type=torch.device(self.device).type, dtype=torch.float16 if fp16 else torch.float32
                ):
                    # Voice Conversion
                    vc_target = self.model.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                        mel2,
                        style2,
                        None,
                        diffusion_steps,
                        inference_cfg_rate=inference_cfg_rate,
                    )
                    vc_target = vc_target[:, :, mel2.size(-1):].clone()
                vc_wave = self.vocoder_fn(vc_target.float()).squeeze()
                vc_wave = vc_wave[None, :]
                if processed_frames == 0:
                    if is_last_chunk:
                        output_wave = vc_wave[0].cpu().numpy()
                        generated_wave_chunks.append(output_wave)
                        break
                    output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    previous_chunk = vc_wave[0, -overlap_wave_len:]
                    processed_frames += vc_target.size(2) - overlap_frame_len
                elif is_last_chunk:
                    output_wave = crossfade(
                        previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
                    )
                    generated_wave_chunks.append(output_wave)
                    processed_frames += vc_target.size(2) - overlap_frame_len
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
            vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
            time_vc_end = time.time()
            rtf = (time_vc_end - time_vc_start) / vc_wave.size(-1) * self.sr
            self.logger.info(f"RTF: {rtf:.2f}")

            source_name = os.path.basename(source_path).split(".")[0]
            target_name = os.path.basename(target_path).split(".")[0]
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav",
            )
            torchaudio.save(output_path, vc_wave.cpu(), self.sr)
            self.logger.info(f"Saved output to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            raise

    def eval_one_loop(self, step):
        # eval model on several example to check the model progress
        save_path = os.path.join(self.log_dir, f'eval_step_{step}')
        os.makedirs(save_path, exist_ok=True)
        
        # Check if evaluation directory exists
        files_path = '../dataset/evaluation_samples'
        if not os.path.exists(files_path):
            self.logger.error(f"Evaluation directory {files_path} does not exist")
            return
            
        files = glob.glob(os.path.join(files_path, '*.wav'))
        if not files:
            self.logger.error(f"No wav files found in {files_path}")
            return
            
        reference_files = [f for f in files if 'ref' in f]
        if not reference_files:
            self.logger.error("No reference files found")
            return
            
        samples = [f for f in files if 'ref' not in f]
        if not samples:
            self.logger.error("No sample files found")
            return
            
        self.logger.info(f"Starting evaluation at step {step}")
        for sample in samples:
            try:
                self.inference(
                    source_path=sample,
                    target_path=reference_files[0],
                    output_dir=save_path
                )
            except Exception as e:
                self.logger.error(f"Error processing {sample}: {str(e)}")
                continue
        self.logger.info(f"Completed evaluation at step {step}")

    def train_one_epoch(self):
        _ = [self.model[key].train() for key in self.model]
        for i, batch in enumerate(tqdm(self.train_dataloader)):
            batch = [b.to(self.device) for b in batch]
            loss = self.train_one_step(batch)
            self.ema_loss = (
                self.ema_loss * self.loss_smoothing_rate + loss * (1 - self.loss_smoothing_rate)
                if self.iters > 0 else loss
            )
            if self.iters % self.log_interval == 0:
                log_message = f"epoch {self.epoch}, step {self.iters}, loss: {self.ema_loss:.6f}"
                print(log_message)
                self.logger.info(log_message)

            if self.iters % self.eval_interval == 0:
                print('Evaluating..')
                self.eval_one_loop(self.iters)

            self.iters += 1

            if self.iters >= self.max_steps:
                break

            if self.iters % self.save_interval == 0:
                print('Saving..')
                self.logger.info(f"Saving checkpoint at epoch {self.epoch}, step {self.iters}")
                state = {
                    'net': {key: self.model[key].state_dict() for key in self.model},
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.optimizer.scheduler_state_dict(),
                    'iters': self.iters,
                    'epoch': self.epoch,
                }
                save_path = os.path.join(
                    self.log_dir,
                    f'DiT_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth'
                )
                torch.save(state, save_path)
                self.logger.info(f"Checkpoint saved to {save_path}")

                # find all checkpoints and remove old ones
                checkpoints = glob.glob(os.path.join(self.log_dir, 'DiT_epoch_*.pth'))
                if len(checkpoints) > 2:
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    for cp in checkpoints[:-2]:
                        os.remove(cp)
                        self.logger.info(f"Removed old checkpoint: {cp}")

    def train(self):
        self.ema_loss = 0
        self.loss_smoothing_rate = 0.99
        self.logger.info(f"Starting training for {self.n_epochs} epochs or {self.max_steps} steps")
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch}")
            self.train_one_epoch()
            self.logger.info(f"Completed epoch {epoch}")
            if self.iters >= self.max_steps:
                self.logger.info(f"Reached max steps {self.max_steps}, stopping training")
                break

        self.logger.info('Saving final model..')
        state = {
            'net': {key: self.model[key].state_dict() for key in self.model},
        }
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, 'ft_model.pth')
        torch.save(state, save_path)
        self.logger.info(f"Final model saved at {save_path}")


def main(args):
    trainer = Trainer(
        config_path=args.config,
        pretrained_ckpt_path=args.pretrained_ckpt,
        data_dir=args.dataset_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_interval=args.save_every,
        num_workers=args.num_workers,
        device=args.device
    )
    trainer.train()
    
if __name__ == '__main__':
    if sys.platform == 'win32':
        mp.freeze_support()
        mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml')
    parser.add_argument('--pretrained-ckpt', type=str, default=None)
    parser.add_argument('--dataset-dir', type=str, default='/path/to/dataset')
    parser.add_argument('--run-name', type=str, default='my_run')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    if torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = f"cuda:{args.gpu}" if args.gpu else "cuda:0"
    main(args)
