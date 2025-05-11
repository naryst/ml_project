import os
import torch
import yaml
import argparse
from tqdm import tqdm
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import logging
from datetime import datetime
import numpy as np
import gc
from torch.amp import autocast, GradScaler
import copy

from modules.commons import recursive_munch, build_model, load_checkpoint, sequence_mask
from optimizers import build_optimizer
from data.ft_dataset import build_ft_dataloader
from models_loading import load_models
from hf_utils import load_custom_model_from_hf

def trajectory_distillation_loss(student_trajectory, teacher_trajectory, mask, trajectory_weights=None, weight_type="linear"):
    """Calculate loss between student and teacher trajectories.
    
    Args:
        student_trajectory: List of student states
        teacher_trajectory: List of teacher states
        mask: Sequence mask for variable lengths
        trajectory_weights: Optional weights for different timesteps
        weight_type: Type of weighting to use if trajectory_weights is None ("linear", "exponential", "uniform")
    """
    student_len = len(student_trajectory)
    teacher_len = len(teacher_trajectory)
    
    if trajectory_weights is None:
        # Create weights based on specified type
        if weight_type == "linear":
            trajectory_weights = torch.linspace(0.1, 1.0, student_len, device=student_trajectory[0].device)
        elif weight_type == "exponential":
            trajectory_weights = torch.exp(torch.linspace(-2, 0, student_len, device=student_trajectory[0].device))
        else:  # uniform
            trajectory_weights = torch.ones(student_len, device=student_trajectory[0].device)
    
    # If trajectories have different lengths, we need to interpolate teacher states
    total_loss = 0
    if student_len != teacher_len:
        # Map student timesteps to corresponding teacher timesteps
        teacher_indices = torch.linspace(0, teacher_len-1, student_len).long().tolist()
        for i, (student_state, teacher_idx) in enumerate(zip(student_trajectory, teacher_indices)):
            # Use the closest teacher state
            teacher_state = teacher_trajectory[teacher_idx]
            
            # Calculate MSE loss
            step_loss = torch.nn.functional.mse_loss(
                student_state * mask,
                teacher_state * mask,
                reduction="mean"
            )
            
            # Apply weight for this timestep
            total_loss += trajectory_weights[i] * step_loss
    else:
        # Same length trajectories - direct comparison
        for i, (student_state, teacher_state) in enumerate(zip(student_trajectory, teacher_trajectory)):
            step_loss = torch.nn.functional.mse_loss(
                student_state * mask,
                teacher_state * mask,
                reduction="mean"
            )
            total_loss += trajectory_weights[i] * step_loss
    
    # Normalize by sum of weights to maintain scale regardless of number of steps
    return total_loss / trajectory_weights.sum()

class Distiller:
    def __init__(self,
                 teacher_config_path,
                 student_config_path,
                 teacher_checkpoint_path,
                 data_dir,
                 output_dir,
                 initial_teacher_steps=30,
                 final_steps=8,
                 steps_reduction_factor=2,
                 epochs_per_iteration=10,
                 batch_size=4,
                 num_workers=4,
                 save_interval=1000,
                 device="cuda",
                 use_trajectory_loss=True,
                 trajectory_weight_type="linear",
                 iterations_per_epoch=1000,
                 commitment_loss_weight=0.05,
                 codebook_loss_weight=0.15,
                 grad_clip_threshold=10.0,
                 checkpoint_cleanup=False,
                 cleanup_keep_last=3,
                 use_amp=True,
                 ):
        self.wav2vec_feature_extractor = None
        self.semantic_fn = None
        self.whisper_feature_extractor = None
        self.vocoder_fn = None
        self.whisper_model = None
        self.se_db = None
        self.f0_fn = None
        self.rmvpe = None
        self.tone_color_converter = None
        self.bigvgan_model = None
        self.wav2vec_model = None
        self.hift_gen = None
        self.campplus_model = None
        self.logger = None
        self.sv_fn = None
        self.device = device
        self.initial_teacher_steps = initial_teacher_steps
        self.final_steps = final_steps
        self.steps_reduction_factor = steps_reduction_factor
        self.epochs_per_iteration = epochs_per_iteration
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.use_trajectory_loss = use_trajectory_loss
        self.trajectory_weight_type = trajectory_weight_type
        self.eval_steps = 1
        self.iterations_per_epoch = iterations_per_epoch
        self.commitment_loss_weight = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.grad_clip_threshold = grad_clip_threshold
        self.checkpoint_cleanup = checkpoint_cleanup
        self.cleanup_keep_last = cleanup_keep_last
        self.scaler = GradScaler() if device == "cuda" and use_amp else None
        self.use_amp = device == "cuda" and use_amp
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        if not self.logger is None:
            self.logger.info(f"Starting distillation process")
            self.logger.info(f"Teacher config: {teacher_config_path}")
            self.logger.info(f"Student config: {student_config_path}")
            self.logger.info(f"Device: {device}")
        else:
            raise NotImplementedError('logger not initialized')
        
        # Load configs
        self.teacher_config_path = teacher_config_path
        self.student_config_path = student_config_path
        self.teacher_config = yaml.safe_load(open(teacher_config_path, "r"))
        self.student_config = yaml.safe_load(open(student_config_path, "r"))
        
        # Save the student config for later reference
        with open(os.path.join(self.output_dir, "student_config.yaml"), "w") as f:
            yaml.dump(self.student_config, f)
        
        # Setup data loader
        preprocess_params = self.teacher_config["preprocess_params"]
        self.sr = preprocess_params.get("sr", 22050)
        self.hop_length = preprocess_params['spect_params'].get('hop_length', 256)
        self.win_length = preprocess_params['spect_params'].get('win_length', 1024)
        self.n_fft = preprocess_params['spect_params'].get('n_fft', 1024)
        
        self.train_dataloader = build_ft_dataloader(
            data_dir,
            preprocess_params["spect_params"],
            self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        # Setup mel function
        from modules.audio import mel_spectrogram
        self.mel_fn_args = {
            "n_fft": self.teacher_config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": self.teacher_config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": self.hop_length,
            "num_mels": self.teacher_config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": self.teacher_config["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None
            if self.teacher_config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
            else 8000,
            "center": False,
        }
        self.mel_function = lambda x: mel_spectrogram(x, **self.mel_fn_args)
        
        # Check if f0 conditioning is used
        self.f0_condition = self.teacher_config['model_params']['DiT'].get('f0_condition', False)
        
        # Build necessary models
        self.build_sv_model(device)
        self.build_semantic_fn(device)
        if self.f0_condition:
            self.build_f0_fn(device)
        self.build_converter(device)
        self.build_vocoder(device)
        
        # Load teacher checkpoint (will be done during first iteration)
        self.teacher_checkpoint_path = teacher_checkpoint_path
        
        # Initialize student model
        self.student_params = recursive_munch(self.student_config["model_params"])
        self.student_model = build_model(self.student_params, stage="DiT")
        _ = [self.student_model[key].to(device) for key in self.student_model]
        self.student_model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)

    def setup_logging(self):
        """Setup logging to file and console"""
        self.logger = logging.getLogger('seed_vc_distiller')
        self.logger.setLevel(logging.INFO)
        
        # Create a file handler for logging to a file
        log_file = os.path.join(self.output_dir, f'distillation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
        
    def build_sv_model(self, device):
        from modules.campplus.DTDNN import CAMPPlus
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_sd_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus_sd = torch.load(campplus_sd_path, map_location='cpu')
        self.campplus_model.load_state_dict(campplus_sd)
        self.campplus_model.eval()
        self.campplus_model.to(device)
        self.sv_fn = self.campplus_model

    def build_f0_fn(self, device):
        from modules.rmvpe import RMVPE
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=device)
        self.f0_fn = self.rmvpe

    def build_converter(self, device):
        from modules.openvoice.api import ToneColorConverter
        ckpt_converter, config_converter = load_custom_model_from_hf("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json")
        self.tone_color_converter = ToneColorConverter(config_converter, device=device)
        self.tone_color_converter.load_ckpt(ckpt_converter)
        self.tone_color_converter.model.eval()
        se_db_path = load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt", None)
        self.se_db = torch.load(se_db_path, map_location='cpu')

    def build_vocoder(self, device):
        vocoder_type = self.teacher_config['model_params']['vocoder']['type']
        vocoder_name = self.teacher_config['model_params']['vocoder'].get('name', None)
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

    def build_semantic_fn(self, device):
        speech_tokenizer_type = self.teacher_config['model_params']['speech_tokenizer'].get('type', 'cosyvoice')
        if speech_tokenizer_type == 'whisper':
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_model_name = self.teacher_config['model_params']['speech_tokenizer']['name']
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

            model_name = self.teacher_config['model_params']['speech_tokenizer']['name']

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
            model_name = self.teacher_config['model_params']['speech_tokenizer']['name']
            output_layer = self.teacher_config['model_params']['speech_tokenizer']['output_layer']
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
    
    def distill_one_step(self, batch, teacher_model, student_model, teacher_steps, student_steps):
        """Train the student model for one batch using the teacher model."""
        waves, mels, wave_lengths, mel_input_length = batch
        
        B = waves.size(0)
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
            F0_ori = self.f0_fn.infer_from_audio_batch(waves_16k)
        else:
            F0_ori = None

        # interpolate speech token to match acoustic feature length
        alt_cond, _, alt_codes, alt_commitment_loss, alt_codebook_loss = (
            teacher_model.length_regulator(S_alt, ylens=target_lengths, f0=F0_ori)
        )
        ori_cond, _, ori_codes, ori_commitment_loss, ori_codebook_loss = (
            teacher_model.length_regulator(S_ori, ylens=target_lengths, f0=F0_ori)
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

        # Create sequence mask
        mask = sequence_mask(target_lengths).unsqueeze(1).to(self.device)
        
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
        
        # Generate teacher trajectory
        with torch.no_grad():
            teacher_t_span = torch.linspace(0, 1, teacher_steps + 1, device=self.device)
            teacher_trajectory = teacher_model.cfm.solve_euler_with_trajectory(
                x=torch.randn_like(target),
                x_lens=target_lengths,
                prompt=prompt_len,
                mu=cond,
                style=y,
                f0=F0_ori,
                t_span=teacher_t_span,
                inference_cfg_rate=0.7,
                use_tqdm=False
            )
        
        # Generate student trajectory
        student_t_span = torch.linspace(0, 1, student_steps + 1, device=self.device)
        student_trajectory = student_model.cfm.solve_euler_with_trajectory(
            x=torch.randn_like(target),
            x_lens=target_lengths,
            prompt=prompt_len,
            mu=cond,
            style=y,
            f0=F0_ori,
            t_span=student_t_span,
            inference_cfg_rate=0.7,
            use_tqdm=False
        )
        
        # Calculate loss
        if self.use_trajectory_loss:
            loss = trajectory_distillation_loss(
                student_trajectory,
                teacher_trajectory,
                mask,
                trajectory_weights=None,
                weight_type=self.trajectory_weight_type
            )
        else:
            # Just match final outputs
            loss = torch.nn.functional.mse_loss(
                student_trajectory[-1] * mask,
                teacher_trajectory[-1] * mask
            )
            
        # Add vector quantization losses for consistency with train.py
        loss_total = (
            loss +
            (alt_commitment_loss + ori_commitment_loss) * self.commitment_loss_weight +
            (ori_codebook_loss + alt_codebook_loss) * self.codebook_loss_weight
        )
            
        return loss_total
        
    def clean_cache(self):
        """Clean CUDA/MPS cache after each iteration."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        # Clear any cached memory
        gc.collect()

    @torch.no_grad()
    def inference_student(self, source_path, target_path, output_dir, diffusion_steps):
        import librosa
        import time
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
            self.logger.warning(f"Source audio truncated from {len(source_audio) / self.sr:.1f}s to {max_duration}s")
            source_audio = source_audio[:max_duration * self.sr]
        if len(ref_audio) > max_duration * self.sr:
            self.logger.warning(f"Reference audio truncated from {len(ref_audio) / self.sr:.1f}s to {max_duration}s")
            ref_audio = ref_audio[:max_duration * self.sr]

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
                                :, traversed_time: traversed_time + 16000 * 30
                                ]
                    else:
                        chunk = torch.cat(
                            [
                                buffer,
                                converted_waves_16k[
                                :,
                                traversed_time: traversed_time
                                                + 16000 * (30 - overlapping_time),
                                ],
                            ],
                            dim=-1,
                        )
                    S_alt = self.semantic_fn(chunk)
                    if traversed_time == 0:
                        S_alt_list.append(S_alt)
                    else:
                        S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                    buffer = chunk[:, -16000 * overlapping_time:]
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
            cond, _, codes, commitment_loss, codebook_loss = self.student_model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
            )
            prompt_condition, _, codes, commitment_loss, codebook_loss = self.student_model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
            )

            max_source_window = max_context_window - mel2.size(2)
            # split source condition (cond) into chunks
            processed_frames = 0
            generated_wave_chunks = []
            # generate chunk by chunk and stream the output
            while processed_frames < cond.size(1):
                chunk_cond = cond[:, processed_frames: processed_frames + max_source_window]
                is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                with autocast(device_type='cuda', enabled=self.use_amp):
                    # Voice Conversion
                    vc_target = self.student_model.cfm.inference(
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

    def eval_one_loop(self, step, diffusion_steps):
        import glob
        save_path = os.path.join(self.output_dir, f'eval_step_{step}')
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
                self.inference_student(
                    source_path=sample,
                    target_path=reference_files[0],
                    output_dir=save_path,
                    diffusion_steps=diffusion_steps,
                )
            except Exception as e:
                self.logger.error(f"Error processing {sample}: {str(e)}")
                continue
        self.logger.info(f"Completed evaluation at step {step}")


    def run_distillation(self):
        """Run the progressive distillation process."""
        current_teacher_steps = self.initial_teacher_steps
        iteration = 0
        
        while current_teacher_steps > self.final_steps:
            self.logger.info(f"\nStarting iteration {iteration + 1}")
            self.logger.info(f"Current teacher steps: {current_teacher_steps}")
            
            if iteration == 0:
                # Load initial teacher model
                teacher_model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = load_models(
                    fp16=False,
                    f0_condition=self.teacher_config["model_params"]["DiT"].get("f0_condition", False),
                    checkpoint=self.teacher_checkpoint_path,
                    config=self.teacher_config_path,
                )
                
                # Initialize student model with teacher's weights
                self.student_model, _, _, _, _, _, _ = load_models(
                    fp16=False,
                    f0_condition=self.student_config["model_params"]["DiT"].get("f0_condition", False),
                    checkpoint=self.teacher_checkpoint_path,  # Use teacher's checkpoint
                    config=self.student_config_path,
                )
                
                # Move models to device (do this only once)
                _ = [teacher_model[key].to(self.device) for key in teacher_model]
                _ = [self.student_model[key].to(self.device) for key in self.student_model]
                
            else:                
                # Create new teacher model with same architecture as student
                teacher_params = recursive_munch(self.teacher_config["model_params"])
                teacher_model = build_model(teacher_params, stage="DiT")
                _ = [teacher_model[key].to(self.device) for key in teacher_model]
                
                # Transfer weights from student to teacher via state dicts
                for key in self.student_model:
                    teacher_model[key].load_state_dict(self.student_model[key].state_dict())
            
            # Setup caches for models
            teacher_model.cfm.estimator.setup_caches(max_batch_size=self.batch_size, max_seq_length=8192)
            self.student_model.cfm.estimator.setup_caches(max_batch_size=self.batch_size, max_seq_length=8192)
            
            # Calculate student steps for this iteration
            current_student_steps = current_teacher_steps // self.steps_reduction_factor
            self.logger.info(f"Student steps for this iteration: {current_student_steps}")
            
            # Initialize optimizer for student
            optimizer = build_optimizer(
                {key: self.student_model[key] for key in self.student_model},
                lr=float(self.student_config.get("distillation_lr", 1e-5))
            )
            
            # Training loop for current iteration
            total_steps = 0
            ema_loss = 0
            loss_smoothing_rate = 0.99
            
            for epoch in range(self.epochs_per_iteration):
                self.logger.info(f"Starting epoch {epoch+1}/{self.epochs_per_iteration}")
                
                for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs_per_iteration}"):
                    # Move batch to device
                    batch = [b.to(self.device) for b in batch]
                    
                    # Forward pass and loss calculation
                    with autocast(device_type='cuda', enabled=self.use_amp):
                        loss = self.distill_one_step(
                            batch, 
                            teacher_model, 
                            self.student_model, 
                            current_teacher_steps, 
                            current_student_steps
                        )
                    
                    # Update student
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_model.cfm.parameters(), self.grad_clip_threshold)
                    torch.nn.utils.clip_grad_norm_(self.student_model.length_regulator.parameters(), self.grad_clip_threshold)
                    optimizer.step('cfm')
                    optimizer.step('length_regulator')
                    optimizer.scheduler(key='cfm')
                    optimizer.scheduler(key='length_regulator')
                    
                    # Update EMA loss
                    ema_loss = (
                        ema_loss * loss_smoothing_rate + loss.item() * (1 - loss_smoothing_rate)
                        if total_steps > 0 else loss.item()
                    )
                    
                    if total_steps % 10 == 0:
                        self.logger.info(f"Iteration {iteration+1}, Epoch {epoch+1}, Step {total_steps}, Loss: {ema_loss:.6f}")
                    
                    total_steps += 1
                    
                    # Clear CUDA cache periodically to prevent memory buildup
                    if total_steps % 100 == 0:
                        self.clean_cache()

                    # Evaluate if needed
                    if self.eval_steps > 0 and total_steps % self.eval_steps == 0:
                        self.logger.info(f"Evaluating at step {total_steps}")
                        self.eval_one_loop(total_steps, current_student_steps)

                    # Save checkpoint
                    if total_steps % self.save_interval == 0:
                        checkpoint_path = os.path.join(
                            self.output_dir, 
                            f"distilled_model_iter_{iteration}_epoch_{epoch}_step_{total_steps}.pth"
                        )
                        torch.save({
                            'iteration': iteration,
                            'epoch': epoch,
                            'step': total_steps,
                            'student_model': {k: v.state_dict() for k, v in self.student_model.items()},
                            'optimizer': optimizer.state_dict(),
                            'loss': loss.item(),
                            'teacher_steps': current_teacher_steps,
                            'student_steps': current_student_steps,
                            'use_trajectory_loss': self.use_trajectory_loss,
                            'trajectory_weight_type': self.trajectory_weight_type,
                        }, checkpoint_path)
                        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                        # Clean up old checkpoints to save disk space
                        self.clean_checkpoint_files(iteration, epoch, self.cleanup_keep_last)
                    
                    if 0 < self.iterations_per_epoch <= total_steps:
                        logging.info(f"Reached {self.iterations_per_epoch} iterations per epoch, stopping epoch")
                        break
                # Save at end of each epoch
                checkpoint_path = os.path.join(
                    self.output_dir, 
                    f"distilled_model_iter_{iteration}_epoch_{epoch+1}.pth"
                )
                torch.save({
                    'iteration': iteration,
                    'epoch': epoch+1,
                    'step': total_steps,
                    'student_model': {k: v.state_dict() for k, v in self.student_model.items()},
                    'optimizer': optimizer.state_dict(),
                    'loss': loss.item(),
                    'teacher_steps': current_teacher_steps,
                    'student_steps': current_student_steps,
                    'use_trajectory_loss': self.use_trajectory_loss,
                    'trajectory_weight_type': self.trajectory_weight_type,
                }, checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                # Clean up old checkpoints after epoch end
                self.clean_checkpoint_files(iteration, epoch+1, self.cleanup_keep_last)
            
            # Update steps for next iteration
            current_teacher_steps = current_student_steps
            iteration += 1
            
            # Save final model of this iteration
            final_path = os.path.join(
                self.output_dir,
                f"distilled_model_iter_{iteration}_final.pth"
            )
            torch.save({
                'iteration': iteration,
                'student_model': {k: v.state_dict() for k, v in self.student_model.items()},
                'teacher_steps': current_teacher_steps,
                'student_steps': current_student_steps,
                'use_trajectory_loss': self.use_trajectory_loss,
                'trajectory_weight_type': self.trajectory_weight_type,
            }, final_path)
            self.logger.info(f"Saved final model of iteration {iteration} to {final_path}")
            
            # Clean cache after each iteration
            self.clean_cache()

    def clean_checkpoint_files(self, iteration, epoch, keep_last=3):
        """Clean up old checkpoint files to save disk space.
        
        Args:
            iteration: Current iteration
            epoch: Current epoch
            keep_last: Number of most recent checkpoints to keep
        """
        if not hasattr(self, 'checkpoint_cleanup') or not self.checkpoint_cleanup:
            return
            
        self.logger.info(f"Cleaning up old checkpoint files, keeping last {keep_last}")
        
        # Get all checkpoint files for the current iteration
        import glob
        import os
        
        # Regular step checkpoints
        pattern = os.path.join(self.output_dir, f"distilled_model_iter_{iteration}_epoch_{epoch}_step_*.pth")
        checkpoint_files = glob.glob(pattern)
        
        # Sort by step number (extract from filename)
        checkpoint_files.sort(key=lambda x: int(x.split('_step_')[1].split('.pth')[0]))
        
        # Keep only the last N checkpoints
        if len(checkpoint_files) > keep_last:
            files_to_delete = checkpoint_files[:-keep_last]
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    self.logger.info(f"Deleted old checkpoint: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Error deleting {file_path}: {e}")
                    
        # Also clean up epoch checkpoints from previous epochs in this iteration
        if epoch > 0:
            pattern = os.path.join(self.output_dir, f"distilled_model_iter_{iteration}_epoch_*.pth")
            epoch_files = glob.glob(pattern)
            epoch_files = [f for f in epoch_files if f"_epoch_{epoch}" not in f]
            
            # Keep only the last N epoch checkpoints
            epoch_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
            if len(epoch_files) > keep_last:
                epoch_files_to_delete = epoch_files[:-keep_last]
                for file_path in epoch_files_to_delete:
                    try:
                        os.remove(file_path)
                        self.logger.info(f"Deleted old epoch checkpoint: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Error deleting {file_path}: {e}")
                        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_config", type=str, required=True)
    parser.add_argument("--student_config", type=str, required=True)
    parser.add_argument("--initial_teacher_checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--initial_teacher_steps", type=int, default=30)
    parser.add_argument("--final_steps", type=int, default=8)
    parser.add_argument("--steps_reduction_factor", type=int, default=2)
    parser.add_argument("--epochs_per_iteration", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_trajectory_loss", action="store_true", help="Use trajectory-based distillation")
    parser.add_argument("--trajectory_weight_type", type=str, default="linear", 
                      choices=["linear", "exponential", "uniform"],
                      help="Type of weighting for trajectory timesteps")
    parser.add_argument("--iterations_per_epoch", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=-1, help="Steps between evaluation runs")
    parser.add_argument("--commitment_loss_weight", type=float, default=0.05,
                        help="Weight for commitment loss")
    parser.add_argument("--codebook_loss_weight", type=float, default=0.15,
                        help="Weight for codebook loss")
    parser.add_argument("--grad_clip_threshold", type=float, default=10.0,
                        help="Threshold for gradient clipping")
    parser.add_argument("--checkpoint_cleanup", action="store_true",
                        help="Delete older checkpoints to save disk space")
    parser.add_argument("--cleanup_keep_last", type=int, default=3,
                        help="Number of recent checkpoints to keep when cleaning up")
    args = parser.parse_args()
    
    # Create the distiller and run distillation
    distiller = Distiller(
        teacher_config_path=args.teacher_config,
        student_config_path=args.student_config,
        teacher_checkpoint_path=args.initial_teacher_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        initial_teacher_steps=args.initial_teacher_steps,
        final_steps=args.final_steps,
        steps_reduction_factor=args.steps_reduction_factor,
        epochs_per_iteration=args.epochs_per_iteration,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        device=args.device,
        use_trajectory_loss=args.use_trajectory_loss,
        trajectory_weight_type=args.trajectory_weight_type,
        iterations_per_epoch=args.iterations_per_epoch,
        commitment_loss_weight=args.commitment_loss_weight,
        codebook_loss_weight=args.codebook_loss_weight,
        grad_clip_threshold=args.grad_clip_threshold,
        checkpoint_cleanup=args.checkpoint_cleanup,
        cleanup_keep_last=args.cleanup_keep_last,
    )
    
    distiller.eval_steps = args.eval_steps
    
    distiller.run_distillation()

if __name__ == "__main__":
    main()