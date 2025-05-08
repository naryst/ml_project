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

from modules.commons import recursive_munch, build_model, load_checkpoint, sequence_mask
from optimizers import build_optimizer
from data.ft_dataset import build_ft_dataloader
from models_loading import load_models
from hf_utils import load_custom_model_from_hf

def trajectory_distillation_loss(student_trajectory, teacher_trajectory, mask, trajectory_weights=None):
    """Calculate loss between student and teacher trajectories.
    
    Args:
        student_trajectory: List of student states
        teacher_trajectory: List of teacher states
        mask: Sequence mask for variable lengths
        trajectory_weights: Optional weights for different timesteps
    """
    if trajectory_weights is None:
        # Default: linear weighting that puts more emphasis on later timesteps
        trajectory_weights = torch.linspace(0.1, 1.0, len(student_trajectory), device=student_trajectory[0].device)
    
    total_loss = 0
    for i, (student_state, teacher_state) in enumerate(zip(student_trajectory, teacher_trajectory)):
        # Calculate MSE loss for this timestep
        step_loss = torch.nn.functional.mse_loss(
            student_state * mask,
            teacher_state * mask
        )
        # Apply weight for this timestep
        total_loss += trajectory_weights[i] * step_loss
    
    return total_loss / len(student_trajectory)

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
                 trajectory_weight_type="linear"
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
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        self.logger.info(f"Starting distillation process")
        self.logger.info(f"Teacher config: {teacher_config_path}")
        self.logger.info(f"Student config: {student_config_path}")
        self.logger.info(f"Device: {device}")
        
        # Load configs
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
                inference_cfg_rate=0.7
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
            inference_cfg_rate=0.7
        )
        
        # Calculate trajectory weights based on specified type
        if self.trajectory_weight_type == "linear":
            trajectory_weights = torch.linspace(0.1, 1.0, len(student_trajectory), device=self.device)
        elif self.trajectory_weight_type == "exponential":
            trajectory_weights = torch.exp(torch.linspace(-2, 0, len(student_trajectory), device=self.device))
        else:  # uniform
            trajectory_weights = torch.ones(len(student_trajectory), device=self.device)
        
        # Calculate loss
        if self.use_trajectory_loss:
            loss = trajectory_distillation_loss(
                student_trajectory,
                teacher_trajectory,
                mask,
                trajectory_weights
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
            (alt_commitment_loss + ori_commitment_loss) * 0.05 +
            (ori_codebook_loss + alt_codebook_loss) * 0.15
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

    def run_distillation(self):
        """Run the progressive distillation process."""
        # Progressive distillation loop
        current_teacher_steps = self.initial_teacher_steps
        iteration = 0
        
        while current_teacher_steps > self.final_steps:
            self.logger.info(f"\nStarting iteration {iteration + 1}")
            self.logger.info(f"Current teacher steps: {current_teacher_steps}")
            
            # Load or initialize teacher model
            if iteration == 0:
                # Load initial teacher model
                teacher_model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = load_models(
                    fp16=False,
                    f0_condition=self.teacher_config["model_params"]["DiT"].get("f0_condition", False),
                    checkpoint=self.teacher_checkpoint_path,
                    config=os.path.join(self.output_dir, "student_config.yaml"),
                )
            else:
                # Use previous student as new teacher
                teacher_model = self.student_model
                # Reinitialize student model
                self.student_model = build_model(self.student_params, stage="DiT")
                self.student_model = {k: v.to(self.device) for k, v in self.student_model.items()}
                self.student_model.cfm.estimator.setup_caches(max_batch_size=self.batch_size, max_seq_length=8192)
            
            _ = [teacher_model[key].to(self.device) for key in teacher_model]
            teacher_model.cfm.estimator.setup_caches(max_batch_size=self.batch_size, max_seq_length=8192)
            
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
                    torch.nn.utils.clip_grad_norm_(self.student_model.cfm.parameters(), 10.0)
                    torch.nn.utils.clip_grad_norm_(self.student_model.length_regulator.parameters(), 10.0)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_config", type=str, required=True)
    parser.add_argument("--student_config", type=str, required=True)
    parser.add_argument("--initial_teacher_checkpoint", type=str, required=True)
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
        trajectory_weight_type=args.trajectory_weight_type
    )
    
    distiller.run_distillation()

if __name__ == "__main__":
    main()