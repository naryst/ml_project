from abc import ABC

import torch
import torch.nn.functional as F

from modules.diffusion_transformer import DiT
from modules.commons import sequence_mask

from tqdm import tqdm

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def _solve_euler_loop(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate, store_trajectory_clones=False, use_tqdm=True):
        """
        Core Euler solver loop. Conditionally stores cloned states for trajectory.
        """
        current_t = t_span[0]
        
        trajectory = []
        if store_trajectory_clones:
            trajectory.append(x.clone()) # Store initial state clone

        # Apply prompt conditioning to initial state `x`
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0 
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        if use_tqdm:
            loop = tqdm(range(1, len(t_span)), desc="Euler ODE Solve")
        else:
            loop = range(1, len(t_span))
        for step_idx in loop:
            dt = t_span[step_idx] - t_span[step_idx - 1]
            # Use the time at the beginning of the interval for the estimator
            time_for_estimator = t_span[step_idx - 1].expand(x.size(0)) if x.ndim > 0 else t_span[step_idx - 1]

            if inference_cfg_rate > 0:
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t_for_estimator = torch.cat([time_for_estimator, time_for_estimator], dim=0)
                
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens.repeat(2) if x_lens.ndim > 0 else x_lens , stacked_t_for_estimator, 
                    stacked_style, stacked_mu,
                )
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, time_for_estimator, style, mu)

            x = x + dt * dphi_dt
            current_t = t_span[step_idx]

            if store_trajectory_clones:
                trajectory.append(x.clone())
            
            # Ensure prompt region in x remains zero for the next step
            x[..., :prompt_len] = 0

        if store_trajectory_clones:
            return trajectory
        else:
            return x

    def solve_euler_with_trajectory(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5, use_tqdm=True):
        """
        Fixed Euler solver for ODEs that returns the entire trajectory of cloned states.
        """
        return self._solve_euler_loop(x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate, store_trajectory_clones=True, use_tqdm=use_tqdm)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        """
        Fixed Euler solver for ODEs that returns only the final state.
        Does not incur the cost of cloning the trajectory.
        """
        return self._solve_euler_loop(x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate, store_trajectory_clones=False)

    def forward(self, x1, x_lens, prompt_lens, mu, style):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, :prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens)
        loss = 0
        for bib in range(b):
            loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z



class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(
            args
        )
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")
