import logging
import math

import torch
from torch import nn

from .loss import Losses
from .mlp import MLP
from .utils import extract

log = logging.getLogger(__name__)


class Diffusion(nn.Module):
    def __init__(
        self,
        loss_type,
        beta_schedule="linear",
        clip_denoised=True,
        predict_epsilons=True,
        device="cuda:0",
        **kwargs,
    ):
        super(Diffusion, self).__init__()

        log.info("Using device: %s", device)

        self.loss_type = loss_type
        self.beta_schedule = beta_schedule
        self.clip_denoised = clip_denoised
        self.device = device
        self.predict_epsilons = predict_epsilons

        self.state_dim = kwargs["state_dim"]
        self.action_dim = kwargs["action_dim"]
        self.diffusion_steps = kwargs["diffusion_steps"]
        self.hidden_dim = kwargs["hidden_dim"]

        self.model = MLP(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            device=self.device,
            # t_dim=16,
        )

        if beta_schedule == "linear":
            betas = torch.linspace(
                0.0001,
                0.02,
                self.diffusion_steps,
                device=self.device,
                dtype=torch.float32,
            )
        elif beta_schedule == "cosine":
            betas = torch.linspace(
                0.0001,
                0.02,
                self.diffusion_steps,
                device=self.device,
                dtype=torch.float32,
            )
            betas = betas * 0.5 * (1.0 - torch.cos(betas / betas[-1] * math.pi))
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # [1, 2, 3] -> [1, 1*2, 1*2*3]
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Forwards
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

        # Backwards
        posterior_variance = betas * (1.0 - alphas_cumprod) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recipm_alphas_cumprod", (1.0 / alphas_cumprod - 1).sqrt())

        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()

    def sample(self, state, *args, **kwargs):
        """
        Samples an action based on the given state.
        Args:
            state (torch.Tensor): The current state tensor with shape (batch_size, ...).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: The sampled action tensor with values clamped between -1 and 1.
        """
        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim]

        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-1, 1)

    def p_sample_loop(self, state, shape, *args, **kwargs) -> torch.Tensor:
        """
        Generates samples by iteratively applying the p_sample function in reverse order of diffusion steps.
        Args:
            state (torch.Tensor): The initial state tensor.
            shape (tuple): The shape of the tensor to be generated.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: The generated sample tensor.
        """
        _ = args, kwargs
        device = self.device
        batch_size = state.shape[0]
        x = torch.randn(shape, device=device, requires_grad=False)

        for i in reversed(range(0, self.diffusion_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state)

        return x

    def p_sample(self, x, t, state) -> torch.Tensor:
        """
        Generate a sample from the diffusion model at a given timestep.
        Args:
            x (torch.Tensor): The input tensor representing the current state.
            t (torch.Tensor): The current timestep tensor.
            state (Any): Additional state information required by the model.
        Returns:
            torch.Tensor: The sampled tensor from the diffusion model.
        """
        batch_size, *_ = x.shape
        model_mean, model_log_variance = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def p_mean_variance(self, x, t, state):
        """
        Compute the mean and log variance of the posterior distribution for the given input.
        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time step tensor.
            state (Any): Additional state information required by the model.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and log variance of the posterior
            distribution.
        """
        # batch_size, *_, device = x.shape, x.device

        pred_noise = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        x_recon.clamp_(-1, 1)

        model_mean, _, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    def predict_start_from_noise(self, x, t, pred_noise) -> torch.Tensor:
        """
        Predicts the starting point from the given noise.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time step tensor.
            pred_noise (torch.Tensor): The predicted noise tensor.

        Returns:
            torch.Tensor: The predicted starting point tensor.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * pred_noise
        )

    def q_posterior(self, x0, xt, t):
        r"""Backward pass of the diffusion model. Compute the posterior mean, variance, and log variance for the given
        inputs: $q(x_{t-1} | x_t, x_0)$.
        $$ q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \mu_{t-1}, \sigma^2_{t-1}) $$
        Args:
            x_recon (torch.Tensor): The reconstructed input tensor.
            x (torch.Tensor): The original input tensor.
            t (torch.Tensor): The time step tensor.
        Returns:
            tuple: A tuple containing:
                - posterior_mean (torch.Tensor): The computed posterior mean.
                - posterior_variance (torch.Tensor): The computed posterior variance.
                - posterior_log_variance (torch.Tensor): The computed posterior log variance.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, xt.shape) * x0 + extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )
        posterior_variance = extract(self.posterior_variance, t, xt.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, xt.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    # ------------------------------ training ------------------------------ #

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)

    def p_losses(self, x_start, state, t, weights=1.0):
        """
        Compute the loss for the given inputs.
        Args:
            x (torch.Tensor): The input tensor.
            state (Any): Additional state information required by the model.
            t (torch.Tensor): The time step tensor.
            weights (float): The weight to apply to the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        noise = torch.randn_like(x_start)
        x_noise = self.q_sample(x_start, t, noise)
        x_recon = self.model(x_noise, t, state)
        assert x_recon.shape == x_start.shape

        if self.predict_epsilons:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def q_sample(self, x0, t, noise=None):
        r"""Forward pass of the diffusion model. Generates a sample from the diffusion process at a given timestep.
        $$
            q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_0, (1-\alpha_t)I)
        $$

        Args:
            x0 (torch.Tensor): The initial data tensor. [batch_size, ...]
            t (torch.Tensor): The current timestep tensor. [batch_size]
            noise (torch.Tensor, optional): Optional noise tensor. If not provided,
                                            random noise will be generated. [batch_size, ...]

        Returns:
            torch.Tensor: The generated sample tensor. [batch_size, ...]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )
        return sample
