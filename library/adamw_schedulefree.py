import torch
import torch.optim
import math
from typing import Callable

class AdamWScheduleFreeHybridAdopt(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW with advanced Cautious updates including a hybrid mode,
    and an option for ADOPT-style delayed denominator.

    This optimizer combines several advanced techniques:
    1.  **Schedule-Free**: Eliminates the need for a learning rate scheduler by
        using a weighted average of past parameter states (inspired by 'z' step).
    2.  **Cautious Updates**: Suppresses updates that are likely to be counter-
        productive by evaluating their alignment with the current gradient.
    3.  **Cautious Hybrid Mode**: A sophisticated cautious mechanism that blends two
        reliability scores for a more nuanced update suppression:
        - `score_cautious`: Spatial consistency (update direction vs. gradient direction).
        - `score_came`: Temporal consistency (update stability over time).
    4.  **ADOPT-style Denominator (Optional)**: Stabilizes updates by normalizing
        the current gradient with the variance from the *previous* step, reducing
        the impact of noisy gradients on the update magnitude.
    """
    def __init__(self,
                 params,
                 lr: float = 0.0025,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 # --- Cautious and Hybrid options ---
                 cautious: bool = False,
                 cautious_adaptive_rescale: bool = False,
                 cautious_adaptive_alpha: float = 0.1,
                 cautious_momentum_mask: bool = False,
                 cautious_beta_mask: float = 0.9,
                 cautious_hybrid: bool = False,
                 cautious_came_beta: float = 0.999,
                 cautious_hybrid_weights: tuple[float, float] = (0.5, 0.5),
                 # --- ADOPT-style option ---
                 use_adopt_denominator: bool = False,
                 ):

        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        cautious=cautious,
                        cautious_adaptive_rescale=cautious_adaptive_rescale,
                        cautious_adaptive_alpha=cautious_adaptive_alpha,
                        cautious_momentum_mask=cautious_momentum_mask,
                        cautious_beta_mask=cautious_beta_mask,
                        cautious_hybrid=cautious_hybrid,
                        cautious_came_beta=cautious_came_beta,
                        cautious_hybrid_weights=cautious_hybrid_weights,
                        use_adopt_denominator=use_adopt_denominator,
                        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        """Switches the optimizer to evaluation mode."""
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Interpolate from current parameters to the long-term average 'z'
                        p.data.lerp_(end=state['z'], weight=1 - 1 / beta1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        """Switches the optimizer to training mode."""
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Revert the interpolation to restore training-time parameters
                        p.data.lerp_(end=state['z'], weight=1 - beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group['train_mode']:
                raise RuntimeError("Optimizer must be in train mode to perform a step. Call optimizer.train()")

            # --- Retrieve Hyperparameters ---
            k = group['k']
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            decay = group['weight_decay']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']

            cautious = group['cautious']
            hybrid = group['cautious_hybrid']
            use_adopt_denom = group['use_adopt_denominator']

            # --- Schedule-Free LR and Weight Calculation ---
            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0
            bias_correction2 = 1 - beta2 ** (k + 1)
            # This is the effective learning rate for the 'z' step
            effective_lr = lr * sched * math.sqrt(bias_correction2)

            lr_max = group['lr_max'] = max(effective_lr, group['lr_max'])
            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            ckp1 = weight / weight_sum if weight_sum > 0 else 0
            
            # This is the effective learning rate for the 'y' (parameter) step
            adaptive_y_lr = effective_lr * (1 - beta1 * (1 - ckp1))

            for p in group['params']:
                if p.grad is None:
                    continue

                y = p.data
                grad = p.grad.data

                if cautious:
                    grad_for_mask = grad.clone()

                state = self.state[p]

                # --- State Initialization ---
                if 'z' not in state:
                    state['z'] = torch.clone(y)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if cautious and group['cautious_momentum_mask']:
                        state['exp_avg_mask'] = torch.zeros_like(p.data)
                    if cautious and hybrid:
                        state['exp_avg_res'] = torch.zeros_like(p.data)

                z = state['z']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # --- Core Update Logic ---

                # 1. Update first moment (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 2. Calculate denominator for normalization
                if use_adopt_denom and k > 0:
                    # ADOPT-style: Use previous step's variance for stability
                    denom = exp_avg_sq.sqrt().add_(eps)
                else:
                    # Adam-style: Use current gradient to compute variance for this step
                    # (This branch is also taken on the first step, k=0)
                    temp_exp_avg_sq = exp_avg_sq.clone()
                    temp_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = temp_exp_avg_sq.sqrt().add_(eps)

                # 3. Normalize gradient
                grad_normalized = grad.div(denom)

                # 4. Apply Weight Decay (AdamW style)
                if decay != 0:
                    grad_normalized.add_(y, alpha=decay)

                # 5. Cautious Logic (if enabled)
                if cautious:
                    # Calculate the proposed update vector 'u'
                    u = (y - z).mul(ckp1).add(grad_normalized, alpha=adaptive_y_lr)

                    if hybrid:
                        w_cautious, w_came = group['cautious_hybrid_weights']
                        
                        # Score 1: Cautious (Spatial Consistency)
                        score_cautious = torch.sigmoid(u * grad_for_mask * 5.0)

                        # Score 2: CAME-like (Temporal Consistency)
                        exp_avg_res = state['exp_avg_res']
                        res = (exp_avg - u).pow(2)
                        exp_avg_res.mul_(group['cautious_came_beta']).add_(res, alpha=1 - group['cautious_came_beta'])
                        
                        # Invert and scale to get a 0-1 score
                        score_came = torch.exp(-exp_avg_res.sqrt())
                        
                        # Blend scores to get final suppression factor
                        suppression_factor = (score_cautious * w_cautious + score_came * w_came)
                        
                        # ===============================================================
                        # ★★★★★★★★★★★★★★★★★★★ エラー修正箇所 ★★★★★★★★★★★★★★★★★★★
                        # ===============================================================
                        # Apply suppression and update parameters
                        # y.sub_(u, alpha=suppression_factor)  <- This caused TypeError
                        y.sub_(u.mul(suppression_factor))  # Correct way
                        # ===============================================================

                    else: # Original Cautious logic (momentum mask / adaptive rescale)
                        current_mask = (u * grad_for_mask > 0).to(grad.dtype)
                        if group['cautious_momentum_mask']:
                            exp_avg_mask = state['exp_avg_mask']
                            exp_avg_mask.mul_(group['cautious_beta_mask']).add_(current_mask, alpha=1 - group['cautious_beta_mask'])
                            final_mask = exp_avg_mask
                        else:
                            final_mask = current_mask

                        if group['cautious_adaptive_rescale']:
                            mask_pos = (final_mask > 0.5).to(grad.dtype)
                            u_pos = u * mask_pos
                            u_neg = u * (1.0 - mask_pos)
                            energy = torch.linalg.vector_norm(u_neg) / (u.numel()**0.5 + eps)
                            if torch.isfinite(energy):
                                u_pos.mul_(1.0 + group['cautious_adaptive_alpha'] * energy)
                            y.sub_(u_pos)
                        else:
                            u_masked = u * final_mask
                            u_masked.mul_(final_mask.numel() / (final_mask.sum() + eps))
                            y.sub_(u_masked)

                else:  # Not cautious: Standard Schedule-Free update
                    y.lerp_(end=z, weight=ckp1)
                    y.sub_(grad_normalized, alpha=adaptive_y_lr)

                # 6. Update long-term average 'z'
                z.sub_(grad_normalized, alpha=effective_lr)

                # 7. Update second moment (variance) for the *next* step
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            group['k'] = k + 1
        return loss
