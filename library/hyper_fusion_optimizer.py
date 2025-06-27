import torch
import torch.optim
import math
from typing import Callable
import torch.fft
from einops import rearrange

# =============================================================================
# Part 1: Main Optimizer Class (Final Version with AdaBelief)
# =============================================================================

class HyperFusionOptimizerScheduleFree(torch.optim.Optimizer):
    r"""
    The ultimate experimental optimizer, fusing all discussed concepts:
    1. Schedule-Free: Base framework for scheduler-free training.
    2. ADOPT: Delayed variance for input stabilization.
    3. DeMo: Sparse updates with error feedback.
    4. Hybrid Cautious: Final reliability check on the proposed update.
    5. GEACS: Gradient Entropy-based Adaptive Confidence Scaling.
    6. FALCON: Frequency-domain Adaptive Learning rate CONtrol.
    7. Partial-FALCON: Apply FALCON only to high-dimensional parameters for speed.
    8. AdaBelief (NEW): Adapting stepsizes by the belief in observed gradients.
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
                 # --- Core Engine Option ---
                 use_adabelief: bool = False,
                 # --- ADOPT option ---
                 use_adopt_denominator: bool = False,
                 # --- DeMo options ---
                 use_demostyle_update: bool = False,
                 compression_decay: float = 0.999,
                 compression_topk: int = 32,
                 compression_chunk: int = 64,
                 clipping_threshold: float = 2.0,
                 # --- Hybrid Cautious options ---
                 cautious: bool = False,
                 cautious_hybrid: bool = False,
                 cautious_came_beta: float = 0.999,
                 cautious_hybrid_weights: tuple[float, float] = (0.5, 0.5),
                 # --- GEACS options ---
                 use_geacs: bool = False,
                 geacs_beta: float = 0.98,
                 geacs_temperature: float = 0.1,
                 geacs_sensitivity: float = 0.5,
                 # --- FALCON options ---
                 use_falcon: bool = False,
                 falcon_freq_bands: tuple[int, int] = (16, 48),
                 falcon_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 falcon_apply_dims_gte: int = 2,
                 ):

        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

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
                        use_adabelief=use_adabelief,
                        use_adopt_denominator=use_adopt_denominator,
                        use_demostyle_update=use_demostyle_update,
                        compression_decay=compression_decay,
                        compression_topk=compression_topk,
                        compression_chunk=compression_chunk,
                        clipping_threshold=clipping_threshold,
                        cautious=cautious,
                        cautious_hybrid=cautious_hybrid,
                        cautious_came_beta=cautious_came_beta,
                        cautious_hybrid_weights=cautious_hybrid_weights,
                        use_geacs=use_geacs,
                        geacs_beta=geacs_beta,
                        geacs_temperature=geacs_temperature,
                        geacs_sensitivity=geacs_sensitivity,
                        use_falcon=use_falcon,
                        falcon_freq_bands=falcon_freq_bands,
                        falcon_scales=falcon_scales,
                        falcon_apply_dims_gte=falcon_apply_dims_gte,
                        )
        super().__init__(params, defaults)

        # DeMoとFALCONの排他利用チェックと、必要なモジュールの初期化
        if use_demostyle_update and use_falcon:
            raise ValueError("use_demostyle_update and use_falcon cannot be used at the same time.")

        if use_demostyle_update or use_falcon:
            # sd-scriptsなどからタプルで渡された場合に対応するための修正
            if isinstance(compression_chunk, tuple):
                processed_chunk = compression_chunk[0]
            else:
                processed_chunk = compression_chunk
            
            self.transform = TransformDCT(self.param_groups, processed_chunk)
        
        if use_demostyle_update:
            self.compress = CompressDCT()


    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1 - 1 / group['betas'][0])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        p.data.lerp_(end=state['z'], weight=1 - group['betas'][0])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            k, use_adopt_denom, use_demostyle, cautious, hybrid, use_geacs, use_falcon, use_adabelief = (
                group['k'], group['use_adopt_denominator'], group['use_demostyle_update'],
                group['cautious'], group['cautious_hybrid'], group['use_geacs'], group['use_falcon'], group['use_adabelief']
            )
            
            if k < group['warmup_steps']: sched = (k + 1) / group['warmup_steps']
            else: sched = 1.0
            bias_correction2 = 1 - group['betas'][1] ** (k + 1)
            effective_lr = group['lr'] * sched * math.sqrt(bias_correction2)
            lr_max = group['lr_max'] = max(effective_lr, group['lr_max'])
            weight = ((k + 1) ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            ckp1 = weight / weight_sum if weight_sum > 0 else 0
            adaptive_y_lr = effective_lr * (1 - group['betas'][0] * (1 - ckp1))

            for p in group['params']:
                if p.grad is None: continue

                y, grad, state = p.data, p.grad.data, self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(y)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if use_demostyle: state['delta'] = torch.zeros_like(p.data)
                    if cautious and hybrid: state['exp_avg_res'] = torch.zeros_like(p.data)
                    if use_geacs: state['exp_avg_entropy'] = torch.tensor(0.0, device=p.device)

                z, exp_avg, exp_avg_sq = state['z'], state['exp_avg'], state['exp_avg_sq']
                
                exp_avg.mul_(group['betas'][0]).add_(grad, alpha=1 - group['betas'][0])
                
                # --- Denominator Calculation (Moved up) ---
                if use_adopt_denom and k > 0:
                    denom = exp_avg_sq.sqrt().add(group['eps'])
                else:
                    # The denominator calculation depends on the second moment (exp_avg_sq)
                    # which is updated differently for Adam vs AdaBelief.
                    # We compute a temporary `v_hat` for the denominator here.
                    
                    if use_adabelief:
                        # AdaBelief-style variance
                        s_t = (grad - exp_avg).pow(2).add_(group['eps'])
                        # Use a temporary v_hat for the denominator, without updating state yet
                        temp_exp_avg_sq = exp_avg_sq.clone().mul_(group['betas'][1]).add_(s_t, alpha=1-group['betas'][1])
                    else:
                        # Adam-style variance
                        temp_exp_avg_sq = exp_avg_sq.clone().mul_(group['betas'][1]).addcmul_(grad, grad, value=1-group['betas'][1])
                    
                    denom = temp_exp_avg_sq.sqrt().add(group['eps'])
                
                grad_normalized = grad.div(denom)

                if group['weight_decay'] != 0:
                    grad_normalized.add_(y, alpha=group['weight_decay'])
                
                
                # --- Update Vector Calculation ---
                apply_falcon_to_this_param = (
                    use_falcon and 
                    p.grad.dim() >= group['falcon_apply_dims_gte']
                )

                if use_demostyle:
                    # (DeMo logic remains the same)
                    delta = state['delta']
                    delta.mul_(group['compression_decay']).add_(grad_normalized)
                    transformed_delta = self.transform.encode(delta)
                    s_idx, s_val, xshape, _ = self.compress.compress(transformed_delta, group['compression_topk'])
                    decomp_trans = self.compress.decompress(p, s_idx, s_val, xshape)
                    immediate_update = self.transform.decode(decomp_trans)
                    update_norm = torch.linalg.vector_norm(immediate_update)
                    grad_norm = torch.linalg.vector_norm(grad_normalized)
                    if update_norm > grad_norm * group['clipping_threshold'] and grad_norm > 0:
                        clip_coef = (grad_norm * group['clipping_threshold']) / (update_norm + group['eps'])
                        immediate_update.mul_(clip_coef)
                    delta.sub_(immediate_update)
                    base_update_vec = immediate_update

                elif apply_falcon_to_this_param:
                    # (FALCON logic remains the same)
                    transformed_grad = self.transform.encode(grad_normalized)
                    if len(transformed_grad.shape) > 2:
                        original_shape = transformed_grad.shape
                        transformed_grad_flat = transformed_grad.flatten(start_dim=-2)
                    else:
                        original_shape = None
                        transformed_grad_flat = transformed_grad
                    
                    freq_dims_total = transformed_grad_flat.shape[-1]
                    low_band_end = min(group['falcon_freq_bands'][0], freq_dims_total)
                    mid_band_end = min(group['falcon_freq_bands'][1], freq_dims_total)
                    
                    base_update_vec = grad_normalized # Default
                    if 0 < low_band_end < mid_band_end < freq_dims_total:
                        split_sizes = [low_band_end, mid_band_end - low_band_end, freq_dims_total - mid_band_end]
                        try:
                            low_freq, mid_freq, high_freq = torch.split(transformed_grad_flat, split_sizes, dim=-1)
                            s_low, s_mid, s_high = group['falcon_scales']
                            low_freq.mul_(s_low); mid_freq.mul_(s_mid); high_freq.mul_(s_high)
                            scaled_transformed_grad_flat = torch.cat([low_freq, mid_freq, high_freq], dim=-1)
                            if original_shape:
                                scaled_transformed_grad = scaled_transformed_grad_flat.view(original_shape)
                            else:
                                scaled_transformed_grad = scaled_transformed_grad_flat
                            base_update_vec = self.transform.decode(scaled_transformed_grad)
                            update_norm = torch.linalg.vector_norm(base_update_vec)
                            grad_norm = torch.linalg.vector_norm(grad_normalized)
                            if update_norm > grad_norm * group['clipping_threshold'] and grad_norm > 0:
                                clip_coef = (grad_norm * group['clipping_threshold']) / (update_norm + group['eps'])
                                base_update_vec.mul_(clip_coef)
                        except RuntimeError:
                            pass
                else:
                    base_update_vec = grad_normalized
                
                # --- Optional Modifiers ---
                if use_geacs:
                    # (GEACS logic remains the same)
                    grad_abs_flat = grad.abs().flatten()
                    prob_dist = torch.softmax(grad_abs_flat / (group['geacs_temperature'] + group['eps']), dim=-1)
                    entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-9))
                    exp_avg_entropy = state['exp_avg_entropy']
                    exp_avg_entropy.mul_(group['geacs_beta']).add_(entropy, alpha=1 - group['geacs_beta'])
                    confidence = torch.exp(-group['geacs_sensitivity'] * exp_avg_entropy)
                    confidence = confidence.clamp(0.1, 1.0)
                    base_update_vec.mul_(confidence)

                # --- Final Update Step ---
                if cautious:
                    # (Cautious logic remains the same)
                    u = (y - z).mul(ckp1).add(base_update_vec, alpha=adaptive_y_lr)
                    if hybrid:
                        exp_avg_res = state['exp_avg_res']
                        w_c, w_h = group['cautious_hybrid_weights']
                        score_cautious = torch.sigmoid(u * grad * 5.0)
                        res = (exp_avg - u).pow(2)
                        exp_avg_res.mul_(group['cautious_came_beta']).add_(res, alpha=1 - group['cautious_came_beta'])
                        score_came = 1.0 / (exp_avg_res.sqrt() + 1.0)
                        suppression_factor = score_cautious * w_c + score_came * w_h
                        y.sub_(u.mul(suppression_factor))
                    else:
                        mask = (u * grad > 0).to(grad.dtype)
                        y.sub_(u.mul(mask))
                else:
                    y.lerp_(end=z, weight=ckp1).sub_(base_update_vec, alpha=adaptive_y_lr)

                z.sub_(base_update_vec, alpha=effective_lr)

                # --- Update Second Moment (v_t) ---
                if use_adabelief:
                    # AdaBelief-style update
                    s_t = (grad - exp_avg).pow(2).add_(group['eps'])
                    exp_avg_sq.mul_(group['betas'][1]).add_(s_t, alpha=1 - group['betas'][1])
                else:
                    # Adam-style update
                    exp_avg_sq.mul_(group['betas'][1]).addcmul_(grad, grad, value=1 - group['betas'][1])

            group['k'] += 1
        return loss

# =============================================================================
# Part 2: DeMo Helper Classes
# =============================================================================

class TransformDCT:
    @torch.no_grad()
    def __init__(self, param_groups, target_chunk, norm="ortho"):
        self.target_chunk = target_chunk
        self.shape_dict = dict()
        self.f_dict = dict()
        self.b_dict = dict()
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if len(p.shape) == 1: s_list = [p.shape[0]]
                else: s_list = p.shape
                for s in s_list:
                    if s in self.shape_dict: continue
                    sc = _get_smaller_split(s, self.target_chunk)
                    self.shape_dict[s] = sc
                    if sc not in self.f_dict:
                        I = torch.eye(sc, device=p.device, dtype=p.dtype)
                        self.f_dict[sc] = _dct(I, norm=norm)
                        self.b_dict[sc] = _idct(I, norm=norm)

    @torch.no_grad()
    def einsum_2d(self, x, b, d=None):
        if d is None: return torch.einsum("...ij, jb -> ...ib", x, b)
        else: return torch.einsum("...ijkl, jb, ld -> ...ikbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None):
        if d is None: return torch.einsum("...ib, jb -> ...ij", x, b)
        else: return torch.einsum("...ikbd, jb, ld -> ...ijkl", x, b, d)

    @torch.no_grad()
    def encode(self, x):
        if len(x.shape) > 1:
            n1 = self.shape_dict.get(x.shape[0], _get_smaller_split(x.shape[0], self.target_chunk))
            n2 = self.shape_dict.get(x.shape[1], _get_smaller_split(x.shape[1], self.target_chunk))
            n1w = self.f_dict[n1].to(x.device, non_blocking=True)
            n2w = self.f_dict[n2].to(x.device, non_blocking=True)
            x = rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            x = self.einsum_2d(x, n1w, n2w)
        else:
            n1 = self.shape_dict.get(x.shape[0], _get_smaller_split(x.shape[0], self.target_chunk))
            n1w = self.f_dict[n1].to(x.device, non_blocking=True)
            x = rearrange(x, "(x w) -> x w", w=n1)
            x = self.einsum_2d(x, n1w)
        return x

    @torch.no_grad()
    def decode(self, x):
        if len(x.shape) > 2:
            n1, n2 = x.shape[2], x.shape[3]
            n1w = self.b_dict[n1].to(x.device, non_blocking=True)
            n2w = self.b_dict[n2].to(x.device, non_blocking=True)
            x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y h x w -> (y h) (x w)")
        else:
            n1 = x.shape[1]
            n1w = self.b_dict[n1].to(x.device, non_blocking=True)
            x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")
        return x

class CompressDCT:
    def _clamp_topk(self, x, topk):
        if topk > x.shape[-1]: topk = x.shape[-1]
        if topk < 1: topk = 1
        return int(topk)

    @torch.no_grad()
    def compress(self, x, topk):
        xshape = x.shape
        if len(x.shape) > 2: x = rearrange(x, "y x h w -> y x (h w)")
        totalk = x.shape[-1]
        topk_clamped = self._clamp_topk(x, topk)
        idx = torch.topk(x.abs(), k=topk_clamped, dim=-1, largest=True, sorted=False).indices
        val = torch.gather(x, dim=-1, index=idx)
        return idx, val, xshape, totalk

    @torch.no_grad()
    def decompress(self, p, idx, val, xshape):
        if len(xshape) > 2:
            x = torch.zeros(xshape[0] * xshape[1], xshape[2] * xshape[3], device=p.device, dtype=p.dtype)
            x = rearrange(x, "(y x) (h w) -> y x (h w)", y=xshape[0], x=xshape[1], h=xshape[2])
        else:
            x = torch.zeros(xshape, device=p.device, dtype=p.dtype)
        x.scatter_(dim=-1, index=idx, src=val)
        if len(xshape) > 2: x = rearrange(x, "y x (h w) -> y x h w", h=xshape[2])
        return x

# =============================================================================
# Part 3: DCT and Math Helper Functions
# =============================================================================

def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))
def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
def _dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = _dct_fft_impl(v)
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r, W_i = torch.cos(k), torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2; V[:, 1:] /= math.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V
def _idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2; X_v[:, 1:] *= math.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r, W_i = torch.cos(k), torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]
    return x.view(*x_shape)
def _get_prime_divisors(n):
    divisors = []; i = 2
    while i * i <= n:
        if n % i: i += 1
        else: n //= i; divisors.append(int(i))
    if n > 1: divisors.append(int(n))
    return divisors
def _get_divisors(n):
    if n <= 0: return []
    if n == 1: return [1]
    prime_factors = _get_prime_divisors(n)
    divs = {1}
    for p in prime_factors: divs.update({d * p for d in divs})
    return sorted(list(divs))
def _get_smaller_split(n, close_to):
    if isinstance(close_to, tuple): ct = close_to[0]
    else: ct = close_to
    all_divisors = _get_divisors(n)
    if not all_divisors: return n
    for i, val in enumerate(all_divisors):
        if val >= ct:
            if i == 0: return val
            if abs(val - ct) < abs(all_divisors[i-1] - ct): return val
            else: return all_divisors[i-1]
    return all_divisors[-1]