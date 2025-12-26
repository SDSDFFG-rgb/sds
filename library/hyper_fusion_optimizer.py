import torch
import torch.optim
import math
from typing import Callable
import torch.fft
from einops import rearrange

# =============================================================================
# Part 1: Main Optimizer Class (with Hybrid foreach/for-loop Execution)
# =============================================================================

class HyperFusionOptimizerScheduleFree(torch.optim.Optimizer):
    r"""
    The ultimate experimental optimizer, with RAdam-gated FALCON activation.
    This version implements a hybrid execution model for maximum performance
    and flexibility, automatically using foreach for simple operations and
    for-loops for complex, parameter-specific logic.
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
                 # --- Core Engine & Stability Options ---
                 use_adabelief: bool = False,
                 use_radam_rectify: bool = False,
                 use_decoupled_wd: bool = False,
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
                        use_radam_rectify=use_radam_rectify,
                        use_decoupled_wd=use_decoupled_wd,
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

        if use_demostyle_update and use_falcon:
            raise ValueError("use_demostyle_update and use_falcon cannot be used at the same time.")

        if use_demostyle_update or use_falcon:
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
            # =================================================================
            # 準備フェーズ: 全パラメータ共通の値を計算
            # =================================================================
            k = group['k']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            # --- Phase 1: RAdam Warmup Check ---
            is_radam_warmup = False
            radam_rectify_term = 1.0
            if group['use_radam_rectify']:
                beta2_t = beta2 ** (k + 1)
                N_sma_max = 2 / (1 - beta2) - 1
                if N_sma_max > 0:
                    N_sma = N_sma_max - 2 * (k + 1) * beta2_t / (1 - beta2_t) if (1 - beta2_t) > 0 else N_sma_max
                    if N_sma < 5:
                        is_radam_warmup = True
                    else:
                        r_t_num = (N_sma - 4) * (N_sma - 2) * N_sma_max
                        r_t_den = (N_sma_max - 4) * (N_sma_max - 2) * N_sma
                        if r_t_den > 0:
                            radam_rectify_term = math.sqrt(r_t_num / r_t_den)
            
            if is_radam_warmup:
                # RAdamウォームアップ中は、元のシンプルなループでモーメント更新のみ行い、終了
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    if 'exp_avg' not in state:
                        state['z'] = torch.clone(p.data)
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    grad, exp_avg, exp_avg_sq = p.grad.data, state['exp_avg'], state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    if group['use_adabelief']:
                        s_t = (grad - exp_avg).pow(2).add_(eps)
                        exp_avg_sq.mul_(beta2).add_(s_t, alpha=1 - beta2)
                    else:
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                group['k'] += 1
                continue
            
            # --- Normal Phase: 共通の係数を計算 ---
            if k < group['warmup_steps']: sched = (k + 1) / group['warmup_steps']
            else: sched = 1.0
            
            bias_correction2 = 1 - beta2 ** (k + 1)
            effective_lr = group['lr'] * sched * math.sqrt(bias_correction2)
            if group['use_radam_rectify']:
                effective_lr *= radam_rectify_term

            lr_max = group['lr_max'] = max(effective_lr, group['lr_max'])
            weight = ((k + 1) ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            ckp1 = weight / weight_sum if weight_sum > 0 else 0
            adaptive_y_lr = effective_lr * (1 - beta1 * (1 - ckp1))

            # =================================================================
            # データ収集フェーズ: foreach化するテンソルをリストにまとめる
            # =================================================================
            params_with_grad = []
            grads_original = []
            exp_avgs = []
            exp_avg_sqs = []
            zs = []
            
            is_complex_option_enabled = (group['use_demostyle_update'] or
                                         group['use_falcon'] or
                                         group['use_geacs'] or
                                         group['cautious'])

            for p in group['params']:
                if p.grad is None: continue
                
                params_with_grad.append(p)
                grads_original.append(p.grad.data)
                
                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(p.data)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['use_demostyle_update']: state['delta'] = torch.zeros_like(p.data)
                    if group['cautious'] and group['cautious_hybrid']: state['exp_avg_res'] = torch.zeros_like(p.data)
                    if group['use_geacs']: state['exp_avg_entropy'] = torch.tensor(0.0, device=p.device)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                zs.append(state['z'])

            if not params_with_grad:
                group['k'] += 1
                continue

            # =================================================================
            # 計算フェーズ: foreachとforループのハイブリッド実行
            # =================================================================
            
            # --- フェーズ1: foreachで可能な計算 (前半) ---
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads_original, alpha=1 - beta1)

            # ▼▼▼ 修正箇所 ▼▼▼
            # use_adopt_denominator=Falseの場合に、この場でexp_avg_sqを更新し、
            # それを使ってdenomを計算するようにロジックを統合。
            if group['use_adopt_denominator'] and k > 0:
                # Adpot有効時: 前回のステップのexp_avg_sqsを分母に使い、更新は関数の最後で行う
                denom = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_add_(denom, eps)
            else:
                # Adpot無効時: 今回の勾配でexp_avg_sqsを更新し、それを分母に使う
                if group['use_adabelief']:
                    # AdaBeliefの更新ロジック
                    for i in range(len(grads_original)):
                        s_t = (grads_original[i] - exp_avgs[i]).pow(2).add_(eps)
                        exp_avg_sqs[i].mul_(beta2).add_(s_t, alpha=1 - beta2)
                else:
                    # Adam形式の更新ロジック
                    torch._foreach_mul_(exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(exp_avg_sqs, grads_original, grads_original, value=1 - beta2)
                
                # 更新したばかりのexp_avg_sqsを使ってdenomを計算
                denom = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_add_(denom, eps)
            # ▲▲▲ 修正完了 ▲▲▲

            grads_for_update = torch._foreach_div(grads_original, denom)

            if not group['use_decoupled_wd'] and group['weight_decay'] != 0:
                torch._foreach_add_(grads_for_update, [p.data for p in params_with_grad], alpha=group['weight_decay'])

            # --- フェーズ2: forループが必須な複雑なオプションの処理か、完全foreachパスかの分岐 ---
            if is_complex_option_enabled:
                final_update_vectors = []
                for i in range(len(params_with_grad)):
                    p, grad_orig, grad_normalized, state = params_with_grad[i], grads_original[i], grads_for_update[i], self.state[params_with_grad[i]]
                    y, z = p.data, zs[i]
                    
                    base_update_vec = grad_normalized
                    apply_falcon_to_this_param = group['use_falcon'] and p.grad.dim() >= group['falcon_apply_dims_gte']

                    if group['use_demostyle_update']:
                        delta = state['delta']
                        encoded_g = self.transform.encode(grad_normalized)
                        idx, val, xshape, totalk = self.compress.compress(encoded_g, group['compression_topk'])
                        compressed_g = self.compress.decompress(grad_normalized, idx, val, xshape)
                        immediate_update = self.transform.decode(compressed_g)
                        delta.mul_(group['compression_decay']).add_(immediate_update, alpha=1-group['compression_decay'])
                        torch.clamp(delta, -group['clipping_threshold'], group['clipping_threshold'], out=delta)
                        base_update_vec = delta
                    elif apply_falcon_to_this_param:
                        encoded_g = self.transform.encode(grad_normalized)
                        low_band, high_band = group['falcon_freq_bands']
                        low_scale, mid_scale, high_scale = group['falcon_scales']
                        mask_low = torch.ones_like(encoded_g)
                        mask_mid = torch.ones_like(encoded_g)
                        mask_low[..., low_band:] = 0
                        mask_mid[..., :low_band] = 0
                        mask_mid[..., high_band:] = 0
                        mask_high = 1 - (mask_low + mask_mid)
                        base_update_vec = self.transform.decode(encoded_g * (mask_low * low_scale + mask_mid * mid_scale + mask_high * high_scale))
                    
                    if group['use_geacs']:
                        exp_avg_entropy = state['exp_avg_entropy']
                        entropy = torch.sum(-torch.log(denom[i]) / math.log(denom[i].numel())).item()
                        exp_avg_entropy.mul_(group['geacs_beta']).add_(entropy, alpha=1-group['geacs_beta'])
                        confidence = torch.sigmoid((exp_avg_entropy - group['geacs_sensitivity']) / group['geacs_temperature'])
                        base_update_vec.mul_(confidence)
                    
                    if group['cautious']:
                        u = (y - z).mul(ckp1).add(base_update_vec, alpha=adaptive_y_lr)
                        if group['cautious_hybrid']:
                            exp_avg_res = state['exp_avg_res']
                            res = y - exp_avg_res
                            wh, wu = group['cautious_hybrid_weights']
                            h = res.mul(wh).add(u, alpha=wu)
                            suppression_factor = torch.sigmoid(-h * u).mul(2)
                            y.sub_(u.mul(suppression_factor))
                            exp_avg_res.mul_(group['cautious_came_beta']).add(y, alpha=1-group['cautious_came_beta'])
                        else:
                            mask = (u * grad_orig > 0).to(grad_orig.dtype)
                            y.sub_(u.mul(mask))
                    else:
                        y.lerp_(end=z, weight=ckp1).sub_(base_update_vec, alpha=adaptive_y_lr)

                    final_update_vectors.append(base_update_vec)
            else:
                # --- 完全foreachパス ---
                y_list = [p.data for p in params_with_grad]
                torch._foreach_mul_(y_list, 1.0 - ckp1)
                torch._foreach_add_(y_list, zs, alpha=ckp1)
                torch._foreach_sub_(y_list, grads_for_update, alpha=adaptive_y_lr)
                final_update_vectors = grads_for_update

            # --- フェーズ3: foreachで可能な計算 (後半) ---
            if group['use_decoupled_wd'] and group['weight_decay'] != 0:
                torch._foreach_add_([p.data for p in params_with_grad], [p.data for p in params_with_grad], alpha=-group['lr'] * group['weight_decay'])
            
            torch._foreach_sub_(zs, final_update_vectors, alpha=effective_lr)

            # ▼▼▼ 修正箇所 ▼▼▼
            # use_adopt_denominator=Trueの時のみ、ここでexp_avg_sqを更新する。
            # Falseの場合は既にdenom計算時に更新済みのため、ここでは何もしない。
            if group['use_adopt_denominator']:
                if group['use_adabelief']:
                    for i in range(len(grads_original)):
                        s_t = (grads_original[i] - exp_avgs[i]).pow(2).add_(eps)
                        exp_avg_sqs[i].mul_(beta2).add_(s_t, alpha=1 - beta2)
                else:
                    torch._foreach_mul_(exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(exp_avg_sqs, grads_original, grads_original, value=1 - beta2)
            # ▲▲▲ 修正完了 ▲▲▲

            group['k'] += 1
        return loss

# =============================================================================
# Part 2 & 3: Helper Classes and Functions (No changes)
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