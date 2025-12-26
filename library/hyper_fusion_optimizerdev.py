import torch
import torch.optim
import math
from typing import Callable, Optional, Tuple
import torch.fft
from einops import rearrange

class devHyperFusionOptimizerScheduleFree(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr: float = 1.0,
                 betas: tuple[float, float] = (0.9, 0.99),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 use_radam_rectify: bool = False,
                 use_adabelief: bool = False,
                 use_decoupled_wd: bool = True,
                 factored: bool = False,
                 use_stableadamw: bool = False,
                 use_prodigy: bool = False,
                 split_groups: bool = False,
                 d0: float = 1e-6,
                 d_coef: float = 1.0,
                 beta3: Optional[float] = None,
                 d_limiter: bool = True,
                 use_weak_ratchet: bool = False,
                 use_dynamic_d: bool = False,
                 beta_d: float = 0.9,
                 use_prodigy_bias_correction_for_d: bool = False,
                 use_prodigy_d_scale_moments: bool = False,
                 use_adopt_denominator: bool = False,
                 cautious: bool = False,
                 cautious_hybrid: bool = False,
                 cautious_came_beta: float = 0.999,
                 cautious_hybrid_weights: tuple[float, float] = (0.5, 0.5),
                 # ▼▼▼ GDCA: パラメータと推奨デフォルト値を追加 ▼▼▼
                 use_gdca: bool = False,
                 gdca_meta_lr: float = 1e-5,
                 gdca_meta_beta: float = 0.9,
                 d_coef_min: float = 1e-4,
                 d_coef_max: float = 100.0,
                 ):

        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, betas=betas, eps=eps, r=r, k=0,
                        warmup_steps=warmup_steps, train_mode=True, weight_sum=0.0, lr_max=-1.0,
                        weight_lr_power=weight_lr_power, weight_decay=weight_decay,
                        use_adabelief=use_adabelief, use_radam_rectify=use_radam_rectify,
                        use_decoupled_wd=use_decoupled_wd, use_prodigy=use_prodigy,
                        split_groups=split_groups, d=d0, d0=d0, d_max=d0, d_numerator=0.0, d_coef=d_coef, beta3=beta3,
                        factored=factored, use_adopt_denominator=use_adopt_denominator,
                        cautious=cautious, cautious_hybrid=cautious_hybrid,
                        cautious_came_beta=cautious_came_beta, cautious_hybrid_weights=cautious_hybrid_weights,
                        use_stableadamw=use_stableadamw, d_limiter=d_limiter,
                        use_weak_ratchet=use_weak_ratchet, use_dynamic_d=use_dynamic_d, beta_d=beta_d,
                        use_prodigy_bias_correction_for_d=use_prodigy_bias_correction_for_d,
                        use_prodigy_d_scale_moments=use_prodigy_d_scale_moments,
                        # ▼▼▼ GDCA: defaultsにパラメータを追加 ▼▼▼
                        use_gdca=use_gdca,
                        gdca_meta_lr=gdca_meta_lr,
                        gdca_meta_beta=gdca_meta_beta,
                        d_coef_min=d_coef_min,
                        d_coef_max=d_coef_max,
                        )
        super().__init__(params, defaults)

    def _rms(self, x):
        return x.pow(2).mean().sqrt()

    def _get_stableadamw_update(self, update, param, eps):
        if (rms_p := self._rms(param.data)) > 0:
            rms_u = self._rms(update)
            return update.clone().div(rms_u.div(rms_p).clamp_min(eps))
        return update

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group['train_mode']:
                beta0 = group['betas'][0]
                if beta0 > 0:
                    weight = 1 - 1 / beta0
                    for p in group['params']:
                        state = self.state[p]
                        if 'z' in state:
                            p.data.lerp_(end=state['z'], weight=weight)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state: p.data.lerp_(end=state['z'], weight=1 - group['betas'][0])
                group['train_mode'] = True

    def _update_d(self, group, d_numerator_new, d_denom):
        d_coef, d, d0, eps = group['d_coef'], group['d'], group['d0'], group['eps']
        beta3 = group['beta3']
        if beta3 is None: beta3 = math.sqrt(group['betas'][1])
        
        d_numerator = group['d_numerator'] * beta3 + d_numerator_new
        
        safe_d_denom = max(d_denom, eps)
        d_hat = d_coef * d_numerator / safe_d_denom
        
        if group['use_dynamic_d']:
            beta_d = group['beta_d']
            new_d = d * beta_d + d_hat * (1 - beta_d)
            new_d = max(new_d, d0) 
            group['d'] = new_d
            group['d_numerator'] = d_numerator
        else:
            if group['use_weak_ratchet']:
                if group['d_limiter']:
                    growth_rate = 2.0
                    d_hat = min(d * growth_rate, d_hat)
                d = max(d, d_hat)
                group.update(d=d, d_numerator=d_numerator)
            else:
                d_max = group['d_max']
                d_limiter = group['d_limiter']
                growth_rate = 2.0 if d_limiter else float('inf')
                d_max = max(d_max, d_hat)
                d = min(d_max, d * growth_rate)
                group.update(d_numerator=d_numerator, d=d, d_max=d_max)

    def _init_state(self, p, group, state):
        if 'z' not in state: state['z'] = torch.clone(p.data)
        if 'exp_avg' not in state: state['exp_avg'] = torch.zeros_like(p.data)
        
        if group['factored']:
            if 'exp_avg_sq_row' not in state:
                state['exp_avg_sq_row'] = torch.zeros(p.shape[:-1], dtype=p.dtype, device=p.device)
            if 'exp_avg_sq_col' not in state:
                state['exp_avg_sq_col'] = torch.zeros(p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device)
            if 'exp_avg_sq' in state: del state['exp_avg_sq']
        else:
            if 'exp_avg_sq' not in state:
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            if 'exp_avg_sq_row' in state: del state['exp_avg_sq_row']
            if 'exp_avg_sq_col' in state: del state['exp_avg_sq_col']

        if group['use_prodigy']:
            if 'p0' not in state: state['p0'] = p.data.flatten().detach().clone()
            if 's' not in state: state['s'] = torch.zeros_like(p.data.flatten()).detach()
        if group['cautious'] and group['cautious_hybrid'] and 'exp_avg_res' not in state:
            state['exp_avg_res'] = torch.zeros_like(p.data)
        
        if group['use_gdca'] and self.param_groups.index(group) == 0:
            if 'gdca_state' not in group:
                group['gdca_state'] = { 'meta_grad_momentum': 0.0 }
    
    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        meta_grad_signal_total = 0.0

        if self.param_groups[0]['use_prodigy'] and not self.param_groups[0]['split_groups']:
            d_numerator_new_total, d_denom_total = 0.0, 0.0
            first_group = self.param_groups[0]
            k, beta1, beta2 = first_group['k'], first_group['betas'][0], first_group['betas'][1]
            beta3 = first_group['beta3']
            if beta3 is None: beta3 = math.sqrt(beta2)
            prodigy_bias_correction_for_d = 1.0
            if first_group['use_prodigy_bias_correction_for_d']:
                bias_correction_beta1_term = (1 - beta1**(k+1))
                bias_correction_beta2_term = math.sqrt(1 - beta2**(k+1))
                if bias_correction_beta1_term > 0:
                    prodigy_bias_correction_for_d = bias_correction_beta2_term / bias_correction_beta1_term
            for group in self.param_groups:
                d = group['d']
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    state = self.state[p]
                    self._init_state(p, group, state)
                    s, p0 = state['s'], state['p0']
                    dlr_for_prodigy = d * group['lr'] * prodigy_bias_correction_for_d
                    d_numerator_new_total += (d / group['d0']) * dlr_for_prodigy * torch.dot(grad.flatten(), p0 - p.data.flatten()).item()
                    s.mul_(beta3).add_(grad.flatten(), alpha=(d / group['d0']) * dlr_for_prodigy)
                    d_denom_total += s.abs().sum().item()
            self._update_d(first_group, d_numerator_new_total, d_denom_total)
            global_d_state = {key: first_group[key] for key in ['d', 'd_max', 'd_numerator', 'd_coef'] if key in first_group}
            for group in self.param_groups[1:]:
                group.update(global_d_state)

        for group in self.param_groups:
            k, beta1, beta2, eps = group['k'], group['betas'][0], group['betas'][1], group['eps']
            
            params_with_grad, grads_original, exp_avgs, zs = [], [], [], []
            for p in group['params']:
                if p.grad is None: continue
                params_with_grad.append(p); grads_original.append(p.grad.data)
                state = self.state[p]; self._init_state(p, group, state)
                exp_avgs.append(state['exp_avg']); zs.append(state['z'])
                
            if not params_with_grad:
                group['k'] += 1
                continue
            
            if group['use_prodigy'] and group['split_groups']:
                d, d0, beta3 = group['d'], group['d0'], group['beta3']
                d_numerator_new_total, d_denom_total = 0.0, 0.0
                if beta3 is None: beta3 = math.sqrt(beta2)
                prodigy_bias_correction_for_d = 1.0
                if group['use_prodigy_bias_correction_for_d']:
                    bias_correction_beta1_term = (1 - beta1**(k+1))
                    bias_correction_beta2_term = math.sqrt(1 - beta2**(k+1))
                    if bias_correction_beta1_term > 0:
                        prodigy_bias_correction_for_d = bias_correction_beta2_term / bias_correction_beta1_term
                for i in range(len(params_with_grad)):
                    p, grad = params_with_grad[i], grads_original[i]
                    state = self.state[p]; s, p0 = state['s'], state['p0']
                    dlr_for_prodigy = d * group['lr'] * prodigy_bias_correction_for_d
                    d_numerator_new_total += (d / d0) * dlr_for_prodigy * torch.dot(grad.flatten(), p0 - p.data.flatten()).item()
                    s.mul_(beta3).add_(grad.flatten(), alpha=(d / d0) * dlr_for_prodigy)
                    d_denom_total += s.abs().sum().item()
                self._update_d(group, d_numerator_new_total, d_denom_total)
            
            r, lr, warmup_steps = group['r'], group['lr'], group['warmup_steps']
            if k < warmup_steps: sched = (k + 1) / warmup_steps
            else: sched = 1.0
            
            # ▼▼▼【修正】バイアス補正項をここで計算 ▼▼▼
            bias_correction2 = 1.0
            if beta2 > 0:
                bias_correction2 = 1 - beta2 ** (k + 1)

            base_lr = group['d'] if group['use_prodigy'] else lr
            # ▼▼▼【修正】effective_lrにsqrt(bias_correction2)を戻し、数値的安定性を確保 ▼▼▼
            effective_lr = base_lr * sched * math.sqrt(bias_correction2)
            
            lr_max = group['lr_max'] = max(effective_lr, group['lr_max'])
            weight = ((k + 1) ** r) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            ckp1 = weight / weight_sum if weight_sum > 0 else 0
            adaptive_y_lr = effective_lr * (1 - beta1 * (1 - ckp1))

            torch._foreach_mul_(exp_avgs, beta1)
            first_moment_alpha = 1 - beta1
            if group['use_prodigy'] and group['use_prodigy_d_scale_moments']:
                first_moment_alpha *= group['d']
            torch._foreach_add_(exp_avgs, grads_original, alpha=first_moment_alpha)
            
            final_update_vectors = []
            for i in range(len(params_with_grad)):
                p, grad_orig, exp_avg, y, z = params_with_grad[i], grads_original[i], exp_avgs[i], params_with_grad[i].data, zs[i]
                state = self.state[p]
                
                p_old = None
                if group['use_gdca']:
                    p_old = y.clone()

                # --- 2nd moment and normalization ---
                if group['factored']:
                    r, c = state['exp_avg_sq_row'], state['exp_avg_sq_col']
                    grad_sq = grad_orig.pow(2)
                    if group['use_adabelief']:
                        grad_sq = (grad_orig - exp_avg).pow(2)
                    
                    r.mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=1 - beta2)
                    c.mul_(beta2).add_(grad_sq.mean(dim=-2), alpha=1 - beta2)
                    
                    # ▼▼▼【修正】分母側でのバイアス補正を削除（学習率側で適用済みのため）▼▼▼
                    denom_rc = r.unsqueeze(-1) * c.unsqueeze(-2)
                    denom_approx = denom_rc.div(r.mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(eps)).sqrt_().add_(eps)
                    grad_normalized = grad_orig.div(denom_approx)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    
                    second_moment_alpha = 1 - beta2
                    if group['use_prodigy'] and group['use_prodigy_d_scale_moments']:
                        second_moment_alpha *= (group['d'] ** 2)
                    
                    if group['use_adabelief']:
                        s_t = (grad_orig - exp_avg).pow(2).add_(eps)
                        exp_avg_sq.mul_(beta2).add_(s_t, alpha=second_moment_alpha)
                    else:
                        exp_avg_sq.mul_(beta2).addcmul_(grad_orig, grad_orig, value=second_moment_alpha)

                    # ▼▼▼【修正】denom計算からバイアス補正を削除（学習率側で適用済みのため）▼▼▼
                    denom = exp_avg_sq.sqrt().add_(eps)
                    grad_normalized = grad_orig.div(denom)

                # --- Update vector calculation ---
                if group['use_stableadamw']:
                    base_update_vec = self._get_stableadamw_update(grad_normalized, p, eps)
                else:
                    base_update_vec = grad_normalized
                
                # --- Parameter update ---
                if group['cautious']:
                    u = (y - z).mul(ckp1).add(base_update_vec, alpha=adaptive_y_lr)
                    if group['cautious_hybrid']:
                        exp_avg_res = state['exp_avg_res']; res = y - exp_avg_res; wh, wu = group['cautious_hybrid_weights']
                        h = res.mul(wh).add(u, alpha=wu); suppression_factor = torch.sigmoid(-h * u).mul(2)
                        y.sub_(u.mul(suppression_factor)); exp_avg_res.mul_(group['cautious_came_beta']).add_(y, alpha=1-group['cautious_came_beta'])
                    else:
                        mask = (u * grad_orig > 0).to(grad_orig.dtype)
                        y.sub_(u.mul(mask))
                else:
                    y.lerp_(end=z, weight=ckp1).sub_(base_update_vec, alpha=adaptive_y_lr)
                
                final_update_vectors.append(base_update_vec)
                
                if group['use_gdca'] and p_old is not None:
                    actual_update_vector = p_old - y
                    if actual_update_vector.numel() > 0:
                        signal = torch.dot(grad_orig.flatten(), actual_update_vector.flatten()).item()
                        if group['split_groups']:
                             meta_grad_signal_total += signal
                        else:
                             meta_grad_signal_total += signal

            if group['use_decoupled_wd'] and group['weight_decay'] != 0:
                torch._foreach_add_([p.data for p in params_with_grad], [p.data for p in params_with_grad], alpha=-base_lr * group['weight_decay'])
            
            torch._foreach_sub_(zs, final_update_vectors, alpha=effective_lr)

            group['k'] += 1
        
        group = self.param_groups[0]
        if group['use_gdca'] and group['use_prodigy']:
            gdca_state = group['gdca_state']
            beta_meta = group['gdca_meta_beta']
            
            current_momentum = gdca_state['meta_grad_momentum']
            new_momentum = beta_meta * current_momentum + (1 - beta_meta) * meta_grad_signal_total
            gdca_state['meta_grad_momentum'] = new_momentum
            
            meta_lr = group['gdca_meta_lr']
            update_factor = torch.exp(torch.tensor(meta_lr * new_momentum)).item()
            
            new_d_coef = group['d_coef'] * update_factor
            
            group['d_coef'] = max(group['d_coef_min'], min(new_d_coef, group['d_coef_max']))
            
            if not group['split_groups']:
                for other_group in self.param_groups[1:]:
                    other_group['d_coef'] = group['d_coef']
                    
        return loss