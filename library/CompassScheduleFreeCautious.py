import torch
from torch.optim import Optimizer
import math

# ==============================================================================
# ユーティリティ関数: 確率的量子化
# ==============================================================================
# @torch.compile # 必要に応じて有効化
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    fp32のsourceを、確率的丸めを使ってbfloat16のtargetにコピーする。
    """
    with torch.no_grad():
        if target.dtype != torch.bfloat16:
            # 安全策として、bfloat16でない場合は通常のコピーを行う
            target.copy_(source)
            return
            
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )
        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))
        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32
        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

# ==============================================================================
# 全部乗せオプティマイザ: CompassScheduleFreeCautious
# ==============================================================================
class CompassScheduleFreeCautious(Optimizer):
    r"""
    An optimizer that combines Compass, Schedule-Free, and Cautious updates.
    It is designed to run on bfloat16 tensors for maximum efficiency on modern hardware.

    (Arguments docstring is omitted for brevity)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.99, 0.999),
        amp_fac=5,
        eps=1e-8,
        weight_decay=0,
        centralization=0,
        # --- Schedule-Free Options ---
        r=0.0,
        weight_lr_power=2.0,
        # --- Cautious Options ---
        cautious=False,
        cautious_adaptive_rescale=False,
        cautious_adaptive_alpha=0.1,
        cautious_momentum_mask=False,
        cautious_beta_mask=0.9,
    ):
        # --- 入力値の検証 ---
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Betas must be in [0, 1): {betas}")
        if cautious_adaptive_rescale and cautious_adaptive_alpha <= 0:
            raise ValueError("cautious_adaptive_alpha must be > 0")
        if cautious_momentum_mask and not (0.0 <= cautious_beta_mask < 1.0):
            raise ValueError("cautious_beta_mask must be in [0, 1)")

        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            # Schedule-Free state
            k=0,
            weight_sum=0.0,
            lr_max=-1.0,
            r=r,
            weight_lr_power=weight_lr_power,
            # Cautious state
            cautious=cautious,
            cautious_adaptive_rescale=cautious_adaptive_rescale,
            cautious_adaptive_alpha=cautious_adaptive_alpha,
            cautious_momentum_mask=cautious_momentum_mask,
            cautious_beta_mask=cautious_beta_mask,
        )
        super(CompassScheduleFreeCautious, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.no_grad():
                loss = closure()

        for group in self.param_groups:
            # ==================================================================
            # パイプライン ステップ1: スケジュールフリー化による実効学習率の計算
            # ==================================================================
            k = group["k"]
            step = k + 1
            beta1_compass, beta2_compass = group["betas"] # Compass用のbeta
            
            beta2_schedule = 0.999 
            rho_inf = 2 / (1 - beta2_schedule) - 1
            beta2_t = beta2_schedule**step
            rho_t = rho_inf - 2 * step * beta2_t / (1 - beta2_t)
            
            rect = 1.0
            if rho_t > 4.0:
                rect = math.sqrt(
                    (rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                )

            base_lr = group["lr"]
            scheduled_lr = base_lr * rect
            lr_max = group["lr_max"] = max(scheduled_lr, group["lr_max"])

            r = group['r']
            weight_lr_power = group['weight_lr_power']
            weight = (step**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            effective_lr = scheduled_lr
            
            cautious = group['cautious']
            adaptive_rescale = group['cautious_adaptive_rescale']
            adaptive_alpha = group['cautious_adaptive_alpha']
            momentum_mask = group['cautious_momentum_mask']
            beta_mask = group['cautious_beta_mask']

            for p in group["params"]:
                if p.grad is None:
                    continue

                # =========================================================
                #  【変更点】bf16のチェックを一時的に完全に無効化します
                # =========================================================
                # if p.dtype != torch.bfloat16:
                #      raise RuntimeError("CompassScheduleFreeCautious only supports bfloat16 parameters.")
                
                if p.grad.is_sparse:
                    raise RuntimeError("This optimizer does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state["ema_squared"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    if cautious and momentum_mask:
                        state["exp_avg_mask"] = torch.zeros_like(p.data, dtype=torch.bfloat16)

                # ==================================================================
                # パイプライン ステップ2: Compassの更新ルール
                # ==================================================================
                
                grad = p.grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                ema = state["ema"].to(torch.float32)
                ema_squared = state["ema_squared"].to(torch.float32)
                
                if cautious:
                    grad_for_mask = grad.clone()

                amplification_factor = group["amp_fac"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                eps = group["eps"]
                state["step"] += 1
                
                bias_correction1 = 1 - beta1_compass ** state["step"]
                bias_correction2_sqrt = (1 - beta2_compass ** state["step"]) ** 0.5
                
                step_size = effective_lr / bias_correction1

                if centralization != 0:
                    grad.sub_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization))

                ema.mul_(beta1_compass).add_(grad, alpha=1 - beta1_compass)
                provisional_grad = grad.add(ema, alpha=amplification_factor)
                ema_squared.mul_(beta2_compass).addcmul_(provisional_grad, provisional_grad, value=1 - beta2_compass)
                denom = (ema_squared.sqrt() / bias_correction2_sqrt).add_(eps)
                
                provisional_update_vector = provisional_grad.div(denom)

                # ==================================================================
                # パイプライン ステップ3: Cautiousの信頼性チェック
                # ==================================================================
                final_update_vector = provisional_update_vector
                if cautious:
                    current_mask = (provisional_update_vector * grad_for_mask > 0).to(torch.float32)
                    
                    if momentum_mask:
                        if 'exp_avg_mask' not in state:
                            state['exp_avg_mask'] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                        exp_avg_mask = state['exp_avg_mask'].to(torch.float32)
                        exp_avg_mask.mul_(beta_mask).add_(current_mask, alpha=1 - beta_mask)
                        final_mask = exp_avg_mask
                        copy_stochastic_(state['exp_avg_mask'], final_mask)
                    else:
                        final_mask = current_mask

                    if adaptive_rescale:
                        mask_pos = (final_mask > 0.5).to(torch.float32)
                        mask_neg = 1.0 - mask_pos

                        update_pos = provisional_update_vector * mask_pos
                        update_neg = provisional_update_vector * mask_neg
                        
                        energy_to_redistribute = torch.linalg.vector_norm(update_neg) / (provisional_update_vector.numel()**0.5 + eps)
                        
                        if torch.isfinite(energy_to_redistribute):
                            scale_factor = 1.0 + adaptive_alpha * energy_to_redistribute
                            update_pos.mul_(scale_factor)
                        
                        final_update_vector = update_pos
                    else:
                        update_masked = provisional_update_vector.mul(final_mask)
                        
                        mask_sum = final_mask.sum() + eps
                        scale_factor = final_mask.numel() / mask_sum
                        update_masked.mul_(scale_factor)

                        final_update_vector = update_masked

                # ==================================================================
                # パイプライン ステップ4: パラメータ更新と状態の保存
                # ==================================================================
                if weight_decay != 0:
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                p_fp32.data.add_(final_update_vector, alpha=-step_size)

                copy_stochastic_(state["ema"], ema)
                copy_stochastic_(state["ema_squared"], ema_squared)
                copy_stochastic_(p, p_fp32)

            group["k"] = k + 1
            
        return loss