import os
import re
import shutil
import textwrap

TARGET_FILE = "train_network.py"
BACKUP_FILE = "train_network.py.bak"
PATCH_ID_TRAIN = "# PATCH_ID: UNET_GROUP_CONSOLIDATION_V5_TRAIN"
PATCH_ID_LOGS = "# PATCH_ID: UNET_GROUP_CONSOLIDATION_V5_LOGS"

# ▼▼▼ 修正1: generate_step_logs の修正 ▼▼▼
# textwrap.dedentを使って、先頭のインデントを削除しておく
LOGS_REPLACEMENT_CODE_TEMPLATE = textwrap.dedent(r"""
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions: list,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        # {patch_id}
        logs = {{"loss/current": current_loss, "loss/average": avr_loss}}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()

        if lr_descriptions is not None and len(lrs) == len(lr_descriptions):
            for i, lr in enumerate(lrs):
                logs[f"lr/{{lr_descriptions[i]}}"] = lr
        else:
            if not args.network_train_unet_only and len(lrs) > 1:
                logs["lr/textencoder"] = float(lrs[0])
                for i in range(1, len(lrs)):
                    logs[f"lr/unet_group{{i-1}}"] = float(lrs[i])
            else:
                for i, lr in enumerate(lrs):
                    logs[f"lr/unet_group{{i}}"] = float(lr)

        is_prodigy_like = (
            "prodigy" in args.optimizer_type.lower() or
            "dadapt" in args.optimizer_type.lower() or
            "hyperfusion" in args.optimizer_type.lower() or
            "emonavi" in args.optimizer_type.lower()
        )

        if is_prodigy_like and optimizer is not None:
            opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            for i, param_group in enumerate(opt.param_groups):
                if 'd' in param_group and 'lr' in param_group:
                    lr_desc = lr_descriptions[i] if lr_descriptions and i < len(lr_descriptions) else f"group{{i}}"
                    logs[f"lr/d*lr/{{lr_desc}}"] = param_group['d'] * param_group['lr']

        return logs
""")

# ▼▼▼ 修正2: train メソッドの修正 ▼▼▼
TRAIN_REPLACEMENT_CODE_TEMPLATE = textwrap.dedent(r"""
        # ▼▼▼【パッチ適用箇所 V5】U-Netのパラメータグループを統合 ▼▼▼
        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        # createoptimizer
        trainable_params_orig = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)

        orig_lr_descriptions = None
        if isinstance(trainable_params_orig, tuple):
            params_for_optimizer_orig, orig_lr_descriptions = trainable_params_orig
        else:
            params_for_optimizer_orig = trainable_params_orig

        accelerator.print("Consolidating U-Net parameter groups for unified 'd' learning.")
        unet_params = []
        text_encoder_groups = []
        
        for i, group in enumerate(params_for_optimizer_orig):
            is_te_group = orig_lr_descriptions and i < len(orig_lr_descriptions) and 'textencoder' in orig_lr_descriptions[i].lower()
            if self.is_sdxl:
                 is_te_group = (i < len(text_encoders)) and not args.network_train_unet_only

            if not args.network_train_unet_only and is_te_group:
                text_encoder_groups.append(group)
            else:
                if 'params' in group:
                    unet_params.extend(group['params'])

        params_for_optimizer = []
        lr_descriptions = [] # 新しいdescriptionリストを作成

        # Text Encoderのグループを先に追加（元の順序を尊重）
        if text_encoder_groups:
            params_for_optimizer.extend(text_encoder_groups)
            te_descs = [desc for desc in (orig_lr_descriptions or []) if 'textencoder' in desc.lower()]
            lr_descriptions.extend(te_descs)
            accelerator.print(f"Text Encoder group(s): {{len(text_encoder_groups)}} groups maintained.")

        # U-Net全体を一つのグループとして追加
        if unet_params:
            unet_lr = args.unet_lr if args.unet_lr is not None else args.learning_rate
            params_for_optimizer.append({{'params': unet_params, 'lr': unet_lr}})
            lr_descriptions.append("unet")
            accelerator.print(f"Consolidated U-Net group: {{len(unet_params)}} params, lr: {{unet_lr}}")
        
        # パッチが適用されたことを示すID
        # {patch_id_train}

        optimizer_name, optimizer_args_str, optimizer = train_util.get_optimizer(args, params_for_optimizer)
""")


def patch_file():
    if not os.path.exists(TARGET_FILE):
        print(f"エラー: '{TARGET_FILE}' が見つかりません。")
        return

    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"エラー: '{TARGET_FILE}' の読み込みに失敗しました: {e}")
        return

    # パッチ適用済みかチェック
    if PATCH_ID_TRAIN in content and PATCH_ID_LOGS in content:
        print(f"'{TARGET_FILE}' には既にパッチ(v5)が適用されています。")
        return

    # バックアップを作成
    if not os.path.exists(BACKUP_FILE):
        shutil.copyfile(TARGET_FILE, BACKUP_FILE)
        print(f"元のファイルを '{BACKUP_FILE}' としてバックアップしました。")
    else:
        print(f"バックアップファイル '{BACKUP_FILE}' は既に存在します。")

    # 1. generate_step_logsを修正
    LOGS_START_MARKER = "    def generate_step_logs(\n"
    LOGS_END_MARKER = "        return logs\n"
    logs_pattern_str = f"({re.escape(LOGS_START_MARKER)}).*?({re.escape(LOGS_END_MARKER)})"
    logs_pattern = re.compile(logs_pattern_str, re.DOTALL)
    
    match = logs_pattern.search(content)
    if match:
        # 元のブロックのインデントを取得
        original_indent = re.match(r"(\s*)", match.group(0)).group(1)
        # f-stringエラーを避ける処理
        logs_replacement_code_str = LOGS_REPLACEMENT_CODE_TEMPLATE.format(patch_id=PATCH_ID_LOGS).replace('{{', '{').replace('}}', '}')
        # インデントを合わせて置換
        indented_replacement = textwrap.indent(logs_replacement_code_str, original_indent)
        content = logs_pattern.sub(indented_replacement, content, count=1)
        print("`generate_step_logs` 関数にパッチを適用しました。")
    else:
        print("警告: `generate_step_logs` のパッチ適用箇所が見つかりませんでした。")

    # 2. trainメソッドを修正
    TRAIN_START_MARKER = "        # 学習に必要なクラスを準備する\n"
    TRAIN_END_MARKER = "        optimizer_name, optimizer_args_str, optimizer = train_util.get_optimizer(args, params_for_optimizer)\n"
    train_pattern_str = f"({re.escape(TRAIN_START_MARKER)}).*?({re.escape(TRAIN_END_MARKER)})"
    train_pattern = re.compile(train_pattern_str, re.DOTALL)

    match = train_pattern.search(content)
    if match:
        original_indent = re.match(r"(\s*)", match.group(1)).group(1)
        train_replacement_code_str = TRAIN_REPLACEMENT_CODE_TEMPLATE.format(patch_id_train=PATCH_ID_TRAIN).replace('{{', '{').replace('}}', '}')
        indented_replacement = textwrap.indent(train_replacement_code_str, original_indent).lstrip() # 先頭のインデントはマーカーに含まれるので削除
        content = train_pattern.sub(match.group(1) + indented_replacement, content, count=1)
        print("`train` メソッドにパッチを適用しました。")
    else:
        print("警告: `train` メソッドのパッチ適用箇所が見つかりませんでした。")
        return

    try:
        with open(TARGET_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"'{TARGET_FILE}' に正常にパッチを適用しました (v5)。")
        print("IndentationErrorを修正しました。")
    except Exception as e:
        print(f"エラー: ファイルの書き込みに失敗しました: {e}")


if __name__ == "__main__":
    patch_file()