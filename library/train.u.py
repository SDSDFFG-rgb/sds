import os
import shutil
import re

# --- 設定項目 ---
TARGET_FILE = "train_util.py"
BACKUP_FILE = "train_util.py.bak"

# --- パッチ内容 ---

# 置き換える対象の関数定義の正規表現パターン
# `def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):` を探す
# 引数リストの括弧内を柔軟にキャプチャする
OLD_FUNCTION_SIGNATURE_PATTERN = re.compile(
    r"def append_lr_to_logs_with_names\(([^)]+)\):"
)

# 新しい関数定義
# 引数に `optimizer=None` を追加
NEW_FUNCTION_SIGNATURE = "def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names, optimizer=None):"

# 追加するロジックのコードブロック
# インデントは後で自動調整する
CODE_TO_APPEND = """
    # === ▼▼▼ [AUTO PATCH] Add logging for custom optimizer metrics ▼▼▼ ============================
    # optimizerが渡され、かつそれがカスタムオプティマイザの場合にのみ指標をログに追加する
    if optimizer is not None:
        # オプティマイザのクラス名を取得して判定
        optimizer_class_name = optimizer.__class__.__name__
        if optimizer_class_name == 'HyperFusionOptimizerScheduleFree':
            # 最初のparam_groupから指標を取得 (通常は一つしかないか、共通の値を持つ)
            param_group = optimizer.param_groups[0]
            
            # Hybrid Cautiousが有効な場合のみログを出力
            if param_group.get('cautious', False) and param_group.get('cautious_hybrid', False):
                suppression_ratio = param_group.get('avg_suppression_ratio', 0.0)
                cautious_contrib = param_group.get('avg_cautious_contribution', 0.0)

                if "cautious/suppression_ratio" not in logs:
                    logs["cautious/suppression_ratio"] = suppression_ratio
                if "cautious/contribution" not in logs:
                    logs["cautious/contribution"] = cautious_contrib
    # === ▲▲▲ [AUTO PATCH] End of patch ▲▲▲ ========================================================
"""

def apply_patch():
    """
    Applies a patch to train_util.py to add logging for custom optimizer metrics.
    """
    if not os.path.exists(TARGET_FILE):
        print(f"エラー: '{TARGET_FILE}' が見つかりません。")
        print("このスクリプトをsd-scriptsの 'library' ディレクトリに置いて実行してください。")
        return

    print(f"'{TARGET_FILE}' をバックアップしています -> '{BACKUP_FILE}'")
    shutil.copyfile(TARGET_FILE, BACKUP_FILE)

    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    patched = False
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if "AUTO PATCH" in line:
            print("警告: 既にパッチが適用されているようです。処理を中断します。")
            return

        match = OLD_FUNCTION_SIGNATURE_PATTERN.search(line)
        if match:
            # 元の引数リストを取得
            original_args = match.group(1).split(',')
            # 引数リストに optimizer=None が既にあるかチェック
            if any("optimizer" in arg for arg in original_args):
                 print("警告: 既に関数の引数が変更されているようです。手動での確認が必要です。")
                 print("処理を中断します。")
                 return
            
            print(f"関数 'append_lr_to_logs_with_names' を発見しました。パッチを適用します。")
            
            # 1. 関数シグネチャを置き換える
            new_lines.append(NEW_FUNCTION_SIGNATURE + "\n")
            
            # 2. 元の関数のボディをそのまま追加する
            i += 1
            indentation_level = 0
            start_of_function_body = -1
            
            # 関数のインデントレベルを特定
            if i < len(lines):
                leading_whitespace = len(lines[i]) - len(lines[i].lstrip(' '))
                if leading_whitespace > 0:
                    indentation_level = leading_whitespace

            # 関数の終わりを探す
            function_body_lines = []
            j = i
            while j < len(lines):
                body_line = lines[j]
                current_indent = len(body_line) - len(body_line.lstrip(' '))
                
                # インデントが元に戻るか、空行が続いたら関数の終わりとみなす
                if body_line.strip() == "" or current_indent < indentation_level:
                    break
                
                function_body_lines.append(body_line)
                j += 1
            
            new_lines.extend(function_body_lines)
            
            # 3. 新しいロジックを追加する（インデントを合わせる）
            indented_code_to_append = ""
            for code_line in CODE_TO_APPEND.strip().split('\n'):
                indented_code_to_append += ' ' * indentation_level + code_line + '\n'
            
            new_lines.append(indented_code_to_append)
            
            patched = True
            i = j  # ポインタを進める
            continue

        new_lines.append(line)
        i += 1

    if patched:
        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("パッチの適用が完了しました。")
        print("\n**重要**: この後、'train_network.py' などの学習スクリプト側で")
        print("`train_util.append_lr_to_logs_with_names` を呼び出している箇所に、")
        print("引数 `optimizer` を追加する必要があります。")
        print("\n例:")
        print("修正前: append_lr_to_logs_with_names(logs, lr_scheduler, args.optimizer_type, ...)")
        print("修正後: append_lr_to_logs_with_names(logs, lr_scheduler, args.optimizer_type, ..., optimizer)")

    else:
        print("エラー: パッチを適用できる箇所が見つかりませんでした。")
        print("'train_util.py'のバージョンが古いか、既になんらかの変更が加えられている可能性があります。")
        # バックアップを削除
        os.remove(BACKUP_FILE)


if __name__ == "__main__":
    apply_patch()