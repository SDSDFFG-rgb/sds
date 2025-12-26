@echo off
setlocal

:: --- ユーザー設定項目 ---
set "NAME=pastel"
set "REVISION=2"
set "PROJECT=%NAME%_%REVISION%"
set "RANK=512"

:: --- パス設定 (自分の環境に合わせて確認・修正してください) ---
set "PRETRAINED_MODEL=E:/SD/SDXL/Illustrious-XL-v1.0.safetensors"
set "TRAIN_DATA_DIR=C:\sd\SDXLdataset2\pastel"
set "OUTPUT_DIR=D:/SD/lora/sdxl/b_lora/%PROJECT%"
set "LYCORIS_PRESET=./lycoris_presets/bdora_content_style_clip.toml"

:: --- 仮想環境の有効化 ---
call ./venv/Scripts/activate

:: --- 学習コマンドの実行 ---
accelerate launch --num_cpu_threads_per_process 8 sdxl_train_network.py ^
	--pretrained_model_name_or_path="%PRETRAINED_MODEL%" ^
	--train_data_dir="%TRAIN_DATA_DIR%" ^
	--output_dir="%OUTPUT_DIR%" ^
	--output_name="%PROJECT%" ^
	--network_module="lycoris.kohya" ^
	--network_args "preset=%LYCORIS_PRESET%"^
	--network_dim=%RANK% ^
	--network_alpha=%RANK% ^
	--caption_extension=".txt" ^
	--enable_bucket ^
	--max_bucket_reso=1280 ^
	--resolution=768 ^
	--min_bucket_reso=256 ^
	--bucket_reso_steps=64 ^
	--flip_aug ^
	--color_aug ^
	--save_model_as="safetensors" ^
	--max_train_steps=3000 ^
	--save_every_n_steps=1000 ^
	--mixed_precision="fp16" ^
	--xformers ^
	--gradient_checkpointing ^
	--network_train_unet_only ^
	--persistent_data_loader_workers ^
	--no_half_vae ^
	--optimizer_type="devHyperFusionScheduleFree" ^
	--learning_rate=1.0 ^
	--optimizer_args "use_prodigy=True" "d_coef=0.2" "cautious=True" "factored=True" ^
	--max_grad_norm=1.0 ^
	--seed=1 ^
	--fp8

echo.
echo 学習が完了しました。
pause