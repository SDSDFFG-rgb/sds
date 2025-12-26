@echo off
setlocal

:: --- ユーザー設定項目 ---
set "NAME=my_lora"
set "REVISION=1"
set "PROJECT=%NAME%_%REVISION%"
set "RANK=1024"

:: --- パス設定 (自分の環境に合わせて確認・修正してください) ---
set "PRETRAINED_MODEL=E:/SD/SDXL/Illustrious-XL-v1.0.safetensors"
set "TRAIN_DATA_DIR=c:/sd/bloradataset/tinachd"
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
	--network_args "preset=%LYCORIS_PRESET%" ^
	--network_dim=%RANK% ^
	--network_alpha=%RANK% ^
	--resolution="1024,1024" ^
	--enable_bucket ^
	--min_bucket_reso=1024 ^
	--bucket_reso_steps=32 ^
	--random_crop ^
	--save_model_as="safetensors" ^
	--max_train_steps=5000 ^
	--save_every_n_steps=1000 ^
	--mixed_precision="fp16" ^
	--xformers ^
	--gradient_checkpointing ^
	--persistent_data_loader_workers ^
	--no_half_vae ^
	--caption_extension=".txt" ^
	--optimizer_type="prodigy" ^
	--learning_rate=1.0 ^
	--lr_scheduler="cosine" ^
	--lr_warmup_steps=0 ^
	--optimizer_args weight_decay=1e-04 betas=(0.9,0.9999) eps=1e-08 decouple=True use_bias_correction=True safeguard_warmup=True beta3=None ^
	--max_grad_norm=1.0 ^
	--seed=0

echo.
echo 学習が完了しました。
pause