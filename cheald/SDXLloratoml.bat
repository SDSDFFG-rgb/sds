call "E:\SD\sd-scripts\sd-scripts\venv\Scripts\activate.bat"
echo "Virtual environment activated."

cd "E:\SD\sd-scripts\sd-scripts\cheald\sd-scripts"

set accelerate_config="accelerate/default_config.yaml"
set num_cpu_threads_per_process=1


accelerate launch ^
--config_file "%accelerate_config%" ^
--num_cpu_threads_per_process %num_cpu_threads_per_process% ^
sdxl_train_network.py ^
--config_file ../SDXLlocalconfig.toml

pause