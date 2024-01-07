file_loc=$(pwd)
cd ~/workspace/Phase-1/FPS_Tests/CPU/
bash run.sh

python3 ~/workspace/Phase-1/FPS_Tests/CPU/export_stats.py 


cd ~/workspace/Phase-1/FPS_Tests/NVIDIA_CUDA/
bash run.sh

python3 ~/workspace/Phase-1/FPS_Tests/NVIDIA_CUDA/export_stats.py 
cd file_loc
