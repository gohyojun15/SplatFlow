
# Via google drive
# gdown 1Ch9YK0eA7-alMIKK8NxKaoyJ7rWZiqif
# gdown 1BUXCmR7jDTiGfbV55NHf7dZp6GsV5LCf


mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/HJGO/splatflow/resolve/main/gs_decoder.pt?download=true -O gs_decoder.pt
wait

wget https://huggingface.co/HJGO/splatflow/resolve/main/mv_rf_ema.pt?download=true -O mv_rf_ema.pt
wait