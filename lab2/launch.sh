LOG_DIR="./notebook/logs/SiameseNetwork"
# clear logs folder
rm -r $LOG_DIR
mkdir -p $LOG_DIR

# run main.py with specified arguments
python main.py \
    --select_scale=0.05 \
    --embed_channels=8 \
    --hidden_channels=32 \
    --learning_rate=1e-3 \
    --batch_size=1024 \
    --num_epoch=1 \
    --num_workers=4 \
    --log_path=$LOG_DIR
