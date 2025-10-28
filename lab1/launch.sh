LOG_DIR="./logs/normal"
# clear logs folder
rm -r $LOG_DIR
mkdir -p $LOG_DIR

# run main.py with specified arguments
python main.py \
    --csv_path="./dataset.csv" \
    --batch_size=32 \
    --learning_rate=1e-3 \
    --num_epoch=2 \
    --criterion="crossentropy" \
    --log_dir=$LOG_DIR \
    --structure="normal"
