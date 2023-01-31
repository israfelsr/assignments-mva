export PYTHONPATH=$(pwd) 

python scripts/main.py \
    --seed 42 \
    --batch_size 16 \
    --model conv \
    --num_hidden_layers 3 \
    --hidden_dim 10 \
    --learning_rate 0.1 \
    --activation relu \
    --epochs 10 \
    --loss mse \
    --optimizer sgd \
    --run_name test1