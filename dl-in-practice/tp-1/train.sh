export PYTHONPATH=$(pwd) 

python scripts/main.py \
    --seed 42 \
    --batch_size 64 \
    --model conv \
    --num_hidden_layers 3 \
    --hidden_dim 20 \
    --learning_rate 0.001 \
    --activation relu \
    --epochs 50 \
    --loss ce \
    --optimizer adam \
    --run_name Best_Model_3