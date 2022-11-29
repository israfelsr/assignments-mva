export PYTHONPATH=$(pwd)
python scripts/evaluate.py\
    --data=./bird_dataset\
    --model_checkpoint=./experiments/original_code_adamw/last-v1.ckpt\
    --outfile=./experiments/original_code_adamw/results.csv\