export PYTHONPATH=$(pwd)
python scripts/evaluate.py\
    --data=./bird_dataset\
    --model_checkpoint=experiments/original_code/Net-epoch=08-validation/accuracy/classification=0.19.ckpt\
    --outfile=./experiments/original_code/results.csv\