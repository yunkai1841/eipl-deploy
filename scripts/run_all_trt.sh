# !/bin/bash

python=/usr/bin/python
main_script=tensorrt/infer.py

models=("sarnn" "cnnrnn" "cnnrnnln" "caebn")
precision=("fp32" "fp16" "int8")
results=("result.mp4" "time.csv" "time.txt" "time.png")

for model in "${models[@]}"; do
    result_dir=result/$model
    echo "Running $model"
    $python $main_script --model $model
    mkdir -p $result_dir/fp32
    for result in "${results[@]}"; do
        echo "Moving $result to $result_dir/fp32"
        mv $result $result_dir/fp32
    done
    $python $main_script --model $model --fp16
    mkdir -p $result_dir/fp16
    for result in "${results[@]}"; do
        echo "Moving $result to $result_dir/fp16"
        mv $result $result_dir/fp16
    done
    $python $main_script --model $model --int8
    mkdir -p $result_dir/int8
    for result in "${results[@]}"; do
        echo "Moving $result to $result_dir/int8"
        mv $result $result_dir/int8
    done
done

# make summary csv file
echo "model,precision,time" > result/result-summary.csv
for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        time=$(cat result/$model/$prec/time.txt | grep "avg inference time" | awk -F "=" '{print $2}')
        echo "$model,$prec,$time" >> result/result-summary.csv
    done
done