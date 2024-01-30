# !/bin/bash

python=/usr/bin/python
main_script=tensorrt/infer.py

models=("sarnn" "cnnrnn" "cnnrnnln" "caebn")
precision=("fp32" "fp16" "int8")
results=("power.csv" "power.png" "power.txt")

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
echo "model,precision,avg-power,total-energy,energy-per-loop" > result/power-summary.csv
for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        avg=$(cat result/$model/$prec/power.txt | grep "^avg power" | awk -F "=" '{print $2}')
        energy=$(cat result/$model/$prec/power.txt | grep "^total energy" | awk -F "=" '{print $2}')
        epl=$(cat result/$model/$prec/power.txt | grep "^energy per loop" | awk -F "=" '{print $2}')
        echo "$model,$prec,$avg,$energy,$epl" >> result/power-summary.csv
    done
done