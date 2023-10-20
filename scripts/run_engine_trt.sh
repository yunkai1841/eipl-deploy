# !/bin/bash
# Run tensorrt engine with trtexec

ENGINE_EXT=".trt"
DIR=temp/
RESULT_DIR=result/
TRTEXEC_OPT=""

if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist"
    exit 1
fi

# find file in directory also in subdirectories
for file in "$DIR"**/*; do
    if [ -f "$file" ]; then
        if [[ "$file" == *"$ENGINE_EXT" ]]; then
            echo "Running $file"
            outfile=${file/$DIR/$RESULT_DIR}
            outfile=${outfile%.*}.txt
            echo "Output file: $outfile"
            trtexec --loadEngine="$file" $TRTEXEC_OPT > $outfile
        fi
    fi
done
