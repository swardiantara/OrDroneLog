#!/bin/bash
datasets=( filtered )
word_embeds=( bert drone-severity ordinal-severity vector-max )
encoders=( transformer )
weight_class=( uniform )
losses=( logloss )
label_schemas=( 0101 1111 0111 )
decodings=( forward backward )
poolings=( avg max )
bidirectionals=( true false )
n_layers=( 3 )
n_heads=( 8 )
# seeds=( 14298463 246773155 30288239 82511865 90995999 )
seeds=( 14298463 )

for dataset in "${datasets[@]}"; do
    echo "dataset: "$dataset""
    for word_embed in "${word_embeds[@]}"; do
        for encoder in "${encoders[@]}"; do
            for weight in "${weight_class[@]}"; do
                for loss in "${losses[@]}"; do
                    for label_schema in "${label_schemas[@]}"; do
                        for decoding in "${decodings[@]}"; do
                            for seed in "${seeds[@]}"; do
                                for pooling in "${poolings[@]}"; do
                                    if [ "$encoder" = "none" ]; then
                                        rm -r cache_dir
                                        python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir fix-pooling --decoding "$decoding" --seed "$seed"
                                    else
                                        for n_layer in "${n_layers[@]}"; do
                                            if [ "$encoder" = "transformer" ]; then
                                                for n_head in "${n_heads[@]}"; do
                                                    rm -r cache_dir
                                                    # for pooling in "${poolings[@]}"; do
                                                    python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --n_heads "$n_head" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir fix-pooling --decoding "$decoding" --seed "$seed"
                                                    # done
                                                done
                                            else
                                                for bidirectional in "${bidirectionals[@]}"; do
                                                    if [ "$bidirectional" = true ]; then
                                                        rm -r cache_dir
                                                        python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --bidirectional --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir fix-pooling --decoding "$decoding" --seed "$seed"
                                                    else
                                                        rm -r cache_dir
                                                        python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir fix-pooling --decoding "$decoding" --seed "$seed"
                                                    fi
                                                done
                                            fi
                                        done
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done