#!/bin/bash
datasets=( filtered )
word_embeds=( bert drone-severity ordinal-severity )
encoders=( transformer lstm gru linear )
weight_class=( uniform )
losses=( logloss )
label_schemas=( 101 111 )
poolings=( cls last )
bidirectionals=( true )
n_layers=( 3 )
n_heads=( 8 )

for dataset in "${datasets[@]}"; do
    echo "dataset: "$dataset""
    for word_embed in "${word_embeds[@]}"; do
        for encoder in "${encoders[@]}"; do
            for weight in "${weight_class[@]}"; do
                for loss in "${losses[@]}"; do
                    for label_schema in "${label_schemas[@]}"; do
                        for pooling in "${poolings[@]}"; do
                            if [ "$encoder" = "none" ]; then
                                rm -r cache_dir
                                python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir multitask
                            else
                                for n_layer in "${n_layers[@]}"; do
                                    if [ "$encoder" = "transformer" ]; then
                                        for n_head in "${n_heads[@]}"; do
                                            rm -r cache_dir
                                            # for pooling in "${poolings[@]}"; do
                                            python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --n_heads "$n_head" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir multitask
                                            # done
                                        done
                                    else
                                        for bidirectional in "${bidirectionals[@]}"; do
                                            if [ "$bidirectional" = true ]; then
                                                rm -r cache_dir
                                                python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --bidirectional --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir multitask
                                            else
                                                rm -r cache_dir
                                                python multitask.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --label_schema "$label_schema" --viz_projection --output_dir multitask
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