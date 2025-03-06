#!/bin/bash
datasets=( filtered )
word_embeds=( bert drone-severity ordinal-severity vector-ordinal )
encoders=( transformer )
weight_class=( uniform )
losses=( cross_entropy )
poolings=( cls )
bidirectionals=( true )
n_layers=( 3 )
n_heads=( 8 )

for dataset in "${datasets[@]}"; do
    echo "dataset: "$dataset""
    for word_embed in "${word_embeds[@]}"; do
        for encoder in "${encoders[@]}"; do
            for weight in "${weight_class[@]}"; do
                for loss in "${losses[@]}"; do
                    if [ "$encoder" = "none" ]; then
                        for pooling in "${poolings[@]}"; do
                            python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --save_best_model --viz_projection --output_dir baseline
                        done
                    else
                        for n_layer in "${n_layers[@]}"; do
                            if [ "$encoder" = "transformer" ]; then
                                for n_head in "${n_heads[@]}"; do
                                    for pooling in "${poolings[@]}"; do
                                        python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --n_heads "$n_head" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --save_best_model --viz_projection --output_dir baseline
                                    done
                                done
                            else
                                for bidirectional in "${bidirectionals[@]}"; do
                                    if [ "$bidirectional" = true ]; then
                                        python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --bidirectional --class_weight "$weight" --loss "$loss" --save_best_model --viz_projection --output_dir baseline
                                    else
                                        python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --class_weight "$weight" --loss "$loss" --save_best_model --viz_projection --output_dir baseline
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