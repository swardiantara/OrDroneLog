#!/bin/bash
datasets=( filtered )
word_embeds=( vector-max )
encoders=( transformer lstm gru none )
weight_class=( uniform )
losses=( cross_entropy )
poolings=( cls avg max last )
bidirectionals=( true false )
n_layers=( 3 )
n_heads=( 8 )
seeds=( 14298463 246773155 30288239 82511865 90995999 )

for dataset in "${datasets[@]}"; do
    echo "dataset: "$dataset""
    for word_embed in "${word_embeds[@]}"; do
        for encoder in "${encoders[@]}"; do
            for weight in "${weight_class[@]}"; do
                for loss in "${losses[@]}"; do
                    for seed in "${seeds[@]}"; do
                        if [ "$encoder" = "none" ]; then
                            for pooling in "${poolings[@]}"; do
                                python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --save_best_model --viz_projection --output_dir baseline --seed "$seed"
                            done
                        else
                            for n_layer in "${n_layers[@]}"; do
                                if [ "$encoder" = "transformer" ]; then
                                    for n_head in "${n_heads[@]}"; do
                                        for pooling in "${poolings[@]}"; do
                                            python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --n_heads "$n_head" --pooling "$pooling" --class_weight "$weight" --loss "$loss" --save_best_model --viz_projection --output_dir baseline --seed "$seed"
                                        done
                                    done
                                else
                                    for bidirectional in "${bidirectionals[@]}"; do
                                        if [ "$bidirectional" = true ]; then
                                            python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --bidirectional --class_weight "$weight" --pooling "$pooling" --loss "$loss" --save_best_model --viz_projection --output_dir baseline --seed "$seed"
                                        else
                                            python baseline.py --dataset "$dataset" --word_embed "$word_embed" --encoder "$encoder" --n_layers "$n_layer" --class_weight "$weight" --pooling "$pooling" --loss "$loss" --save_best_model --viz_projection --output_dir baseline --seed "$seed"
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