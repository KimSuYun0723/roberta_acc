EXP_FOL=/home/ubuntu/storage/bias_supplementary_BBQ_acc/EXP_FOL_roberta_84

python /home/ubuntu/storage/bias_supplementary_BBQ_acc/evaluation/combine_results.py \
    --result_dir ${EXP_FOL}/bbq \
    --output_file ${EXP_FOL}/bbq/combined_results.json