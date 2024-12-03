EXP_FOL=/home/ubuntu/storage/bbq_roberta_acc/EXP_FOL_roberta_16

python /home/ubuntu/storage/bbq_roberta_acc/evaluation/combine_results.py \
    --result_dir ${EXP_FOL}/bbq \
    --output_file ${EXP_FOL}/bbq/combined_results.json