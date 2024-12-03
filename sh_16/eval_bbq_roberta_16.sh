EXP_FOL=/home/ubuntu/storage/bbq_roberta_acc/EXP_FOL_roberta_16
BATCH_SIZE=8
MODEL_PATH=/home/ubuntu/storage/bbq_roberta_acc/EXP_FOL_roberta_16/race_run/checkpoint-last
MAX_SEQ_LENGTH=512
BBQ_DATA=/home/ubuntu/storage/bbq_roberta_acc/BBQ/data  

python /home/ubuntu/storage/bbq_roberta_acc/lrqa/scripts/bbq_preproc.py \
    --input_data_path=${BBQ_DATA} \
    --data_path ${EXP_FOL}/bbq

for CATEGORY in Age Disability_status Gender_identity Nationality Physical_appearance Race_ethnicity Race_x_SES Race_x_gender Religion SES Sexual_orientation; do
    echo "Evaluating category: ${CATEGORY}"
    python /home/ubuntu/storage/bbq_roberta_acc/evaluation/eval.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_file ${EXP_FOL}/bbq/${CATEGORY}/validation.jsonl \
        --batch_size ${BATCH_SIZE} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --output_dir ${EXP_FOL}/bbq/${CATEGORY} > ${EXP_FOL}/bbq/${CATEGORY}/log.txt 2>&1 \
        --device cuda
done