#!/bin/sh
# Default values of arguments
EXP_NAME='test'
EXP_ID=0
DATA_PATH='/ahc/work3/kosuke-t/data/SRHDA/15-18/bert/new/'
TRAIN_BATCH_SIZE=4
LEARNING_RATE=()
NUM_TRAIN_EPOCHS=10
OP_TIMES=10
N_OPERATION="False"
SAVE_DIR='/ahc/work3/kosuke-t/SRHDA/bert/log'
TEST_LANGS='cs-en_de-en_fi-en_lv-en_ru-en_tr-en_zh-en'
BERT_CONFIG=/ahc/work3/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/bert_config.json
VOCAB_FILE=/ahc/work3/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/vocab.txt
INIT_CHECKPOINT=/ahc/work3/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/bert_model.ckpt
ADDSRC='False'
ONLYSRC='False'
ONLYREF='False'
TRAIN_SHRINK='1.0'

# Loop through arguments and process them
while [ -n "$1" ];
do
    arg="$1"
    case $arg in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --learning_rate)
            shift
            LEARNING_RATE+=("$1")
            LEARNING_RATE+=("$2")
            LEARNING_RATE+=("$3")
#             LEARNING_RATE+=("$4")
            shift 3
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --op_times)
          OP_TIMES="$2"
          shift 2
          ;;
        --n_operation)
          N_OPERATION="$2"
          shift 2
          ;;
        --save_dir)
          SAVE_DIR="$2"
          shift 2
          ;;
        --test_langs)
          TEST_LANGS="$2"
          shift 2
          ;;
		--bert_config_file)
		  BERT_CONFIG="$2"
		  shift 2
		  ;;
		--vocab_file)
		  VOCAB_FILE="$2"
		  shift 2
		  ;;
		--init_checkpoint)
		  INIT_CHECKPOINT="$2"
		  shift 2
		  ;;
		--addSRC)
		  ADDSRC="$2"
		  shift 2
		  ;;
		--onlySRC)
		  ONLYSRC="$2"
		  shift 2
		  ;;
		--onlyREF)
		  ONLYREF="$2"
		  shift 2
		  ;;
		--train_shrink)
		  TRAIN_SHRINK="$2"
		  shift 2
		  ;;
        --*)
          echo NOT DEFINED ARGUMENTS: "$arg"
          echo "$arg"
          exit
          ;;
    esac
done

if [ ${#LEARNING_RATE[@]} -eq 0 ]; then
  LEARNING_RATE=()
  #LEARNING_RATE=(5e-5 3e-5 2e-5)
  LEARNING_RATE=(5e-5)
  #LEARNING_RATE=(3e-5 2e-5 1e-5)
fi

for i in `seq "${OP_TIMES}"` ; do
    N_OPERATION="$i"
    #EXP_ID=0
    for opt in "${LEARNING_RATE[@]}" ; do
        for j in 4; do
			for k in 16; do
				EXP_ID=`expr "$EXP_ID" + 1`
# 				if [ ${EXP_ID} -gt 108 ]; then
				  python bertTensor.py \
					--exp_name "$EXP_NAME" \
					--exp_id "$EXP_ID" \
					--data_dir "$DATA_PATH" \
					--train_batch_size "$k" \
					--learning_rate "$opt" \
					--num_train_epochs "$j" \
					--n_operation "$i" \
					--bert_config_file "$BERT_CONFIG" \
					--vocab_file "$VOCAB_FILE" \
					--output_dir /ahc/work3/kosuke-t/SRHDA/bert/log/ \
					--init_checkpoint "$INIT_CHECKPOINT" \
					--test_langs "$TEST_LANGS" \
					--addSRC "$ADDSRC" \
					--onlySRC "$ONLYSRC" \
					--onlyREF "$ONLYREF" \
					--train_shrink "$TRAIN_SHRINK"
					#rm -rf /ahc/work3/kosuke-t/SRHDA/bert/log/${EXP_NAME}/${EXP_ID}/model*
# 				fi
				
			done
        done
    done
done

# for i in `seq "${OP_TIMES}"` ; do
#     N_OPERATION="$i"
#     EXP_ID=0
#     for opt in "${LEARNING_RATE[@]}" ; do
#         for j in 3 4 ; do
# 			for k in 16 32; do
# 				EXP_ID=`expr "$EXP_ID" + 1`
# 				python Shimanaka-san_model.py \
# 				--exp_name "$EXP_NAME" \
# 				--exp_id "$EXP_ID" \
# 				--data_path "$DATA_PATH" \
# 				--train_batch_size "$k" \
# 				--learning_rate "$opt" \
# 				--num_train_epochs "$j" \
# 				--n_operation "$i" \
# 				--bert_config_file /project/nakamura-lab08/Work/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/bert_config.json \
# 				--vocab_file /project/nakamura-lab08/Work/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/vocab.txt \
# 				--output_dir /project/nakamura-lab08/Work/kosuke-t/SRHDA/bert/log/ \
# 				--init_checkpoint /project/nakamura-lab08/Work/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/bert_model.ckpt
# 			done
#         done
#     done
# done

# python bertTensor.py \
#             --exp_name "$EXP_NAME" \
#             --exp_id 1 \
#             --data_dir "$DATA_PATH" \
#             --train_batch_size 16 \
#             --learning_rate 3e-5 \
# 			--num_train_epochs 1 \
# 			--n_operation 1 \
# 			--bert_config_file /project/nakamura-lab08/Work/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/bert_config.json \
# 			--vocab_file /project/nakamura-lab08/Work/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/vocab.txt \
# 			--output_dir /project/nakamura-lab08/Work/kosuke-t/SRHDA/bert/log/ \
# 			--init_checkpoint /project/nakamura-lab08/Work/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12/bert_model.ckpt