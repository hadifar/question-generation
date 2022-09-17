PROJECT_NAME=question_generation

SERVER=hadifar@n087-09.wall2.ilabt.iminds.be

LOCAL=~/PycharmProjects/$PROJECT_NAME

REMOTE_DIR=/groups/wall2-ilabt-iminds-be/cmsearch/users/amir/projects/$PROJECT_NAME

#while inotifywait -r -e modify,create,delete,move $LOCAL; do
#sudo rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/

#sudo rsync --delete --archive --verbose --compress --update --owner -e ssh $LOCAL/dataset_large/qg_train.json  $SERVER:$REMOTE_DIR/dataset_large/
#sudo rsync --delete --archive --verbose --compress --update --owner -e ssh $LOCAL/dataset_large/test_brusselsairport.json  $SERVER:$REMOTE_DIR/dataset_large/
#sudo rsync --delete --archive --verbose --compress --update -e ssh $LOCAL/eval_runs/Oct13_xlm_roberta-base-4epoch/  $SERVER:$REMOTE_DIR/eval_runs/Oct13_xlm_roberta-base-4epoch/

#sudo rsync --delete --archive --verbose --compress --update -e ssh $LOCAL/dataset_final2_small/  $SERVER:$REMOTE_DIR/dataset_final2_small/
#sudo rsync --delete --archive --verbose --compress --update -e ssh $LOCAL/dataset_final2/  $SERVER:$REMOTE_DIR/dataset_final2/
#sudo rsync --delete --archive --verbose --compress --update -e ssh $LOCAL/data_complaint/  $SERVER:$REMOTE_DIR/data_complaint/
#sudo rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/data_small_xlm15/  $SERVER:$REMOTE_DIR/data_small_xlm15/

#rsync --delete --archive --verbose --compress --exclude 'cache/' --exclude 'train_data_qg_highlight_qg_format_t5_full.pt' --exclude 'valid_data_qg_highlight_qg_format_t5_full.pt' --update -e ssh $LOCAL/data/  $SERVER:$REMOTE_DIR/data/

#script.sh
rsync --include './' --include '*.py' --exclude '*' --exclude 'qgenv/' --exclude 'data/' --delete --archive --verbose --compress --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/
rsync --include './' --include '*.txt' --exclude '*' --exclude 'qgenv/' --exclude 'data/' --delete --archive --verbose --compress --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/
rsync --include './' --include '*.sh' --exclude '*' --exclude 'qgenv/' --exclude 'data/' --delete --archive --verbose --compress --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/
rsync --include './' --include '*.json' --exclude '*' --exclude 'qgenv/' --exclude 'data/' --delete --archive --verbose --compress --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/

#rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/data/cc_openstax_cloze_gen/  $SERVER:$REMOTE_DIR/data/cc_openstax_cloze_gen/

rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/raw_data/dev-v2.0.json  $SERVER:$REMOTE_DIR/raw_data/dev-v2.0.json
rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/raw_data/qg_train.json  $SERVER:$REMOTE_DIR/raw_data/qg_train.json
rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/raw_data/qg_valid.json  $SERVER:$REMOTE_DIR/raw_data/qg_valid.json
rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/raw_data/qg_dutch.json  $SERVER:$REMOTE_DIR/raw_data/dutch_qg.json

#rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/data/valid_data_qg_conv_highlight_qg_format_t5.pt  $SERVER:$REMOTE_DIR/data/valid_data_qg_conv_highlight_qg_format_t5.pt
#rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/data/train_data_qg_gen_highlight_qg_format_t5.pt  $SERVER:$REMOTE_DIR/data/train_data_qg_gen_highlight_qg_format_t5.pt
#rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/data/valid_data_qg_gen_highlight_qg_format_t5.pt  $SERVER:$REMOTE_DIR/data/valid_data_qg_gen_highlight_qg_format_t5.pt
#
#rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/t5_qg_tokenizer/  $SERVER:$REMOTE_DIR/t5_qg_tokenizer/


ssh $SERVER sudo chmod -R 777 $REMOTE_DIR/script.sh
ssh $SERVER sudo chmod -R 777 $REMOTE_DIR/
ssh $SERVER sudo chmod -R 777 $REMOTE_DIR/raw_data/
#done
