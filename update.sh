PROJECT_NAME=question_generation

SERVER=hadifar@n086-03.wall2.ilabt.iminds.be

LOCAL=/home/amir/PycharmProjects/$PROJECT_NAME

REMOTE_DIR=/groups/wall2-ilabt-iminds-be/cmsearch/users/amir/projects/$PROJECT_NAME

#while inotifywait -r -e modify,create,delete,move $LOCAL; do
#sudo rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/

#sudo rsync --delete --archive --verbose --compress --update --owner -e ssh $LOCAL/dataset_large/train.json  $SERVER:$REMOTE_DIR/dataset_large/
#sudo rsync --delete --archive --verbose --compress --update --owner -e ssh $LOCAL/dataset_large/test_brusselsairport.json  $SERVER:$REMOTE_DIR/dataset_large/
#sudo rsync --delete --archive --verbose --compress --update -e ssh $LOCAL/eval_runs/Oct13_xlm_roberta-base-4epoch/  $SERVER:$REMOTE_DIR/eval_runs/Oct13_xlm_roberta-base-4epoch/

#sudo rsync --delete --archive --verbose --compress --update -e ssh $LOCAL/dataset_final2_small/  $SERVER:$REMOTE_DIR/dataset_final2_small/
#sudo rsync --delete --archive --verbose --compress --update -e ssh $LOCAL/dataset_final2/  $SERVER:$REMOTE_DIR/dataset_final2/
#sudo rsync --delete --archive --verbose --compress --update -e ssh $LOCAL/data_complaint/  $SERVER:$REMOTE_DIR/data_complaint/
#sudo rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/data_small_xlm15/  $SERVER:$REMOTE_DIR/data_small_xlm15/
#sudo rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/data_answer_selection/  $SERVER:$REMOTE_DIR/data_answer_selection/

#script.sh
sudo rsync --include './' --include '*.py' --exclude '*' --delete --archive --verbose --compress --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/
sudo rsync --include './' --include '*.txt' --exclude '*' --delete --archive --verbose --compress --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/
sudo rsync --include './' --include '*.sh' --exclude '*' --delete --archive --verbose --compress --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/
sudo rsync --include './' --include '*.json' --exclude '*' --delete --archive --verbose --compress --update -e ssh $LOCAL/  $SERVER:$REMOTE_DIR/

#sudo rsync --delete --archive --verbose --compress --owner --update -e ssh $LOCAL/dataset_cache_XLMTokenizer  $SERVER:$REMOTE_DIR/dataset_cache_XLMTokenizer

ssh hadifar@n086-03.wall2.ilabt.iminds.be sudo chmod -R 777 $REMOTE_DIR/script.sh
ssh hadifar@n086-03.wall2.ilabt.iminds.be sudo chmod -R 777 $REMOTE_DIR/run.sh
ssh hadifar@n086-03.wall2.ilabt.iminds.be sudo chmod -R 777 $REMOTE_DIR/
#done