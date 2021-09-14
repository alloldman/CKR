search_str='--job search
--max_episode_len 40
--early_stop
--logit
--distance_aware
--distance_weight 0.5'

train='python train.py'
search='python run_search.py'
debug='python train.py --debug --image_feature_type none'

TORUN='--experiment_name check-train-1
--feedback_method sample2step --instruction_from reverie --batch_size=100
--en_nhead 6 --en_nlayer 2 --use_glove
--num_multihead 8 --num_layer 4
--max_degree 15 --object_top_n 5 --short_cut --num_gcn 3
--use_angle_distance_reward --soft_room_label
--loss_weight 5 1 0 1'

case $1 in
'train')
CUDA_VISIBLE_DEVICES=$2 $train $TORUN;;
'search')
CUDA_VISIBLE_DEVICES=$3 $search $TORUN $search_str --load_follower $2;;
'debug')
CUDA_VISIBLE_DEVICES=$2 $debug $TORUN;;
esac
