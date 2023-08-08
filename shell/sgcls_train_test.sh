export CUDA_VISIBLE_DEVICES="0"
export num_gpu=1
export model_config="e2e_relation_X_101_32_8_FPN_1x"
export predictor="MotifPredictor"
export train_batch=8
export test_batch=4
export glove_dir="/home/user/Glove"
export pretrained_model_dir="/home/user/wight/pretrained_faster_rcnn"
export output_dir="/home/user/outputs/${predictor}/sgcls-$(date "+%m-%d-%H:%M:%S")"
export test_final_model=True

    echo "#####################  ${predictor}--sgcls  #####################"
    python -m torch.distributed.launch --master_port 10027 --nproc_per_node=${num_gpu} \
    tools/relation_train_net.py --config-file configs/${model_config}.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} \
    SOLVER.IMS_PER_BATCH ${train_batch}  TEST.IMS_PER_BATCH ${test_batch} \
    DTYPE float16 SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    GLOVE_DIR ${glove_dir} \
    MODEL.PRETRAINED_DETECTOR_CKPT ${pretrained_model_dir}/model_final.pth \
    OUTPUT_DIR ${output_dir} \
    TEST_FINAL_MODEL ${test_final_model}