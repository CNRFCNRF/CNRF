export CUDA_VISIBLE_DEVICES="0"
export num_gpu=1
export model_config="e2e_relation_X_101_32_8_FPN_1x"
export predictor="MotifPredictor"
export test_batch=1
export data_dir="/home/user/dataset/vg"
export model_dir="/home/user/test/wight"
export output_dir="/home/user/outputs/test/${predictor}/predcls-$(date "+%m-%d-%H:%M:%S")"

    python -m torch.distributed.launch --master_port 10028 --nproc_per_node=${num_gpu} \
    tools/relation_test_net.py --config-file configs/${model_config}.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR ${predictor} TEST.IMS_PER_BATCH ${test_batch} \
    MODEL.WEIGHT ${model_dir}/model_final.pth \
    DATASETS.DATA_DIR ${data_dir} \
    OUTPUT_DIR ${output_dir}