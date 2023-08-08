## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.


## Pretrained Models

You can download the [pretrained Faster R-CNN](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw) we used in the paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), which is the most time consuming step in the whole training process (it took 4 2080ti GPUs). As to the SGG model, you can follow the rest instructions to train your own, which only takes 2 GPUs to train each SGG model.

After you download the [Faster R-CNN model](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw), please extract all the files to the directory `/home/username/checkpoints/pretrained_faster_rcnn`. To train your own Faster R-CNN model, please follow the next section.

## Train and Test
By default, we will test the final model directly after training.

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use two switches ```MODEL.ROI_RELATION_HEAD.USE_GT_BOX``` and ```MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL``` to select the protocols. 

``` python
## PredCls
bash shell/predcls_train_test.sh

## SGCls
bash shell/sgcls_train_test.sh  

## SGGen  
bash shell/sgdet_train_test.sh
```  

## Perform training on Scene Graph Generation

For **Predicate Classification (PredCls)**, we need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Predefined Models
We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor. To select our predefined models, you can use ```MODEL.ROI_RELATION_HEAD.PREDICTOR```.

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```
For [VCTree](https://arxiv.org/abs/1812.01880) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
```
For our predefined Transformer Model (Note that Transformer Model needs to change SOLVER.BASE_LR to 0.001, SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000, SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000).), which is provided by [Jiaxin Shi](https://github.com/shijx12):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
```
For [Unbiased-Causal-TDE](https://arxiv.org/abs/2002.11949) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor
```

The default settings are under ```configs/e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```maskrcnn_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```

### Examples of the Training Command
Training Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```
where ```GLOVE_DIR``` is the directory used to save glove initializations, ```MODEL.PRETRAINED_DETECTOR_CKPT``` is the pretrained Faster R-CNN model you want to load, ```OUTPUT_DIR``` is the output directory used to save checkpoints and the log. Since we use the ```WarmupReduceLROnPlateau``` as the learning scheduler for SGG, ```SOLVER.STEPS``` is not required anymore.

Training Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```


## Evaluation

### Examples of the Test Command
Test Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```

Test Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10028 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgcls-exmp OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```
