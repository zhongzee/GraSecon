# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Mingxuan Liu from https://github.com/facebookresearch/Detic/blob/main/train_net.py

import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    inference_on_dataset_IFC,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config

sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.config import add_detic_config
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.data.custom_dataset_dataloader import  build_custom_train_loader
from detic.data.custom_dataset_mapper import CustomDatasetMapper, DetrDatasetMapper
from detic.custom_solver import build_custom_optimizer
from detic.evaluation.oideval import OIDEvaluator
from detic.evaluation.custom_coco_eval import CustomCOCOEvaluator
from detic.modeling.utils import reset_cls_test
from detic.evaluation.inateval import INATEvaluator
from detic.evaluation.fsodeval import FSODEvaluator

logger = logging.getLogger("detectron2")


import pandas as pd
import os
import re
def do_test(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        if cfg.MODEL.RESET_CLS_TESTS:
            reset_cls_test(
                model,
                cfg.MODEL.TEST_CLASSIFIERS[d],
                cfg.MODEL.TEST_NUM_CLASSES[d]
            )

        # create data loader and mapper
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))

        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper) #这个里面设置batchsize detectron2/data/build.py

        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}_{}".format(dataset_name, cfg.MODEL.TEST_CLASSIFIERS[d])) # Miu: added classifier name into output name

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # create evaluator
        if evaluator_type == "lvis" or cfg.GEN_PSEDO_LABELS:
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            if dataset_name == 'coco_generalized_zeroshot_val':
                # Additionally plot mAP for 'seen classes' and 'unseen classes'
                evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'oid':
            evaluator = OIDEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'inat':
            evaluator = INATEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'fsod':
            evaluator = FSODEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type

        # perform evaluation
        results[dataset_name] = inference_on_dataset_IFC(model,data_loader, evaluator)

        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results[dataset_name])

    if len(results) == 1:
        results = list(results.values())[0]

    # 保存结果到 CSV 文件使用 pandas
    print("results=",results)
    df_rows = []
    for dataset, metrics in results.items():
        for metric, value in metrics.items():
            df_rows.append({
                'Dataset': dataset,
                'Metric': metric,
                'Value': value
            })

    df = pd.DataFrame(df_rows)
    # OUTPUT_DIR = "./UnSec"+cfg.OUTPUT_DIR
    # print("cfg.OUTPUT_DIR=",OUTPUT_DIR)
    csv_file = os.path.join(cfg.OUTPUT_DIR, 'evaluation_results.csv')
    df.to_csv(csv_file, index=False)

    print(f"Results saved to {csv_file}")

    return results

# def do_test(cfg, model):
#     results = OrderedDict()
#     for d, dataset_name in enumerate(cfg.DATASETS.TEST):
#         if cfg.MODEL.RESET_CLS_TESTS:
#             reset_cls_test(
#                 model,
#                 cfg.MODEL.TEST_CLASSIFIERS[d],
#                 cfg.MODEL.TEST_NUM_CLASSES[d]
#             )
#
#         # create data loader and mapper
#         mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
#             else DatasetMapper(
#                 cfg, False, augmentations=build_custom_augmentation(cfg, False))
#
#         data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
#
#         output_folder = os.path.join(
#             cfg.OUTPUT_DIR, "inference_{}_{}".format(dataset_name, cfg.MODEL.TEST_CLASSIFIERS[d])) # Miu: added classifier name into output name
#
#         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#
#         # create evaluator
#         if evaluator_type == "lvis" or cfg.GEN_PSEDO_LABELS:
#             evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
#         elif evaluator_type == 'coco':
#             if dataset_name == 'coco_generalized_zeroshot_val':
#                 # Additionally plot mAP for 'seen classes' and 'unseen classes'
#                 evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
#             else:
#                 evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
#         elif evaluator_type == 'oid':
#             evaluator = OIDEvaluator(dataset_name, cfg, True, output_folder)
#         elif evaluator_type == 'inat':
#             evaluator = INATEvaluator(dataset_name, cfg, True, output_folder)
#         elif evaluator_type == 'fsod':
#             evaluator = FSODEvaluator(dataset_name, cfg, True, output_folder)
#         else:
#             assert 0, evaluator_type
#
#         # perform evaluation
#         results[dataset_name] = inference_on_dataset(model, data_loader, evaluator)
#
#         if comm.is_main_process():
#             logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#             print_csv_format(results[dataset_name])
#
#     if len(results) == 1:
#         results = list(results.values())[0]
#     #########
#     print("当前的results结果是",results)
#     # OrderedDict([('bbox', {'AP50': 0.7499999999999998})])
#     return results


def do_train(cfg, model, resume=False):
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == 'SGD'
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != 'full_model'
        assert cfg.SOLVER.BACKBONE_MULTIPLIER == 1.
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    if not resume:
        start_iter = 0

    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    use_custom_mapper = cfg.WITH_IMAGE_LABELS
    MapperClass = CustomDatasetMapper if use_custom_mapper else DatasetMapper

    mapper = MapperClass(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
        DetrDatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == 'DETR' else \
        MapperClass(cfg, True, augmentations=build_custom_augmentation(cfg, True))

    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    else:
        data_loader = build_custom_train_loader(cfg, mapper=mapper) # 进入到registry_lvis_v1函数里

    if cfg.FP16:
        scaler = GradScaler()

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items()
            )
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if cfg.FP16:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if cfg.TEST.EVAL_PERIOD > 0 and iteration % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter:
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))


# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     add_centernet_config(cfg)
#     add_detic_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     if '/auto' in cfg.OUTPUT_DIR:
#         file_name = os.path.basename(args.config_file)[:-5]
#         cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
#         logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
#     cfg.freeze()
#     default_setup(cfg, args)
#     setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detic")
#     return cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Define the regex pattern to match '/auto' exactly as a directory name
    pattern = r'(^|/)(auto)(/|$)'

    # Replace '/auto' if it matches the pattern exactly
    matches = re.search(pattern, cfg.OUTPUT_DIR)
    if matches:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = re.sub(pattern, r'\1{}\3'.format(file_name), cfg.OUTPUT_DIR)
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detic")
    return cfg

def main(args):
    cfg = setup(args)

    # build model from configuration file
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    OUTPUT_DIR = "./UnSec"+cfg.OUTPUT_DIR
    # >>> eval only >>>
    if args.eval_only: # 这里选择是eval_only
        DetectionCheckpointer(model, save_dir=OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return do_test(cfg, model)
    # <<< eval only <<<
    # >>> train chunk >>>
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )

    do_train(cfg, model, resume=args.resume)
    # <<< train chunk <<<
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser()
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        if args.dist_url == 'host':
            args.dist_url = 'tcp://{}:12345'.format(
                os.environ['SLURM_JOB_NODELIST'])
        elif not args.dist_url.startswith('tcp'):
            tmp = os.popen(
                    'echo $(scontrol show job {} | grep BatchHost)'.format(
                        args.dist_url)
                ).read()
            tmp = tmp[tmp.find('=') + 1: -1]
            args.dist_url = 'tcp://{}:12345'.format(tmp)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

"""
训练
# python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml

# 测试

--num-gpus 1 
--config-file ./configs_detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml
--eval-only
DATASETS.TEST "('inat_val_l1', 'inat_val_l2', 'inat_val_l3','inat_val_l4','inat_val_l5','inat_val_l6',)"
MODEL.WEIGHTS ./models/detic/cross_eval/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
MODEL.RESET_CLS_TESTS True
MODEL.TEST_CLASSIFIERS "('./nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l1.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l2.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l3.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l4.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l5.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l6.npy', )"
MODEL.TEST_NUM_CLASSES "(5, 18, 61, 184, 317, 500,)"
MODEL.MASK_ON False

--num-gpu 1
--config-file ./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml
--eval-only
DATASETS.TEST "('inat_val_l1', 'inat_val_l2', 'inat_val_l3','inat_val_l4','inat_val_l5','inat_val_l6',)"
MODEL.WEIGHTS ./models/detic/cross_eval/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
MODEL.RESET_CLS_TESTS True
MODEL.TEST_CLASSIFIERS "('./nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l1.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l2.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l3.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l4.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l5.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l6.npy', )"
MODEL.TEST_NUM_CLASSES "(5, 18, 61, 184, 317, 500,)"
MODEL.MASK_ON False


# RN50

--num-gpus 1
--config-file ./configs_detic/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml
--eval-only
DATASETS.TEST "('inat_val_l1', 'inat_val_l2', 'inat_val_l3','inat_val_l4','inat_val_l5','inat_val_l6',)"
MODEL.WEIGHTS ./models/detic/lvis_std/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth
MODEL.RESET_CLS_TESTS True
MODEL.TEST_CLASSIFIERS "('./nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l1.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l2.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l3.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l4.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l5.npy', './nexus/inat/vitB32/inat_llm_detail_answers_1028/inat_clip_hrchy_l6.npy', )"
MODEL.TEST_NUM_CLASSES "(5, 18, 61, 184, 317, 500,)"
MODEL.MASK_ON False


--num-gpus 1
--config-file ./configs_detic/Detic_LbaseCCimg_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml
--eval-only
DATASETS.TEST ('lvis_v1_val',)
MODEL.WEIGHTS ./models/detic/lvis_ovod/Detic_LbaseCCimg_CLIP_R5021k_640b64_4x_ft4x_max-size.pth
MODEL.RESET_CLS_TESTS True
MODEL.TEST_CLASSIFIERS './nexus/lvis/baseline/lvis_clip_hrchy_l1.npy'
MODEL.TEST_NUM_CLASSES (1203,)
MODEL.MASK_ON False


--num-gpus 1
--config-file ./configs_detic/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml
--eval-only
DATASETS.TRAIN "('lvis_v1_train',)"
DATASETS.TEST "('lvis_v1_val',)"
MODEL.WEIGHTS ./models/detic/lvis_ovod/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth
MODEL.RESET_CLS_TESTS True
MODEL.TEST_CLASSIFIERS "('./nexus/lvis/baseline/lvis_clip_hrchy_l1.npy',)"
MODEL.TEST_NUM_CLASSES "(1203,)"
MODEL.MASK_ON False



"""