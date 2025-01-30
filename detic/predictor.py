# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from .modeling.utils import reset_cls_test
from torch.nn import functional as F


def get_clip_embeddings(vocabulary, prompt='a '):
    from detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def get_UnSec_embeddings(vocabulary):
    from detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()

    classifier = []
    for text in vocabulary:
        emb = text_encoder(text)
        print(f"shape = {emb.shape}")
        emb = torch.mean(emb, dim=0)
        print(f"shape = {emb.shape}")
        classifier.append(emb)

    emb = torch.stack(classifier)
    # emb = F.normalize(emb)
    return emb.detach().permute(1, 0).contiguous().cpu()

BUILDIN_CLASSIFIER = {
    # FSOD
    'fsod_test_l1_baseline': "datasets/metadata_fsod/plain/fsod_clip_hrchy_l1.npy",
    'fsod_test_l2_baseline': "datasets/metadata_fsod/plain/fsod_clip_hrchy_l2.npy",
    'fsod_test_l3_baseline': "datasets/metadata_fsod/plain/fsod_clip_hrchy_l3.npy",

    'fsod_test_l1_ours': "datasets/metadata_fsod/aggr_mean/fsod_clip_hrchy_l1.npy",
    'fsod_test_l2_ours': "datasets/metadata_fsod/aggr_mean/fsod_clip_hrchy_l2.npy",
    'fsod_test_l3_ours': "datasets/metadata_fsod/aggr_mean/fsod_clip_hrchy_l3.npy",

    # iNat
    'inat_val_l1_baseline': "datasets/metadata_inat/metadata_inat_plain/inat_clip_a+cname_hrchy_l1.npy",
    'inat_val_l2_baseline': "datasets/metadata_inat/metadata_inat_plain/inat_clip_a+cname_hrchy_l2.npy",
    'inat_val_l3_baseline': "datasets/metadata_inat/metadata_inat_plain/inat_clip_a+cname_hrchy_l3.npy",
    'inat_val_l4_baseline': "datasets/metadata_inat/metadata_inat_plain/inat_clip_a+cname_hrchy_l4.npy",
    'inat_val_l5_baseline': "datasets/metadata_inat/metadata_inat_plain/inat_clip_a+cname_hrchy_l5.npy",
    'inat_val_l6_baseline': "datasets/metadata_inat/metadata_inat_plain/inat_clip_a+cname_hrchy_l6.npy",

    'inat_val_l1_ours': "datasets/metadata_inat/mixed_mean_peigen_alpha=1.0/inat_clip_hrchy_l1.npy",
    'inat_val_l2_ours': "datasets/metadata_inat/mixed_mean_peigen_alpha=1.0/inat_clip_hrchy_l2.npy",
    'inat_val_l3_ours': "datasets/metadata_inat/mixed_mean_peigen_alpha=1.0/inat_clip_hrchy_l3.npy",
    'inat_val_l4_ours': "datasets/metadata_inat/mixed_mean_peigen_alpha=1.0/inat_clip_hrchy_l4.npy",
    'inat_val_l5_ours': "datasets/metadata_inat/mixed_mean_peigen_alpha=1.0/inat_clip_hrchy_l5.npy",
    'inat_val_l6_ours': "datasets/metadata_inat/mixed_mean_peigen_alpha=1.0/inat_clip_hrchy_l6.npy",

    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    # 'lvis': "datasets/metadata_novel/metadata_lvis_wo_hrchy/aggr_mean/lvis_clip_hrchy_l1.npy",
    # 'lvis': "datasets/metadata_trimmed/metadata_lvis_wo_hrchy/aggr_mean/lvis_clip_hrchy_l1.npy",
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    # FSOD
    'fsod_l1': "fsod_test_l1",
    'fsod_l2': "fsod_test_l2",
    'fsod_l3': "fsod_test_l3",

    # iNat
    'inat_l1': "inat_val_l1",
    'inat_l2': "inat_val_l2",
    'inat_l3': "inat_val_l3",
    'inat_l4': "inat_val_l4",
    'inat_l5': "inat_val_l5",
    'inat_l6': "inat_val_l6",

    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

class VisualizationDemo(object):
    def __init__(self, cfg, args, 
        instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        if args.vocabulary == 'custom':
            PERSON = [
                "kid, which is a person, which is human",
                "child, which is a person, which is human",
                "boy, which is a person, which is human",
                "girl, which is a person, which is human",
                "student, which is a person, which is human",
            ]

            TREE = [
                "cedar, which is a tree, which is a plant",
                "oak tree, which is a tree, which is a plant",
                "maple , which is a tree, which is a plant",
                "apple tree, which is a tree, which is a plant",
                "spruce, which is a tree, which is a plant",

            ]

            FLOWER = [
                "daisy, which is a flower, which is a plant",
                "chrysanthemum, which is a flower, which is a plant",
                "peony, which is a flower, which is a plant",
                "tulip, which is a flower, which is a plant",
                "orchid, which is a flower, which is a plant",
            ]

            EQUIPMENT = [
                "glasses, which is a equipment, which is a tool",
                "helmet, which is a equipment, which is a tool",
                "screw driver, which is a equipment, which is a tool",
                "hammer, which is a equipment, which is a tool",
                "chair, which is a equipment, which is a tool",

            ]

            FOOD = [
                "burger, which is a fast food, which is food",
                "fries, which is a fast food, which is food",
                "kebab, which is a fast food, which is food",
                "fried onion ring, which is a fast food, which is food",
                "fried chicken wings, which is a fast food, which is food",
            ]

            voc_bl   = ["person", "tree", "flower", "equipment", "fast food"]
            voc_ours = [PERSON, TREE, FLOWER, EQUIPMENT, FOOD]

            # self.metadata = MetadataCatalog.get("__unused")
            # self.metadata.thing_classes = voc_bl

            # classifier = get_clip_embeddings(self.metadata.thing_classes)
            # classifier = get_UnSec_embeddings(voc_ours)

            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        elif "inat" in args.vocabulary:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.custom_vocabulary]
        elif "fsod" in args.vocabulary:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.custom_vocabulary]
        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
