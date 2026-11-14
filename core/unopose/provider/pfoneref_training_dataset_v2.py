import logging
import os
import os.path as osp
import cv2
import numpy as np
import time
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from detectron2.utils.logger import log_first_n

import imgaug.augmenters as iaa
from imgaug.augmenters import (
    Sequential,
    SomeOf,
    OneOf,
    Sometimes,
    WithColorspace,
    WithChannels,
    Noop,
    Lambda,
    AssertLambda,
    AssertShape,
    Scale,
    CropAndPad,
    Pad,
    Crop,
    Fliplr,
    Flipud,
    Superpixels,
    ChangeColorspace,
    PerspectiveTransform,
    Grayscale,
    GaussianBlur,
    AverageBlur,
    MedianBlur,
    Convolve,
    Sharpen,
    Emboss,
    EdgeDetect,
    DirectedEdgeDetect,
    Add,
    AddElementwise,
    AdditiveGaussianNoise,
    Multiply,
    MultiplyElementwise,
    Dropout,
    CoarseDropout,
    Invert,
    ContrastNormalization,
    Affine,
    PiecewiseAffine,
    ElasticTransformation,
    pillike,
    LinearContrast,
)  # noqa

from core.unopose.utils.data_utils import (
    load_im,
    load_json,
    io_load_gt,
    io_load_masks,
    backproject,
    get_resize_rgb_choose,
    get_random_rotation,
    get_bbox,
)

# from core.csrc.fps.fps_utils import farthest_point_sampling
from core.unopose.utils.model_utils import pairwise_distance

logger = logging.getLogger(__name__)


class DatasetPoseFreeOneRefv2:
    """Pose free one reference for fast speed"""

    def __init__(self, cfg, num_img_per_epoch=-1):
        self.cfg = cfg

        self.data_dir = cfg.data_dir  # datasets/MegaPose-Training-Data
        self.num_img_per_epoch = num_img_per_epoch
        self.min_visib_px = cfg.min_px_count_visib  # 512
        self.min_visib_frac = cfg.min_visib_fract  # 0.1
        self.dilate_mask = cfg.dilate_mask  # True
        self.rgb_mask_flag = cfg.rgb_mask_flag  # True
        self.shift_range = cfg.shift_range  # 0.01 for translation
        self.img_size = cfg.img_size  # 224
        self.n_sample_observed_point = cfg.n_sample_observed_point  # 2048
        self.n_sample_model_point = cfg.n_sample_model_point  # 2048
        self.n_sample_template_point = cfg.n_sample_template_point  # NOTE: 5000
        self.rgb_to_bgr = cfg.get("rgb_to_bgr", False)  # sam6d is True, which is weird

        self.data_paths = [
            osp.join("MegaPose-GSO", "train_pbr_web"),
            osp.join("MegaPose-ShapeNetCore", "train_pbr_web"),
        ]
        self.model_paths = [
            osp.join(self.data_dir, "MegaPose-GSO", "Google_Scanned_Objects"),
            osp.join(self.data_dir, "MegaPose-ShapeNetCore", "shapenetcorev2"),
        ]
        # self.templates_paths = [
        #     osp.join(self.data_dir, "MegaPose-GSO", "templates"),
        #     osp.join(self.data_dir, "MegaPose-ShapeNetCore", "templates"),
        # ]
        self.templates_paths = [
            osp.join(self.data_dir, "megapose_gso_fixed_obj_id_to_visib0_8_scene_im_inst_ids.json"),
            osp.join(self.data_dir, "megapose_shapenetcore_fixed_obj_id_to_visib0_8_scene_im_inst_ids.json"),
        ]
        self.valid_insts_paths = [
            osp.join(self.data_dir, "megapose_gso_fixed_valid_inst_ids.json"),
            osp.join(self.data_dir, "megapose_shapenetcore_fixed_valid_inst_ids.json"),
        ]

        cache = self._index_cache_path()
        if osp.exists(cache):  # compact index built earlier for the same data: skip the ~4 GB json path
            self._load_index(cache)
        else:
            self._build_index()
            self._save_index(cache)

        # gdrnpp aug
        aug_code = (
            "Sequential(["
            "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
            "Sometimes(0.4, GaussianBlur((0., 3.))),"
            "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
            "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
            "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
            "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
            "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
            "Sometimes(0.3, Invert(0.2, per_channel=True)),"
            "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
            "Sometimes(0.5, Multiply((0.6, 1.4))),"
            "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
            "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
            "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"
            "], random_order=True)"
            # cosy+aae
        )
        self.color_augmentor = eval(aug_code)
        # [0, 255] to float [0, 1], rgb normalization
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def _build_index(self):
        self.dataset_paths = []
        self.available_shards = {}  # dataset_type -> set of locally downloaded shard ids
        for f in self.data_paths:
            key_shards = load_json(osp.join(self.data_dir, f, "key_to_shard.json"))
            # only keep entries whose shard is actually present locally, so we don't
            # waste retries re-sampling shards that were never downloaded
            local_shard_dir = osp.join(self.data_dir, f)
            available_shards = {int(d) for d in os.listdir(local_shard_dir) if d.isdigit()}
            self.available_shards[f.split("/")[0][9:]] = available_shards  # "GSO" / "ShapeNetCore"
            for k in tqdm(key_shards.keys()):
                shard_id = key_shards[k]
                if shard_id not in available_shards:
                    continue
                # path_name = osp.join(f, "shard-" + f"{key_shards[k]:06d}", k)
                path_name = osp.join(f, f"{shard_id:06d}", k)
                self.dataset_paths.append(path_name)

        # optionally restrict to a fixed-seed random subset representing X% of the FULL
        # original dataset (2,008,971 images), drawn from whatever is currently downloaded.
        # Works no matter how much (or how little, or in what order) has been downloaded so
        # far - it always samples `int(2008971 * subset_fraction)` images, so re-running with
        # the same subset_fraction/subset_seed later (after downloading more shards) is still
        # comparable as long as the earlier subset stayed a subset of the later pool.
        # (ported from COG's core/COG/provider/pfoneref_training_dataset_v2.py)
        subset_fraction = self.cfg.get("subset_fraction", None)
        if subset_fraction is not None:
            FULL_DATASET_LEN = 2008971  # size of the complete MegaPose GSO+ShapeNetCore train set
            subset_seed = self.cfg.get("subset_seed", 42)
            target_n = int(FULL_DATASET_LEN * subset_fraction)
            if target_n < len(self.dataset_paths):
                rng = np.random.RandomState(subset_seed)
                chosen = np.sort(rng.choice(len(self.dataset_paths), size=target_n, replace=False))
                self.dataset_paths = [self.dataset_paths[i] for i in chosen]
                logger.info(
                    f"subset_fraction={subset_fraction} (seed={subset_seed}): "
                    f"using {target_n} / {FULL_DATASET_LEN} images (fixed random subset)"
                )
            else:
                logger.warning(
                    f"subset_fraction={subset_fraction} needs {target_n} images, "
                    f"but only {len(self.dataset_paths)} are downloaded locally; using all of them"
                )

        self.length = len(self.dataset_paths)
        logger.info("Total {} images .....".format(self.length))

        # load reference informations
        self.templates_infos = {}
        self.templates_infos["GSO"] = load_json(self.templates_paths[0])
        self.templates_infos["ShapeNetCore"] = load_json(self.templates_paths[1])
        # same fix as dataset_paths above: reference templates also point at shard ids
        # across the full dataset, so filter out ones we haven't downloaded locally
        for dataset_type, shards in self.available_shards.items():
            for obj_id, tem_list in self.templates_infos[dataset_type].items():
                self.templates_infos[dataset_type][obj_id] = [
                    tem_info for tem_info in tem_list if tem_info[0] in shards
                ]
        logger.info("loaded templates infos")

        self.model_info = [load_json(osp.join(self.data_dir, self.data_paths[0], "gso_models.json"))]

        self.model_info.append(load_json(osp.join(self.data_dir, self.data_paths[1], "shapenet_models.json")))

        # load valid insts
        self.valid_insts = {}
        self.valid_insts["GSO"] = load_json(self.valid_insts_paths[0])
        self.valid_insts["ShapeNetCore"] = load_json(self.valid_insts_paths[1])
        logger.info("loaded valid insts")
        self._compact_index()

    def _index_cache_path(self):
        """Cache file for the compact index, keyed by the local shards, the index json files and the subset setting."""
        import hashlib
        import json

        parts = [sorted(int(d) for d in os.listdir(osp.join(self.data_dir, f)) if d.isdigit()) for f in self.data_paths]
        for path in self.templates_paths + self.valid_insts_paths + [osp.join(self.data_dir, f, "key_to_shard.json") for f in self.data_paths]:
            st = os.stat(path)
            parts.append([osp.realpath(path), st.st_size, int(st.st_mtime)])
        parts.append([self.cfg.get("subset_fraction", None), self.cfg.get("subset_seed", 42)])
        h = hashlib.md5(json.dumps(parts).encode()).hexdigest()[:16]
        return osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "../../../output/cache/megapose_index", f"{h}.npz"))

    def _save_index(self, cache):
        arrays = {"dataset_paths": self.dataset_paths, "valid_offsets": self.valid_offsets, "valid_flat": self.valid_flat}
        for t, rng in self.tem_range.items():
            arrays[f"{t}.obj"] = np.asarray(list(rng.keys()), dtype=np.bytes_)
            arrays[f"{t}.range"] = np.asarray(list(rng.values()), dtype=np.int64).reshape(-1, 2)
            arrays[f"{t}.shard"], arrays[f"{t}.key"], arrays[f"{t}.inst"] = self.tem_shard[t], self.tem_key[t], self.tem_inst[t]
        os.makedirs(osp.dirname(cache), exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, cache)  # atomic: ranks starting together may both write
        logger.info(f"saved compact sample index to {cache}")

    def _load_index(self, cache):
        z = np.load(cache, allow_pickle=False)
        self.dataset_paths, self.valid_offsets, self.valid_flat = z["dataset_paths"], z["valid_offsets"], z["valid_flat"]
        self.tem_range, self.tem_shard, self.tem_key, self.tem_inst = {}, {}, {}, {}
        for t in ("GSO", "ShapeNetCore"):
            self.tem_range[t] = {o.decode(): (int(a), int(b)) for o, (a, b) in zip(z[f"{t}.obj"], z[f"{t}.range"])}
            self.tem_shard[t], self.tem_key[t], self.tem_inst[t] = z[f"{t}.shard"], z[f"{t}.key"], z[f"{t}.inst"]
        self.length = len(self.dataset_paths)
        self.model_info = [load_json(osp.join(self.data_dir, self.data_paths[0], "gso_models.json"))]
        self.model_info.append(load_json(osp.join(self.data_dir, self.data_paths[1], "shapenet_models.json")))
        logger.info(f"Total {self.length} images ..... (compact index from {cache})")

    def _compact_index(self):
        """Store the sample index as flat numpy arrays instead of millions of small Python objects.

        As lists/dicts (~16.7M template entries, 2M paths) the index takes ~4 GB per process, and every
        dataloader worker slowly copies all of it (refcount writes and the cyclic GC touch every object's page,
        breaking fork's copy-on-write), so 2 ranks x 8 workers ran the host out of RAM. Numpy arrays hold no
        per-entry objects and stay shared. Same entries, same order, same RNG calls as before.
        """
        paths = self.dataset_paths
        valid_counts = np.zeros(len(paths), dtype=np.int64)
        valid_flat = []
        for i, path_head in enumerate(paths):  # valid instances aligned to dataset index (was keyed by "shard/key")
            dataset_type = path_head.split("/")[0][9:]
            shard_name, key_name = path_head.split("/")[-2:]
            v = self.valid_insts[dataset_type].get(f"{shard_name}/{key_name}", [])
            valid_counts[i] = len(v)
            valid_flat.extend(v)
        self.valid_offsets = np.concatenate([[0], np.cumsum(valid_counts)])
        self.valid_flat = np.asarray(valid_flat, dtype=np.int64)
        self.dataset_paths = np.asarray(paths, dtype=np.bytes_)
        del paths, valid_flat, self.valid_insts

        # templates: per dataset type, obj_id -> (start, end) into flat shard / key / inst arrays
        self.tem_range, self.tem_shard, self.tem_key, self.tem_inst = {}, {}, {}, {}
        for dataset_type, infos in self.templates_infos.items():
            rng, shard, key, inst, n = {}, [], [], [], 0
            for obj_id, tem_list in infos.items():
                rng[obj_id] = (n, n + len(tem_list))
                n += len(tem_list)
                for s, k, j in tem_list:
                    shard.append(s)
                    key.append(k)
                    inst.append(j)
            self.tem_range[dataset_type] = rng
            self.tem_shard[dataset_type] = np.asarray(shard, dtype=np.int64)
            self.tem_key[dataset_type] = np.asarray(key, dtype=np.bytes_)
            self.tem_inst[dataset_type] = np.asarray(inst, dtype=np.int64)
            del shard, key, inst
        del self.templates_infos
        import gc

        gc.collect()
        logger.info("compacted sample index")

    def __len__(self):
        return self.length if self.num_img_per_epoch == -1 else self.num_img_per_epoch

    def reset(self):
        if self.num_img_per_epoch == -1:
            self.num_img_per_epoch = self.length

        num_img = self.length
        if num_img <= self.num_img_per_epoch:
            self.img_idx = np.random.choice(num_img, self.num_img_per_epoch)
        else:
            self.img_idx = np.random.choice(num_img, self.num_img_per_epoch, replace=False)

    def __getitem__(self, index):
        while True:  # return valid data for train
            try:
                processed_data = self.read_data(self.img_idx[index])
                if processed_data is None:
                    raise Exception("processed data is None")
            except Exception as e:
                # e.g. reference template points to a shard we haven't downloaded
                # (only relevant when training on a partial/subset dataset)
                logger.warning(f"read data exception: {e}, rand another")
                index = self._rand_another(index)
                continue
            return processed_data

    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)

    def read_data(self, index):
        path_head = self.dataset_paths[index].decode()  # e.g.: 'MegaPose-GSO/train_pbr_web/000126/002342_000028'
        dataset_type = path_head.split("/")[0][9:]  # MegaPose-GSO/  MegaPose-ShapeNetCore --> GSO, ShapeNetCore
        if not self._check_path(osp.join(self.data_dir, path_head)):
            logger.warning("check path failed!")
            return None

        # valid_idx = load_json(osp.join(self.data_dir, path_head + ".valid_insts.json"))
        shard_name, key_name = path_head.split("/")[-2:]
        valid_idx = [int(v) for v in self.valid_flat[self.valid_offsets[index] : self.valid_offsets[index + 1]]]

        if len(valid_idx) == 0:
            logger.warning("no valid instance!")
            return None

        num_instance = len(valid_idx)
        # NOTE: randomly choose one valid instance!!!
        valid_idx = valid_idx[np.random.randint(0, num_instance)]

        # gt_info
        gt_info = io_load_gt(osp.join(self.data_dir, path_head + ".gt_info.json"))[valid_idx]
        # bbox = gt_info['bbox_visib']
        # x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]

        # gt (obj_id, R, t)
        gt = io_load_gt(osp.join(self.data_dir, path_head + ".gt.json"))[valid_idx]
        obj_id = gt["obj_id"]
        # if no reference can be found, return None
        # import ipdb; ipdb.set_trace()
        s, e = self.tem_range[dataset_type][str(obj_id)]
        assert e - s != 0

        target_R = np.array(gt["cam_R_m2c"]).reshape(3, 3).astype(np.float32)
        target_t = np.array(gt["cam_t_m2c"]).reshape(3).astype(np.float32) / 1000.0
        pose_camtgt_obj = np.eye(4, dtype=np.float32)
        pose_camtgt_obj[:3, :3] = target_R
        pose_camtgt_obj[:3, 3] = target_t

        # camera (K)
        camera = load_json(osp.join(self.data_dir, path_head + ".camera.json"))
        K = np.array(camera["cam_K"]).reshape(3, 3).astype(np.float32)

        # NOTE: template
        # randomly choose one reference for this obj during training
        tic = time.perf_counter()
        tem1_rgb, tem1_choose, tem1_pts, pose_camtem1_obj = self._get_template(dataset_type, obj_id)
        log_first_n(logging.WARNING, f"get 1 template: {time.perf_counter() - tic}s", n=1)
        if tem1_rgb is None:
            logger.warning("tem1_rgb is None!")
            return None

        # pose_tgt_tem  (tem to tgt relative pose)
        pose_tgt_tem1 = pose_camtgt_obj @ np.linalg.inv(pose_camtem1_obj)
        tem1_mean_point = np.mean(tem1_pts, axis=0)  # TODO: median vs mean?
        tem1_pts_minus_mean = tem1_pts - tem1_mean_point.reshape(1, 3)

        # mask
        mask = io_load_masks(osp.join(self.data_dir, path_head + ".mask_visib.json"))[valid_idx]
        if np.sum(mask) == 0:
            logger.warning("no valid mask")
            return None
        if self.dilate_mask and np.random.rand() < 0.5:
            tic = time.perf_counter()
            mask = np.array(mask > 0).astype(np.uint8)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
            log_first_n(logging.WARNING, f"dilate mask: {time.perf_counter() - tic}s", n=1)

        bbox = get_bbox(mask > 0)  # y1, y2, x1, x2
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        if np.sum(mask) == 0:
            logger.warning("no valid mask")
            return None

        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # depth ----------------------------------------------------------------------
        # depth --> bbox crop --> point cloud --> cam to object space (need gt R,t) --> outlier removal
        # why not use mask for crop?
        depth = load_im(osp.join(self.data_dir, path_head + ".depth.png"), format="unchanged").astype(np.float32)
        depth = depth * camera["depth_scale"] / 1000.0
        pts = backproject(depth, K, bbox)  # HW3
        pts = pts.reshape(-1, 3)[choose, :]  # camera space target points

        # R_tgt_tem @ (tem_pts - tem_mean) + t_tgt_tem = tgt_pts - R_tgt_tem @ tem_mean

        # equivalent to: R_tgt_tem^T @ (p - t_tgt_tem) --transpose--> (p^T - t^T) @ R
        # tem space target points
        # """
        # # can not use pose for data pre-processing!
        # target_pts_in_tem = (pts - pose_tgt_tem1[:3, 3].reshape(1, 3)) @ pose_tgt_tem1[:3, :3]
        # target_pts_in_tem_minus_tem_mean = target_pts_in_tem - tem1_mean_point.reshape(1, 3)
        # """
        pts_mean = np.mean(pts, axis=0)
        pts_minus_mean = pts - pts_mean.reshape(1, 3)
        # NOTE: this radius is not accurate when there is only one ref, and ref may come from real scenes
        radius = np.max(np.linalg.norm(tem1_pts_minus_mean, axis=1))
        flag = np.linalg.norm(pts_minus_mean, axis=1) < 1.2 * radius  # for outlier removal

        # tem_mean^T @ (R_tgt_tem)^T  --> R_tgt_tem @ tem_mean
        pts = pts[flag]  # - shift_in_tgt
        choose = choose[flag]

        if len(choose) < 32:
            logger.warning(f"too few points after outlier removal: {len(choose)}!")
            return None

        if len(choose) <= self.n_sample_observed_point:  # may sample same points
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        pts = pts[choose_idx]

        # rgb ----------------------------------------------------------------------
        # rgb --> crop --> color aug --> mask crop --> naive resize --> normalize
        rgb = load_im(osp.join(self.data_dir, path_head + ".rgb.jpg")).astype(np.uint8)
        if self.rgb_to_bgr:
            rgb = rgb[..., ::-1][y1:y2, x1:x2, :]  # rgb-->bgr-->crop
            log_first_n(logging.WARNING, "sam6d's setting, rgb to bgr, this is wrong!")
        else:
            rgb = rgb[y1:y2, x1:x2, :]  # rgb-->crop
            log_first_n(logging.WARNING, "use rgb target image")
        if np.random.rand() < 0.8:
            tic = time.perf_counter()
            rgb = self.color_augmentor.augment_image(rgb)
            log_first_n(logging.WARNING, f"color aug target image: {time.perf_counter() - tic}s")
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        if False:
            print("shift_in_tgt: ", shift_in_tgt)
            print("in read data")
            print(pose_tgt_tem1)
            print("pts mean:", pts.mean(0))
            print("tem1_pts_minus_mean mean: ", tem1_pts_minus_mean.mean(0))
            pts_transformed = (pts - pose_tgt_tem1[:3, 3].reshape(1, 3)) @ pose_tgt_tem1[:3, :3]
            print("pts transformed mean: ", pts_transformed.mean(0))
            plot_3d(pts, tem1_pts_minus_mean, "pts vs pts_tem1 in read data")
            plot_3d(pts_transformed, tem1_pts_minus_mean, "pts_tfm vs pts_tem1 in read data")
            # plt.figure()
            # plt.imshow(rgb.numpy().transpose(1,2,0))
            # plt.title("rgb")
            # plt.figure()
            # plt.imshow(tem1_rgb.numpy().transpose(1,2,0))
            # plt.title("tem rgb")
            # plot_3d(pts, tem1_pts, "pts vs pts_tem1 in read data")
            # plot_3d(pts_transformed, tem1_pts, "pts_tfm vs pts_tem1 in read data")
            plt.show()

        if True:
            # rotation aug --------------------------------------------
            # rand_R =
            rand_pose = np.eye(4, dtype=np.float32)
            rand_pose[:3, :3] = get_random_rotation()
            # tem1_pts = target_pts[flag][choose_idx].copy()  # NOTE: just for checking whether pose is correct
            tem1_pts = tem1_pts @ rand_pose[:3, :3]  # rand_R^T @ tem_p, rand_R is randR_tem_temRand
            target_pose = pose_tgt_tem1 @ rand_pose
            target_R = target_pose[:3, :3]
            # target_R = pose_tgt_tem1[:3, :3] @ rand_R  # R_tgt_tem1 @ randR_tem_temRand

            # translation aug -----------------------------------------
            # t: + [-0.01, 0.01]
            # pts: + [-0.01, 0.01] + [0, 0.001]
            # TODO: z shift range 0.05
            add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            target_t = target_pose[:3, 3] + add_t[0]
            add_t = add_t + 0.001 * np.random.randn(pts.shape[0], 3)
            pts = np.add(pts, add_t)  # augment observed points with random gaussian nose!!
        else:
            tem1_pts = tem1_pts_minus_mean
            target_R = pose_tgt_tem1[:3, :3]
            target_t = pose_tgt_tem1[:3, 3]

        ret_dict = {
            # observed points and rgb (after crop&resize)
            "pts": torch.FloatTensor(pts),  # in camera space
            "rgb": torch.FloatTensor(rgb),
            "rgb_choose": torch.IntTensor(rgb_choose).long(),
            # gt pose (after augmentation)
            "translation_label": torch.FloatTensor(target_t),
            "rotation_label": torch.FloatTensor(target_R),
            # tem1
            "tem1_rgb": torch.FloatTensor(tem1_rgb),
            "tem1_choose": torch.IntTensor(tem1_choose).long(),
            "tem1_pts": torch.FloatTensor(tem1_pts),
            # intrinsic
            "K": torch.FloatTensor(K),
        }
        return ret_dict

    def _get_template(self, dataset_type, obj_id):
        """Returns rgb, chooce, xyz, pose"""
        # if dataset_type == "GSO":
        #     info = self.model_info[0][obj_id]
        #     assert info["obj_id"] == obj_id
        # elif dataset_type == "ShapeNetCore":
        #     info = self.model_info[1][obj_id]
        #     assert info["obj_id"] == obj_id
        # else:
        #     raise ValueError("Unknown dataset_type: {}".format(dataset_type))
        s, e = self.tem_range[dataset_type][str(obj_id)]
        if e - s == 0:
            return None, None, None, None
        tem_indices = [i for i in range(e - s)]
        tem_idx = s + np.random.choice(tem_indices)
        tem_info = (
            int(self.tem_shard[dataset_type][tem_idx]),
            self.tem_key[dataset_type][tem_idx].decode(),
            int(self.tem_inst[dataset_type][tem_idx]),
        )
        if dataset_type == "GSO":
            dir_idx = 0
        elif dataset_type == "ShapeNetCore":
            dir_idx = 1
        else:
            raise ValueError("Unknown dataset_type: {}".format(dataset_type))
        shard_idx, key_name, inst_id = tem_info
        rgb_path = osp.join(self.data_dir, self.data_paths[dir_idx], f"{shard_idx:06d}/{key_name}.rgb.jpg")
        depth_path = osp.join(self.data_dir, self.data_paths[dir_idx], f"{shard_idx:06d}/{key_name}.depth.png")
        mask_path = osp.join(self.data_dir, self.data_paths[dir_idx], f"{shard_idx:06d}/{key_name}.mask_visib.json")
        camera_path = osp.join(self.data_dir, self.data_paths[dir_idx], f"{shard_idx:06d}/{key_name}.camera.json")
        gt_path = osp.join(self.data_dir, self.data_paths[dir_idx], f"{shard_idx:06d}/{key_name}.gt.json")

        # mask ------------------------------------------------------------
        # mask -> cropped mask
        mask = io_load_masks(mask_path, instance_ids=[inst_id])[0]
        if np.sum(mask) == 0:
            logger.warning(f"no valid mask for reference: {dataset_type} {tem_info}")
            return None, None, None, None
        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]
        if np.sum(mask) == 0:
            logger.warning(f"no valid mask for reference: {dataset_type} {tem_info}")
            return None, None, None, None

        # rgb
        # crop --> mask --> resize (naive) --> normalize
        if self.rgb_to_bgr:
            rgb = load_im(rgb_path).astype(np.uint8)[..., ::-1][y1:y2, x1:x2, :]
            log_first_n(logging.WARNING, "sam6d's setting, rgb to bgr, this is wrong!")
        else:
            rgb = load_im(rgb_path).astype(np.uint8)[y1:y2, x1:x2, :]
            log_first_n(logging.WARNING, "use rgb template image")
        if np.random.rand() < 0.8:
            tic = time.perf_counter()
            rgb = self.color_augmentor.augment_image(rgb)
            log_first_n(logging.WARNING, f"color aug template rgb: {time.perf_counter() - tic}s")
        if self.rgb_mask_flag:  # only keep fg for template
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        # naive resize??
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))

        # xyz in camera space
        # (NOTE: for unknwon ref pose case, we can not get the object space xyz)
        # xyz in [-1, 1] --> crop --> sample masked pts --> * 0.1: [-0.1, 0.1]
        choose = mask.astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point, replace=False)
        choose = choose[choose_idx]

        # camera (K)
        camera = load_json(camera_path)
        K = np.array(camera["cam_K"]).reshape(3, 3).astype(np.float32)

        # xyz = np.load(xyz_path).astype(np.float32)[y1:y2, x1:x2, :]
        depth = load_im(depth_path, format="unchanged").astype(np.float32)
        depth = depth * camera["depth_scale"] / 1000.0

        xyz = backproject(depth, K, bbox=bbox)
        xyz = xyz.reshape((-1, 3))[choose, :]  # # camera space pts

        # xyz, model_choose = farthest_point_sampling(xyz, self.n_sample_observed_point, init_center=True)
        # choose = choose[model_choose]

        choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        gt = io_load_gt(gt_path, instance_ids=[inst_id])[0]
        tem_R = np.array(gt["cam_R_m2c"]).reshape(3, 3).astype(np.float32)
        tem_t = np.array(gt["cam_t_m2c"]).reshape(3).astype(np.float32) / 1000.0
        tem_pose = np.eye(4, dtype=np.float32)
        tem_pose[:3, :3] = tem_R
        tem_pose[:3, 3] = tem_t
        return rgb, choose, xyz, tem_pose

    def _check_path(self, path_head):
        keys = [
            ".camera.json",
            ".depth.png",
            ".gt_info.json",
            ".gt.json",
            ".mask_visib.json",
            ".rgb.jpg",
        ]  # ".valid_insts.json"]

        for k in keys:
            if not osp.exists(path_head + k):
                logger.warning(f"{path_head + k} does not exist!")
                return False
        return True


def get_batch_lrf(pts):
    # pts: B*N*3
    centroids = torch.mean(pts, 1, True)  # [B, 1, 3]
    max_pts = torch.max(pts, 1, False).values  # [B, 3]
    min_pts = torch.min(pts, 1, False).values  # [B, 3]
    r_lrf = torch.norm(max_pts - min_pts, dim=1) / 2.0  # [B]

    batch_lrf = LRF(r_lrf)
    pts_lrf = batch_lrf(centroids.transpose(1, 2), pts.transpose(1, 2))
    pts_lrf = pts_lrf.transpose(1, 2).contiguous()
    return pts_lrf


if __name__ == "__main__":
    from easydict import EasyDict as edict
    from pathlib import Path
    from core.unopose.utils.vis_utils import plot_3d
    from core.unopose.utils.model_utils import LRF
    import matplotlib.pyplot as plt

    np.random.seed(1)

    PROJ_ROOT = Path(__file__).parent.parent.parent.parent
    print(PROJ_ROOT)

    cfg = edict(
        data_dir=osp.join(PROJ_ROOT, "datasets/MegaPose-Training-Data"),
        img_size=224,
        n_sample_observed_point=2048,
        n_sample_model_point=2048,
        n_sample_template_point=2500,
        min_visib_fract=0.1,
        min_px_count_visib=512,
        shift_range=0.01,
        rgb_mask_flag=True,
        dilate_mask=True,
        rgb_to_bgr=False,  # NOTE: added
    )
    dataset = DatasetPoseFreeOneRefv2(cfg)
    dataset.reset()

    # idx = 10000
    # data = dataset[idx]
    for data in dataset:
        # dict_keys(['pts', 'rgb', 'rgb_choose', 'translation_label', 'rotation_label', 'tem1_rgb', 'tem1_choose', 'tem1_pts', 'K'])
        pts = data["pts"]
        pts_tem1 = data["tem1_pts"]

        # # vis diff strategy after LRF
        # # code for v0
        # radius = torch.norm(pts_tem1, dim=1).max(0)[0]
        # normed_pts = pts / (radius + 1e-6)
        # normed_pts_tem1 = pts_tem1 / (radius + 1e-6)
        # # plot_3d(normed_pts.numpy(), normed_pts_tem1.numpy(), "v0 normed pts",)

        # v0_pts_lrf = get_batch_lrf(normed_pts[None])[0]
        # v0_pts_tem1_lrf = get_batch_lrf(normed_pts_tem1[None])[0]
        # # plot_3d(v0_pts_lrf.numpy(), v0_pts_tem1_lrf.numpy(), "v0 grfed pts",)

        # # code for v1
        tem1_mean_point = torch.mean(pts_tem1, dim=0, keepdim=True)
        tem1_pts_minus_mean = pts_tem1 - tem1_mean_point
        radius = torch.norm(tem1_pts_minus_mean, dim=1).max(0)[0]
        normed_pts = pts / (radius + 1e-6)
        normed_pts_tem1 = pts_tem1 / (radius + 1e-6)
        # plot_3d(normed_pts.numpy(), normed_pts_tem1.numpy(), "v1 normed pts")

        # v1_pts_lrf = get_batch_lrf(normed_pts[None])[0]
        # v1_pts_tem1_lrf = get_batch_lrf(normed_pts_tem1[None])[0]
        # plot_3d(v1_pts_lrf.numpy(), v1_pts_tem1_lrf.numpy(), "v1 grfed pts")

        # plot_3d(v0_pts_lrf.numpy(), v1_pts_lrf.numpy(), "grfed pts", save_path="output/grfed_pts.png")
        # plot_3d(v0_pts_tem1_lrf.numpy(), v1_pts_tem1_lrf.numpy(), "grfed tem pts", save_path="output/grfed_tem_pts.png")

        # (p^T - t_T) @ R_c_o @ randR_o_oRand, means:
        # camera space --> object space --> random rotation in object space
        trans = data["translation_label"].reshape(1, 3) / (radius + 1e-6)
        dis_thres = 0.15
        normed_pts_transformed = (normed_pts - trans) @ data["rotation_label"]
        dis_mat = torch.sqrt(pairwise_distance(normed_pts_transformed, normed_pts_tem1))
        corr = torch.stack(torch.where(dis_mat < dis_thres), dim=-1)
        idx1, idx2 = torch.unique(corr[:, 0]), torch.unique(corr[:, 1])
        plot_3d(pts, pts_tem1, "pts vs pts_tem1", save_path="output/raw_pts_tem.png")
        plot_3d(
            normed_pts_transformed, normed_pts_tem1, "normed pts_tfm vs pts_tem1", save_path="output/normed_pts_tem.png"
        )
        plot_3d(
            normed_pts_transformed[idx1],
            normed_pts_tem1[idx2],
            "overlapped normed pts_tfm vs pts_tem1",
            save_path="output/overlap03_pts_tem.png",
        )

        # vis gt overlap
