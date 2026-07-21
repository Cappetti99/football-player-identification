from pathlib import Path
import os
import tempfile
import uuid

import numpy as np


ROLE_MAPPING = {"ball": 0, "goalkeeper": 1, "other": 2, "player": 3, "referee": 4, None: -1}
INVERSE_ROLE_MAPPING = {value: key for key, value in ROLE_MAPPING.items()}


class PRTReIDUnavailableError(RuntimeError):
    pass


class PRTReIDFeatureExtractor:
    """FT-native adapter for SoccerNet PRTReID feature extraction.

    The SoccerNet wrapper is a TrackLab DetectionLevelModule. This adapter keeps
    FT independent from TrackLab and only uses the underlying PRTReID extractor.
    """

    def __init__(
        self,
        enabled=False,
        weights_path=None,
        hrnet_pretrained_path=None,
        device="auto",
        batch_size=32,
        image_width=128,
        image_height=256,
        test_embeddings=None,
        download_weights=False,
        role_enabled=True,
        extractor_factory=None,
        extract_embeddings_fn=None,
    ):
        self.enabled = bool(enabled)
        self.weights_path = str(weights_path or "models/reid/prtreid-soccernet-baseline.pth.tar")
        self.hrnet_pretrained_path = str(hrnet_pretrained_path or "models/reid")
        self.device = normalize_device(device)
        self.batch_size = int(batch_size or 32)
        self.image_width = int(image_width or 128)
        self.image_height = int(image_height or 256)
        self.test_embeddings = list(test_embeddings or ["globl"])
        self.download_weights = bool(download_weights)
        self.role_enabled = bool(role_enabled)
        self._save_dir = str(Path(tempfile.gettempdir()) / f"ft_prtreid_{os.getpid()}_{uuid.uuid4().hex}")
        self._extractor_factory = extractor_factory
        self._extract_embeddings_fn = extract_embeddings_fn
        self._feature_extractor = None
        self.stats = {
            "backend": "prtreid",
            "enabled": self.enabled,
            "weights_path": self.weights_path,
            "hrnet_pretrained_path": self.hrnet_pretrained_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "image_size": [self.image_height, self.image_width],
            "test_embeddings": self.test_embeddings,
            "computed": 0,
            "reused": 0,
            "failed": 0,
            "embedding_dim": None,
            "role_enabled": self.role_enabled,
            "status": "not_initialized",
        }

    def add_row_features(self, rows):
        if not self.enabled:
            raise PRTReIDUnavailableError("PRTReID requested but prtreid.enabled=false")
        pending = []
        for row in rows:
            if row.get("visual_embedding") is not None and row.get("reid_model") == "prtreid":
                self.stats["reused"] += 1
            else:
                pending.append(row)
        if not pending:
            self.stats["status"] = "ok"
            return rows
        self._ensure_ready()
        for start in range(0, len(pending), self.batch_size):
            batch_rows = pending[start : start + self.batch_size]
            crops = []
            valid_indices = []
            for index, row in enumerate(batch_rows):
                crop = read_crop(row.get("crop_path"))
                if crop is None:
                    self._mark_failed(row, "missing_or_unreadable_crop")
                    continue
                crops.append(crop)
                valid_indices.append(index)
            if not crops:
                continue
            batch_result = self._extract(crops)
            for offset, feature in zip(valid_indices, batch_result):
                apply_feature(batch_rows[offset], feature)
                self.stats["computed"] += 1
                embedding = feature.get("visual_embedding")
                if embedding is not None:
                    self.stats["embedding_dim"] = len(embedding)
        return rows

    def extract_crops(self, crops):
        """Extract normalized PRTReID features from in-memory BGR crops."""
        if not self.enabled:
            raise PRTReIDUnavailableError("PRTReID requested but prtreid.enabled=false")
        self._ensure_ready()
        output = []
        for start in range(0, len(crops), self.batch_size):
            batch = [crop for crop in crops[start : start + self.batch_size] if crop is not None and crop.size]
            if not batch:
                continue
            features = self._extract(batch)
            output.extend(features)
            self.stats["computed"] += len(features)
            for feature in features:
                embedding = feature.get("visual_embedding")
                if embedding is not None:
                    self.stats["embedding_dim"] = len(embedding)
        return output

    def diagnostics(self):
        return dict(self.stats)

    def _ensure_ready(self):
        if self._feature_extractor is not None:
            return
        if self.download_weights:
            download_default_weights(self.weights_path, self.hrnet_pretrained_path)
        if self._extractor_factory is not None:
            self._feature_extractor = self._extractor_factory(self)
            self.stats["status"] = "ok"
            return
        try:
            from omegaconf import OmegaConf
            from yacs.config import CfgNode as CN
            from prtreid.data import register_image_dataset
            from prtreid.scripts.main import build_config
            from prtreid.tools.feature_extractor import FeatureExtractor
            from prtreid.utils.tools import extract_test_embeddings
        except Exception as exc:
            raise PRTReIDUnavailableError(
                "PRTReID dependencies are missing. Install prtreid, torchreid, torch, omegaconf and yacs; "
                "then run scripts/check_prtreid_env.py."
            ) from exc

        self._extract_embeddings_fn = extract_test_embeddings
        register_ftnative_dataset(register_image_dataset)
        cfg = build_config(config=CN(OmegaConf.to_container(OmegaConf.create(self._config_dict()), resolve=True)))
        self._feature_extractor = FeatureExtractor(
            cfg,
            model_path=self.weights_path,
            device=self.device,
            image_size=(self.image_height, self.image_width),
            model=None,
            verbose=False,
        )
        self.stats["status"] = "ok"

    def _config_dict(self):
        return {
            "use_gpu": self.device.startswith("cuda"),
            "project": {
                "name": "FT",
                "job_id": int(os.getpid()),
                "logger": {"use_tensorboard": False, "use_wandb": False},
            },
            "data": {
                "root": "models/reid",
                "type": "image",
                "sources": ["FTNative"],
                "targets": ["FTNative"],
                "height": self.image_height,
                "width": self.image_width,
                "combineall": False,
                "transforms": ["rc", "re"],
                "save_dir": self._save_dir,
                "workers": 0,
            },
            "sampler": {
                "train_sampler": "PrtreidSampler",
                "train_sampler_t": "PrtreidSampler",
                "num_instances": 4,
            },
            "model": {
                "name": "bpbreid",
                "pretrained": True,
                "save_model_flag": False,
                "load_config": True,
                "load_weights": self.weights_path,
                "bpbreid": {
                    "pooling": "gwap",
                    "normalization": "identity",
                    "mask_filtering_training": False,
                    "mask_filtering_testing": False,
                    "training_binary_visibility_score": True,
                    "testing_binary_visibility_score": True,
                    "last_stride": 1,
                    "learnable_attention_enabled": False,
                    "dim_reduce": "after_pooling",
                    "dim_reduce_output": 256,
                    "backbone": "hrnet32",
                    "test_embeddings": self.test_embeddings,
                    "test_use_target_segmentation": "none",
                    "shared_parts_id_classifier": False,
                    "hrnet_pretrained_path": self.hrnet_pretrained_path,
                    "masks": {"type": "disk", "dir": "", "preprocess": "id"},
                },
            },
            "loss": {
                "name": "part_based",
                "part_based": {
                    "name": "part_averaged_triplet_loss",
                    "ppl": "cl",
                    "weights": {
                        "globl": {"id": 1.0, "tr": 1.0},
                        "foreg": {"id": 0.0, "tr": 0.0},
                        "conct": {"id": 0.0, "tr": 0.0},
                        "parts": {"id": 0.0, "tr": 0.0},
                        "pixls": {"ce": 0.0},
                    },
                },
            },
            "train": {"batch_size": self.batch_size, "max_epoch": 20},
            "test": {
                "evaluate": True,
                "detailed_ranking": False,
                "start_eval": 40,
                "batch_size": self.batch_size,
                "batch_size_pairwise_dist_matrix": 5000,
                "normalize_feature": True,
                "dist_metric": "euclidean",
                "visrank": False,
                "visrank_per_body_part": False,
                "vis_embedding_projection": False,
                "vis_feature_maps": False,
                "visrank_topk": 10,
                "visrank_count": 4,
                "visrank_q_idx_list": [],
                "part_based": {"dist_combine_strat": "mean"},
            },
        }

    def _extract(self, crops):
        result = self._feature_extractor(crops, external_parts_masks=None)
        if isinstance(result, list):
            return [normalize_feature_dict(item) for item in result]
        embeddings, visibility_scores, body_masks, _pixels, role_cls_scores = self._extract_embeddings_fn(
            result, self.test_embeddings
        )
        embeddings = to_numpy(embeddings)
        visibility_scores = to_numpy(visibility_scores)
        body_masks = to_numpy(body_masks)
        role_scores = extract_role_scores(role_cls_scores)
        output = []
        for index in range(len(embeddings)):
            output.append(
                {
                    "visual_embedding": normalize_vector(embeddings[index]),
                    "reid_visibility_scores": to_list_or_none(visibility_scores, index),
                    "reid_body_masks": to_list_or_none(body_masks, index),
                    "reid_role_detection": role_from_scores(role_scores, index) if self.role_enabled else None,
                    "reid_role_confidence": role_confidence(role_scores, index) if self.role_enabled else None,
                    "reid_model": "prtreid",
                }
            )
        return output

    def _mark_failed(self, row, reason):
        row["visual_embedding"] = None
        row["reid_model"] = "prtreid"
        row["reid_error"] = reason
        self.stats["failed"] += 1


def read_crop(path):
    if not path:
        return None
    import cv2

    crop = cv2.imread(str(path))
    if crop is None or crop.size == 0:
        return None
    return crop


def normalize_device(device):
    value = str(device or "auto").lower()
    if value == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return value


def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_vector(value):
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return None
    return (vector / norm).astype(float).tolist()


def to_list_or_none(values, index):
    if values is None:
        return None
    if index >= len(values):
        return None
    return np.asarray(values[index]).astype(float).tolist()


def extract_role_scores(role_cls_scores):
    if role_cls_scores is None:
        return None
    if isinstance(role_cls_scores, dict):
        role_cls_scores = role_cls_scores.get("globl")
    if role_cls_scores is None:
        return None
    return to_numpy(role_cls_scores)


def role_from_scores(scores, index):
    if scores is None or index >= len(scores):
        return None
    role_index = int(np.argmax(scores[index]))
    return INVERSE_ROLE_MAPPING.get(role_index, "other")


def role_confidence(scores, index):
    if scores is None or index >= len(scores):
        return None
    # PRTReID exposes role-classification logits. Convert them to a calibrated
    # [0, 1] diagnostic value instead of reporting the raw maximum logit as a
    # confidence (which can legitimately be greater than one).
    logits = np.asarray(scores[index], dtype=np.float64).reshape(-1)
    if not len(logits) or not np.all(np.isfinite(logits)):
        return None
    probabilities = np.exp(logits - np.max(logits))
    denominator = float(np.sum(probabilities))
    if denominator <= 0.0:
        return None
    return float(np.max(probabilities) / denominator)


def apply_feature(row, feature):
    feature = normalize_feature_dict(feature)
    row.update(feature)
    row.setdefault("reid_model", "prtreid")


def normalize_feature_dict(feature):
    embedding = value_for(feature, "visual_embedding", "embedding")
    return {
        "visual_embedding": normalize_vector(embedding),
        "reid_visibility_scores": value_for(feature, "reid_visibility_scores", "visibility_scores"),
        "reid_body_masks": value_for(feature, "reid_body_masks", "body_masks"),
        "reid_role_detection": value_for(feature, "reid_role_detection", "role_detection"),
        "reid_role_confidence": value_for(feature, "reid_role_confidence", "role_confidence"),
        "reid_model": value_for(feature, "reid_model") or "prtreid",
    }


def value_for(mapping, *keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def check_prtreid_files(weights_path, hrnet_pretrained_path):
    weights = Path(weights_path)
    hrnet = Path(hrnet_pretrained_path)
    expected_hrnet = hrnet / "hrnetv2_w32_imagenet_pretrained.pth"
    return {
        "weights_path": str(weights),
        "weights_exists": weights.exists(),
        "hrnet_pretrained_path": str(expected_hrnet),
        "hrnet_exists": expected_hrnet.exists(),
    }


def download_default_weights(weights_path, hrnet_pretrained_path):
    weights = Path(weights_path)
    hrnet = Path(hrnet_pretrained_path) / "hrnetv2_w32_imagenet_pretrained.pth"
    if not weights.exists():
        download_file(
            "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1",
            weights,
        )
    if not hrnet.exists():
        download_file(
            "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1",
            hrnet,
        )


def download_file(url, path):
    from urllib.request import urlretrieve

    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)


class FTNativeReIDDataset:
    dataset_dir = "FTNative"

    @staticmethod
    def get_masks_config(_masks_dir):
        return None


def register_ftnative_dataset(register_image_dataset):
    try:
        register_image_dataset("FTNative", FTNativeReIDDataset, "ftnative")
    except Exception as exc:
        message = str(exc).lower()
        if "already" not in message and "registered" not in message:
            raise
