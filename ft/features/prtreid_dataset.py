import csv
import random
from collections import defaultdict
from pathlib import Path


def register_ft_manifest_dataset(manifest_path, dataset_name="FTNativeTrain", nickname="fttrain"):
    """Register a manifest-backed PRTReID ImageDataset without TrackLab."""
    from prtreid.data import ImageDataset, register_image_dataset

    manifest_path = Path(manifest_path).resolve()

    class FTManifestImageDataset(ImageDataset):
        dataset_dir = "FTNativeTrain"

        @staticmethod
        def get_masks_config(_masks_dir):
            return None

        def __init__(self, root="", **kwargs):
            rows = read_manifest(manifest_path)
            train = [torchreid_row(row) for row in rows if row["split"] == "train"]
            query = [torchreid_row(row) for row in rows if row["split"] == "query"]
            gallery = [torchreid_row(row) for row in rows if row["split"] == "gallery"]
            self.column_mapping = {
                "role": {3: "player"},
                "team": {int(row["team_id"]): row["team"] for row in rows},
            }
            super().__init__(train, query, gallery, **kwargs)

    try:
        register_image_dataset(dataset_name, FTManifestImageDataset, nickname)
    except Exception as exc:
        message = str(exc).lower()
        if "already" not in message and "registered" not in message:
            raise
    return dataset_name


def install_same_team_sampler():
    """Install an FT sampler under PRTReID's existing sampler extension point."""
    import prtreid.data.sampler as sampler_module

    sampler_module.PrtreidSampler = FTSameTeamIdentitySampler


class FTSameTeamIdentitySampler:
    """Build identity batches dominated by hard negatives from one match/team."""

    def __init__(self, data_source, batch_size, num_instances, column_mapping=None):
        del column_mapping
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.pids_per_batch = self.batch_size // self.num_instances
        self.by_pid = defaultdict(list)
        self.by_group = defaultdict(lambda: defaultdict(list))
        for index, sample in enumerate(data_source):
            pid = int(sample["pid"])
            self.by_pid[pid].append(index)
            self.by_group[(sample.get("video_id"), int(sample.get("team", -1)))][pid].append(index)
        self.groups = [group for group in self.by_group.values() if len(group) >= 2]
        self.pids = sorted(self.by_pid)
        self.length = (len(data_source) // self.batch_size) * self.batch_size
        if not self.groups:
            raise ValueError("Same-team sampler requires at least one match/team with two identities")

    def __iter__(self):
        indices = []
        for _batch in range(self.length // self.batch_size):
            group = random.choice(self.groups)
            group_pids = list(group)
            take = min(len(group_pids), self.pids_per_batch)
            selected = random.sample(group_pids, take)
            remaining = [pid for pid in self.pids if pid not in selected]
            if len(selected) < self.pids_per_batch:
                selected.extend(random.sample(remaining, self.pids_per_batch - len(selected)))
            for pid in selected:
                source = group.get(pid) or self.by_pid[pid]
                indices.extend(random.choices(source, k=self.num_instances) if len(source) < self.num_instances else random.sample(source, self.num_instances))
        return iter(indices)

    def __len__(self):
        return self.length


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def torchreid_row(row):
    return {
        "img_path": row["img_path"],
        "pid": int(row["pid"]),
        "camid": int(row["camid"]),
        "masks_path": row.get("masks_path") or "",
        "team": int(row["team_id"]),
        "role": int(row.get("role_id") or 3),
        "jersey_number": parse_optional_int(row.get("jersey_number")),
        "video_id": row["video_id"],
    }


def parse_optional_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
