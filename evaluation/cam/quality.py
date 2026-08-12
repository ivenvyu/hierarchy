#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset.imagefolder import JointImageTransform
from data.original_taxonomy import list_taxonomy_nodes, node_lookup
from model.factory import get_model
from evaluation import checkpoints as eot
from evaluation.cam.species import _species_logits

try:
    from scipy.stats import kruskal, mannwhitneyu, spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


QUALITY_COLUMNS = {
    ("Flat", "Species"): "flat_species_cam_quality",

    ("Independent", "Family"): "ind_family_cam_quality",
    ("Independent", "Genus"): "ind_genus_cam_quality",
    ("Independent", "Species"): "ind_species_cam_quality",

    ("Shared", "Family"): "shared_family_cam_quality",
    ("Shared", "Genus"): "shared_genus_cam_quality",
    ("Shared", "Species"): "shared_species_cam_quality",
}

CONFIDENCE_COLUMNS = {
    ("Flat", "Species"): "flat_species_conf",

    ("Independent", "Family"): "ind_family_conf",
    ("Independent", "Genus"): "ind_genus_joint_conf",
    ("Independent", "Species"): "ind_species_joint_conf",

    ("Shared", "Family"): "shared_family_conf",
    ("Shared", "Genus"): "shared_genus_joint_conf",
    ("Shared", "Species"): "shared_species_joint_conf",
}

QUALITY_TO_SCORE = {
    "background-heavy": 0,
    "mixed": 1,
    "object-focused": 2,
}


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(
        path.read_text(encoding="utf-8")
    ) or {}

    if not isinstance(value, dict):
        raise ValueError(
            f"YAML 최상위 값이 mapping이 아닙니다: {path}"
        )

    return value


def torch_load(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def checkpoint_state(payload):
    if isinstance(payload, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = payload.get(key)
            if isinstance(value, dict):
                return value

    return payload


def load_images(
    image_paths: list[Path],
    transform: JointImageTransform,
):
    tensors = []

    for path in image_paths:
        image = Image.open(
            path
        ).convert("RGB")

        tensor, _, _ = transform(
            image,
            bbox=None,
            bbox_coordinate_mode="normalized",
        )

        tensors.append(tensor)

    return torch.stack(
        tensors,
        dim=0,
    )


def collect_metadata(
    metadata_root: Path,
) -> pd.DataFrame:
    records = []
    seen = set()

    for path in sorted(
        metadata_root.rglob(
            "cam_comparison_metadata.json"
        )
    ):
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        image_path = Path(
            str(payload["image"])
        ).expanduser().resolve()

        species_index = int(
            payload["species_index"]
        )

        key = (
            str(image_path),
            species_index,
        )

        if key in seen:
            continue

        seen.add(key)

        comparison_image = (
            path.parent
            / "cam_comparison_all_models.jpg"
        )

        records.append(
            {
                "image_path": str(
                    image_path
                ),
                "cam_comparison_image": str(
                    comparison_image.resolve()
                ),
                "metadata_json": str(
                    path.resolve()
                ),
                "species_index": species_index,
                "species": str(
                    payload.get(
                        "species",
                        image_path.parent.name,
                    )
                ),
            }
        )

    if not records:
        raise RuntimeError(
            "cam_comparison_metadata.json을 "
            f"찾지 못했습니다: {metadata_root}"
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "species_index",
                "image_path",
            ]
        )
        .reset_index(drop=True)
    )


def build_flat_model(
    config_path: Path,
    checkpoint_path: Path,
    device,
):
    config = load_yaml(
        config_path
    )

    config.update(
        {
            "load_pretrained_backbone": False,
            "promptcam_checkpoint": None,
            "resume": None,
            "vis_attn": False,
            "debug": False,
            "distributed": False,
        }
    )

    params = SimpleNamespace(
        **config
    )

    model, _, _ = get_model(
        params
    )

    state = checkpoint_state(
        torch_load(
            checkpoint_path
        )
    )

    incompatible = model.load_state_dict(
        state,
        strict=False,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Flat checkpoint 구조 불일치: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected="
            f"{incompatible.unexpected_keys}"
        )

    model = model.to(
        device
    ).eval()

    transform = JointImageTransform(
        params,
        training=False,
    )

    return model, transform


def build_shared_model(
    run_dir: Path,
    device,
):
    args_path = (
        run_dir / "args.yaml"
    )

    checkpoint_path = (
        run_dir / "model.pt"
    )

    taxonomy_path = (
        run_dir / "taxonomy.json"
    )

    config = load_yaml(
        args_path
    )

    config.update(
        {
            "load_pretrained_backbone": False,
            "promptcam_checkpoint": None,
            "resume": None,
            "vis_attn": False,
            "debug": False,
            "distributed": False,
        }
    )

    params = SimpleNamespace(
        **config
    )

    model, _, _ = get_model(
        params
    )

    state = checkpoint_state(
        torch_load(
            checkpoint_path
        )
    )

    incompatible = model.load_state_dict(
        state,
        strict=False,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Shared checkpoint 구조 불일치: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected="
            f"{incompatible.unexpected_keys}"
        )

    model = model.to(
        device
    ).eval()

    transform = JointImageTransform(
        params,
        training=False,
    )

    taxonomy = json.loads(
        taxonomy_path.read_text(
            encoding="utf-8"
        )
    )

    return (
        model,
        transform,
        taxonomy,
    )


def export_flat(
    df,
    *,
    config_path,
    checkpoint_path,
    device,
    batch_size,
):
    print(
        "[1/3] Flat confidence 추출"
    )

    model, transform = (
        build_flat_model(
            config_path,
            checkpoint_path,
            device,
        )
    )

    image_paths = [
        Path(value)
        for value in df[
            "image_path"
        ]
    ]

    images = load_images(
        image_paths,
        transform,
    )

    targets = torch.tensor(
        df["species_index"].tolist(),
        dtype=torch.long,
    )

    probability_chunks = []
    prediction_chunks = []

    with torch.inference_mode():
        for start in range(
            0,
            images.shape[0],
            batch_size,
        ):
            batch = images[
                start:start + batch_size
            ].to(device)

            output, _ = model(
                batch
            )

            logits = _species_logits(
                output
            ).float()

            probs = F.softmax(
                logits,
                dim=1,
            ).cpu()

            probability_chunks.append(
                probs
            )

            prediction_chunks.append(
                probs.argmax(dim=1)
            )

    probabilities = torch.cat(
        probability_chunks
    )

    predictions = torch.cat(
        prediction_chunks
    )

    indices = torch.arange(
        len(df)
    )

    result = df.copy()

    result[
        "flat_species_conf"
    ] = probabilities[
        indices,
        targets,
    ].numpy()

    result[
        "flat_species_correct"
    ] = predictions.eq(
        targets
    ).numpy()

    del model

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def export_shared(
    df,
    *,
    run_dir,
    device,
    batch_size,
):
    print(
        "[2/3] Shared hierarchy confidence 추출"
    )

    (
        model,
        transform,
        taxonomy,
    ) = build_shared_model(
        run_dir,
        device,
    )

    image_paths = [
        Path(value)
        for value in df[
            "image_path"
        ]
    ]

    images = load_images(
        image_paths,
        transform,
    )

    species_to_genus = torch.tensor(
        taxonomy[
            "species_to_genus"
        ],
        dtype=torch.long,
    )

    genus_to_family = torch.tensor(
        taxonomy[
            "genus_to_family"
        ],
        dtype=torch.long,
    )

    species_targets = torch.tensor(
        df[
            "species_index"
        ].tolist(),
        dtype=torch.long,
    )

    genus_targets = (
        species_to_genus[
            species_targets
        ]
    )

    family_targets = (
        genus_to_family[
            genus_targets
        ]
    )

    keys = [
        "family_probabilities",
        "genus_probabilities",
        "species_probabilities",
        "genus_conditional_probabilities",
        "species_conditional_probabilities",
    ]

    chunks = {
        key: []
        for key in keys
    }

    with torch.inference_mode():
        for start in range(
            0,
            images.shape[0],
            batch_size,
        ):
            batch = images[
                start:start + batch_size
            ].to(device)

            output, _ = model(
                batch
            )

            if not isinstance(
                output,
                dict,
            ):
                raise TypeError(
                    "Shared 모델 출력이 "
                    "dict가 아닙니다"
                )

            for key in keys:
                chunks[key].append(
                    output[key]
                    .detach()
                    .float()
                    .cpu()
                )

    out = {
        key: torch.cat(
            values
        )
        for key, values
        in chunks.items()
    }

    indices = torch.arange(
        len(df)
    )

    result = df.copy()

    result[
        "shared_family_conf"
    ] = out[
        "family_probabilities"
    ][
        indices,
        family_targets,
    ].numpy()

    result[
        "shared_genus_cond_conf"
    ] = out[
        "genus_conditional_probabilities"
    ][
        indices,
        genus_targets,
    ].numpy()

    result[
        "shared_genus_joint_conf"
    ] = out[
        "genus_probabilities"
    ][
        indices,
        genus_targets,
    ].numpy()

    result[
        "shared_species_cond_conf"
    ] = out[
        "species_conditional_probabilities"
    ][
        indices,
        species_targets,
    ].numpy()

    result[
        "shared_species_joint_conf"
    ] = out[
        "species_probabilities"
    ][
        indices,
        species_targets,
    ].numpy()

    result[
        "shared_family_correct"
    ] = (
        out[
            "family_probabilities"
        ]
        .argmax(dim=1)
        .eq(family_targets)
        .numpy()
    )

    result[
        "shared_genus_correct"
    ] = (
        out[
            "genus_probabilities"
        ]
        .argmax(dim=1)
        .eq(genus_targets)
        .numpy()
    )

    result[
        "shared_species_correct"
    ] = (
        out[
            "species_probabilities"
        ]
        .argmax(dim=1)
        .eq(species_targets)
        .numpy()
    )

    del model

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def export_independent(
    df,
    *,
    training_summary,
    device,
    batch_size,
):
    print(
        "[3/3] Independent taxonomy confidence 추출"
    )

    checkpoint_paths = (
        eot._discover_checkpoints(
            [],
            None,
            str(
                training_summary
            ),
        )
    )

    records, _ = (
        eot._checkpoint_records(
            checkpoint_paths,
            duplicate_policy="latest",
        )
    )

    eot._validate_checkpoint_compatibility(
        records
    )

    reference_config = dict(
        records[
            "root"
        ]["config"]
    )

    _, taxonomy = (
        eot._dataset_and_taxonomy(
            reference_config,
            "test",
        )
    )

    all_nodes = (
        list_taxonomy_nodes(
            taxonomy,
            trainable_only=False,
        )
    )

    trainable_nodes = (
        list_taxonomy_nodes(
            taxonomy,
            trainable_only=True,
        )
    )

    eot._validate_checkpoint_taxonomy_mappings(
        records,
        trainable_nodes,
    )

    nodes = node_lookup(
        all_nodes
    )

    params = SimpleNamespace(
        **reference_config
    )

    transform = JointImageTransform(
        params,
        training=False,
    )

    image_paths = [
        Path(value)
        for value in df[
            "image_path"
        ]
    ]

    images = load_images(
        image_paths,
        transform,
    )

    node_probs = {}
    node_preds = {}

    for node in trainable_nodes:
        print(
            f"  - node {node.node_id}"
        )

        model = eot._build_model(
            records[
                node.node_id
            ],
            device,
        )

        chunks = []

        with torch.inference_mode():
            for start in range(
                0,
                images.shape[0],
                batch_size,
            ):
                batch = images[
                    start:start + batch_size
                ].to(device)

                output, _ = model(
                    batch
                )

                logits = (
                    eot._squeeze_logits(
                        output
                    )
                    .float()
                )

                chunks.append(
                    F.softmax(
                        logits,
                        dim=1,
                    ).cpu()
                )

        probs = torch.cat(
            chunks
        )

        node_probs[
            node.node_id
        ] = probs

        node_preds[
            node.node_id
        ] = probs.argmax(
            dim=1
        )

        del model

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    species_to_genus = [
        int(x)
        for x in taxonomy.species_to_genus
    ]

    genus_to_family = [
        int(x)
        for x in taxonomy.genus_to_family
    ]

    family_names = list(
        taxonomy.family_names
    )

    genus_names = list(
        taxonomy.genus_names
    )

    family_node_by_name = {
        node.name: node
        for node in all_nodes
        if node.rank == "family"
    }

    genus_node_by_name = {
        node.name: node
        for node in all_nodes
        if node.rank == "genus"
    }

    root = nodes[
        "root"
    ]

    family_confs = []
    genus_cond_confs = []
    genus_joint_confs = []
    species_cond_confs = []
    species_joint_confs = []

    family_correct = []
    genus_correct = []
    species_correct = []

    species_indices = (
        df[
            "species_index"
        ]
        .astype(int)
        .tolist()
    )

    for row_index, species_index in enumerate(
        species_indices
    ):
        genus_index = (
            species_to_genus[
                species_index
            ]
        )

        family_index = (
            genus_to_family[
                genus_index
            ]
        )

        root_target = (
            root.local_target(
                species_index
            )
        )

        family_conf = float(
            node_probs[
                "root"
            ][
                row_index,
                root_target,
            ].item()
        )

        family_ok = (
            int(
                node_preds[
                    "root"
                ][row_index].item()
            )
            == root_target
        )

        family_name = (
            family_names[
                family_index
            ]
        )

        family_node = (
            family_node_by_name[
                family_name
            ]
        )

        if family_node.trainable:
            genus_target = (
                family_node.local_target(
                    species_index
                )
            )

            genus_cond = float(
                node_probs[
                    family_node.node_id
                ][
                    row_index,
                    genus_target,
                ].item()
            )

            genus_local_ok = (
                int(
                    node_preds[
                        family_node.node_id
                    ][row_index].item()
                )
                == genus_target
            )

        else:
            genus_cond = 1.0
            genus_local_ok = True

        genus_joint = (
            family_conf
            * genus_cond
        )

        genus_ok = (
            family_ok
            and genus_local_ok
        )

        genus_name = (
            genus_names[
                genus_index
            ]
        )

        genus_node = (
            genus_node_by_name[
                genus_name
            ]
        )

        if genus_node.trainable:
            species_target = (
                genus_node.local_target(
                    species_index
                )
            )

            species_cond = float(
                node_probs[
                    genus_node.node_id
                ][
                    row_index,
                    species_target,
                ].item()
            )

            species_local_ok = (
                int(
                    node_preds[
                        genus_node.node_id
                    ][row_index].item()
                )
                == species_target
            )

        else:
            species_cond = 1.0
            species_local_ok = True

        species_joint = (
            genus_joint
            * species_cond
        )

        species_ok = (
            genus_ok
            and species_local_ok
        )

        family_confs.append(
            family_conf
        )

        genus_cond_confs.append(
            genus_cond
        )

        genus_joint_confs.append(
            genus_joint
        )

        species_cond_confs.append(
            species_cond
        )

        species_joint_confs.append(
            species_joint
        )

        family_correct.append(
            family_ok
        )

        genus_correct.append(
            genus_ok
        )

        species_correct.append(
            species_ok
        )

    result = df.copy()

    result[
        "ind_family_conf"
    ] = family_confs

    result[
        "ind_genus_cond_conf"
    ] = genus_cond_confs

    result[
        "ind_genus_joint_conf"
    ] = genus_joint_confs

    result[
        "ind_species_cond_conf"
    ] = species_cond_confs

    result[
        "ind_species_joint_conf"
    ] = species_joint_confs

    result[
        "ind_family_correct"
    ] = family_correct

    result[
        "ind_genus_correct"
    ] = genus_correct

    result[
        "ind_species_correct"
    ] = species_correct

    return result


def run_export(args):
    project_root = Path(
        args.project_root
    ).expanduser().resolve()

    metadata_root = resolve(
        project_root,
        args.metadata_root,
    )

    output_csv = resolve(
        project_root,
        args.output_csv,
    )

    flat_config = resolve(
        project_root,
        args.flat_config,
    )

    flat_checkpoint = resolve(
        project_root,
        args.flat_checkpoint,
    )

    shared_run = resolve(
        project_root,
        args.shared_run,
    )

    independent_summary = resolve(
        project_root,
        args.independent_summary,
    )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA를 요청했지만 "
            "사용할 수 없습니다"
        )

    df = collect_metadata(
        metadata_root
    )

    print(
        f"CAM 비교 이미지 수: {len(df)}"
    )

    required = [
        flat_config,
        flat_checkpoint,
        shared_run / "model.pt",
        shared_run / "args.yaml",
        shared_run / "taxonomy.json",
        independent_summary,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    df = export_flat(
        df,
        config_path=flat_config,
        checkpoint_path=flat_checkpoint,
        device=device,
        batch_size=args.batch_size,
    )

    df = export_shared(
        df,
        run_dir=shared_run,
        device=device,
        batch_size=args.batch_size,
    )

    df = export_independent(
        df,
        training_summary=independent_summary,
        device=device,
        batch_size=args.batch_size,
    )

    for column in (
        QUALITY_COLUMNS.values()
    ):
        df[column] = ""

    df["notes"] = ""

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"[저장] {output_csv}"
    )

    print()
    print(
        "CAM quality 입력값:"
    )
    print(
        "  2 = object-focused"
    )
    print(
        "  1 = mixed"
    )
    print(
        "  0 = background-heavy"
    )

    print()
    print(
        "세 모델 정답 여부:"
    )

    correct_columns = [
        "flat_species_correct",
        "ind_family_correct",
        "ind_genus_correct",
        "ind_species_correct",
        "shared_family_correct",
        "shared_genus_correct",
        "shared_species_correct",
    ]

    print(
        df[
            correct_columns
        ]
        .all()
        .to_string()
    )


def normalize_quality(value):
    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .lower()
        .replace("_", "-")
    )

    aliases = {
        "0": "background-heavy",
        "0.0": "background-heavy",
        "bg": "background-heavy",
        "background": "background-heavy",
        "background-heavy": "background-heavy",

        "1": "mixed",
        "1.0": "mixed",
        "mixed": "mixed",

        "2": "object-focused",
        "2.0": "object-focused",
        "object": "object-focused",
        "focused": "object-focused",
        "object-focused": "object-focused",
    }

    return aliases.get(
        text
    )


def to_long_form(df):
    records = []

    for (
        model,
        rank,
    ), quality_column in (
        QUALITY_COLUMNS.items()
    ):
        confidence_column = (
            CONFIDENCE_COLUMNS[
                (model, rank)
            ]
        )

        for _, row in df.iterrows():
            quality = (
                normalize_quality(
                    row.get(
                        quality_column
                    )
                )
            )

            confidence = (
                pd.to_numeric(
                    row.get(
                        confidence_column
                    ),
                    errors="coerce",
                )
            )

            if (
                quality is None
                or pd.isna(confidence)
            ):
                continue

            records.append(
                {
                    "image_path": row[
                        "image_path"
                    ],
                    "species": row[
                        "species"
                    ],
                    "model": model,
                    "rank": rank,
                    "cam_quality": quality,
                    "cam_quality_score": (
                        QUALITY_TO_SCORE[
                            quality
                        ]
                    ),
                    "confidence": float(
                        confidence
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def run_analyze(args):
    input_csv = Path(
        args.input_csv
    ).expanduser().resolve()

    out_dir = Path(
        args.out_dir
    ).expanduser().resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        input_csv
    )

    long_df = to_long_form(
        df
    )

    if long_df.empty:
        raise RuntimeError(
            "CAM quality annotation이 없습니다. "
            "*_cam_quality 컬럼을 먼저 채우세요."
        )

    long_df.to_csv(
        out_dir / "long_form.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []

    for (
        model,
        rank,
        quality,
    ), group in long_df.groupby(
        [
            "model",
            "rank",
            "cam_quality",
        ]
    ):
        values = group[
            "confidence"
        ].astype(float)

        summary_rows.append(
            {
                "model": model,
                "rank": rank,
                "cam_quality": quality,
                "n": len(values),
                "mean_confidence": values.mean(),
                "median_confidence": values.median(),
                "std_confidence": (
                    values.std(ddof=1)
                    if len(values) >= 2
                    else math.nan
                ),
                "q1": values.quantile(
                    0.25
                ),
                "q3": values.quantile(
                    0.75
                ),
            }
        )

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "model",
                "rank",
                "cam_quality",
            ]
        )
    )

    summary.to_csv(
        out_dir
        / "group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_rows = []

    for (
        model,
        rank,
    ), group in long_df.groupby(
        [
            "model",
            "rank",
        ]
    ):
        rho = math.nan
        p_spearman = math.nan
        h = math.nan
        p_kruskal = math.nan
        u = math.nan
        p_mw = math.nan

        if (
            len(group) >= 3
            and group[
                "cam_quality_score"
            ].nunique() >= 2
        ):
            if SCIPY_AVAILABLE:
                (
                    rho,
                    p_spearman,
                ) = spearmanr(
                    group[
                        "cam_quality_score"
                    ],
                    group[
                        "confidence"
                    ],
                )

                quality_groups = [
                    group.loc[
                        group[
                            "cam_quality"
                        ] == label,
                        "confidence",
                    ].values
                    for label
                    in QUALITY_TO_SCORE
                    if (
                        group[
                            "cam_quality"
                        ] == label
                    ).any()
                ]

                if len(
                    quality_groups
                ) >= 2:
                    (
                        h,
                        p_kruskal,
                    ) = kruskal(
                        *quality_groups
                    )

                obj = group.loc[
                    group[
                        "cam_quality"
                    ] == "object-focused",
                    "confidence",
                ].values

                bg = group.loc[
                    group[
                        "cam_quality"
                    ] == "background-heavy",
                    "confidence",
                ].values

                if (
                    len(obj)
                    and len(bg)
                ):
                    (
                        u,
                        p_mw,
                    ) = mannwhitneyu(
                        obj,
                        bg,
                        alternative="two-sided",
                    )

                status = "ok"

            else:
                rho = (
                    group[
                        "cam_quality_score"
                    ]
                    .rank()
                    .corr(
                        group[
                            "confidence"
                        ].rank()
                    )
                )

                status = (
                    "scipy_not_available"
                )

        else:
            status = (
                "too_few_annotations"
            )

        obj = group.loc[
            group[
                "cam_quality"
            ] == "object-focused",
            "confidence",
        ]

        bg = group.loc[
            group[
                "cam_quality"
            ] == "background-heavy",
            "confidence",
        ]

        test_rows.append(
            {
                "model": model,
                "rank": rank,
                "n": len(group),
                "spearman_rho": rho,
                "spearman_p": p_spearman,
                "kruskal_H": h,
                "kruskal_p": p_kruskal,
                "object_n": len(obj),
                "background_n": len(bg),
                "object_mean_confidence": (
                    obj.mean()
                    if len(obj)
                    else math.nan
                ),
                "background_mean_confidence": (
                    bg.mean()
                    if len(bg)
                    else math.nan
                ),
                "object_minus_background": (
                    obj.mean()
                    - bg.mean()
                    if (
                        len(obj)
                        and len(bg)
                    )
                    else math.nan
                ),
                "mannwhitney_U": u,
                "mannwhitney_p": p_mw,
                "status": status,
            }
        )

    tests = (
        pd.DataFrame(
            test_rows
        )
        .sort_values(
            [
                "model",
                "rank",
            ]
        )
    )

    tests.to_csv(
        out_dir
        / "statistical_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# CAM quality vs confidence",
        "",
        f"Annotated CAM rows: {len(long_df)}",
        "",
        "## Group summary",
        "",
        summary.to_markdown(
            index=False
        ),
        "",
        "## Statistical tests",
        "",
        tests.to_markdown(
            index=False
        ),
        "",
        "## Interpretation",
        "",
        "- Spearman rho > 0: object-focused할수록 confidence가 높은 경향.",
        "- object_minus_background > 0: object-focused 그룹의 평균 confidence가 더 높음.",
        "- 표본 수가 작으면 p-value보다 효과 방향과 크기를 우선 해석.",
    ]

    (
        out_dir
        / "report.md"
    ).write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        tests.to_string(
            index=False
        )
    )

    print()
    print(
        f"[저장] {out_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "CAM quality와 predictive "
            "confidence 관계 분석"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    export = sub.add_parser(
        "export"
    )

    export.add_argument(
        "--project-root",
        default=str(
            PROJECT_ROOT
        ),
    )

    export.add_argument(
        "--metadata-root",
        default=(
            "output/cam/"
            "qualitative_comparison"
        ),
    )

    export.add_argument(
        "--output-csv",
        default="output/evaluation/cam_quality.csv",
    )

    export.add_argument(
        "--flat-config",
        default="configs/flat.yaml",
    )

    export.add_argument(
        "--flat-checkpoint",
        required=True,
    )

    export.add_argument(
        "--shared-run",
        required=True,
    )

    export.add_argument(
        "--independent-summary",
        required=True,
    )

    export.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    export.add_argument(
        "--device",
        default="cuda",
    )

    analyze = sub.add_parser(
        "analyze"
    )

    analyze.add_argument(
        "--input-csv",
        default="output/evaluation/cam_quality.csv",
    )

    analyze.add_argument(
        "--out-dir",
        default="output/evaluation/cam_quality_results",
    )

    args = parser.parse_args()

    if args.command == "export":
        run_export(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
