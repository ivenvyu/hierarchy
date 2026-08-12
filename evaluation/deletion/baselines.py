#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flat / Independent Prompt-CAM decision-boundary deletion experiment.

Shared 모델에서 이미 수행한 decoder-level patch-token deletion과 같은 protocol을
Flat patch-only Prompt-CAM과 Independent taxonomy-node Prompt-CAM에 적용한다.

공통 primary outcomes
-----------------------
- top_species_flip vs size-matched random deletion
- target-vs-best-competitor log-probability margin drop
- minimal CAM deletion fraction to species flip

Independent 추가 outcomes
---------------------------
- validation에서 고정한 reliability operating point의 species/genus thresholds 사용
- species -> genus/family fallback
- emitted-rank correctness
- minimal CAM deletion fraction to fallback

Intervention
------------
원본 pixel은 건드리지 않는다. Backbone을 한 번 통과한 뒤 decoder에 들어가는
patch token을 mean token으로 대체한다.

Independent에서는 target species CAM이 지목한 동일 spatial patch indices를
모든 taxonomy-node decoder의 patch token에서 제거한다. Shared에서 하나의 공통
patch representation을 제거해 family/genus/species decoder가 모두 영향을 받는 것과
가장 직접적으로 대응하는 intervention이다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def core_model(model):
    return model.module if hasattr(model, "module") else model


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "species_logits", "output", "predictions"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError(
                f"모델 출력 dict에서 logits를 찾지 못했습니다: {sorted(output)}"
            )
    if not torch.is_tensor(output):
        raise TypeError(f"모델 출력이 tensor가 아닙니다: {type(output)!r}")
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    if output.ndim != 2:
        raise ValueError(f"logits는 [B,C]여야 합니다: {tuple(output.shape)}")
    return output


def image_rng(seed: int, image_path: Path) -> np.random.Generator:
    """같은 image/seed면 모델과 무관하게 동일 random patch permutation을 사용한다."""
    payload = f"{int(seed)}|{image_path.resolve()}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    derived = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived)


def prompt_count(core) -> int:
    if hasattr(core, "vpt") and hasattr(core.vpt, "prompt_count"):
        return int(core.vpt.prompt_count)
    return int(getattr(core.params, "vpt_num"))


def patch_slice(core, features: torch.Tensor) -> tuple[int, int]:
    start = prompt_count(core) + int(getattr(core, "num_prefix_tokens", 1))
    count = int(features.shape[1] - start)
    if count <= 0:
        raise RuntimeError(
            f"patch token을 찾지 못했습니다: shape={tuple(features.shape)}, start={start}"
        )
    return start, count


def replacement_token(
    features: torch.Tensor,
    *,
    patch_start: int,
    patch_count: int,
    mode: str,
) -> torch.Tensor:
    patches = features[:, patch_start : patch_start + patch_count]
    if mode == "mean":
        return patches.mean(dim=1, keepdim=True)
    if mode == "zero":
        return torch.zeros_like(patches[:, :1])
    raise ValueError(mode)


def find_head_weights(core, head_count: int) -> tuple[torch.Tensor, str]:
    """patch-only Prompt-CAM의 learned head mixing weight를 runtime introspection으로 찾는다."""
    module = getattr(core, "prompt_patch_head", None)
    if module is None:
        return torch.full((head_count,), 1.0 / head_count), "uniform:no_prompt_patch_head"

    for attr in (
        "head_logits",
        "attention_head_logits",
        "head_weight_logits",
        "head_weights",
        "head_weight",
    ):
        value = getattr(module, attr, None)
        if torch.is_tensor(value) and value.ndim == 1 and value.numel() == head_count:
            v = value.detach().float().cpu()
            if "logit" in attr:
                return v.softmax(dim=0), f"prompt_patch_head.{attr}:softmax"
            if bool((v >= 0).all()) and float(v.sum()) > 0:
                return v / v.sum(), f"prompt_patch_head.{attr}:normalized"
            return v.softmax(dim=0), f"prompt_patch_head.{attr}:softmax_fallback"

    candidates = []
    for name, value in module.named_parameters(recurse=True):
        if (
            value.ndim == 1
            and value.numel() == head_count
            and "head" in name.lower()
        ):
            candidates.append((name, value.detach().float().cpu()))

    if len(candidates) == 1:
        name, v = candidates[0]
        if "logit" in name.lower():
            return v.softmax(dim=0), f"prompt_patch_head.{name}:softmax"
        if bool((v >= 0).all()) and float(v.sum()) > 0:
            return v / v.sum(), f"prompt_patch_head.{name}:normalized"
        return v.softmax(dim=0), f"prompt_patch_head.{name}:softmax_fallback"

    return torch.full((head_count,), 1.0 / head_count), "uniform:no_unique_head_weight"


def aggregate_promptcam_maps(
    head_maps: torch.Tensor,
    core,
) -> tuple[np.ndarray, str]:
    """[H,P] raw Prompt-CAM head maps -> learned-weight spatial CAM [P]."""
    if head_maps.ndim != 2:
        raise ValueError(f"head_maps는 [H,P]여야 합니다: {tuple(head_maps.shape)}")
    weights, source = find_head_weights(core, int(head_maps.shape[0]))
    cam = (
        head_maps.detach().float().cpu()
        * weights[:, None]
    ).sum(dim=0)
    return cam.numpy().astype(np.float64), source


def capture_features_and_attention(
    model,
    batch: torch.Tensor,
    *,
    target_local_index: int,
    params,
):
    """full forward 1회에서 logits, final features, Prompt-CAM head maps를 얻는다."""
    from evaluation.cam.species import _species_head_maps

    core = core_model(model)
    core.params.vis_attn = True
    try:
        params.vis_attn = True
    except Exception:
        pass

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        if torch.is_tensor(output):
            captured["features"] = output

    handle = core.norm.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output, attention = model(batch, patch_prior=None)
    finally:
        handle.remove()

    if "features" not in captured:
        raise RuntimeError("core.norm hook에서 final token features를 capture하지 못했습니다")

    features = captured["features"].detach()
    logits = extract_logits(output).detach().float()

    maps = _species_head_maps(
        output,
        attention,
        int(target_local_index),
        params,
        sample_index=0,
    )
    if maps.ndim != 3 or maps.shape[0] != 1:
        raise RuntimeError(f"_species_head_maps 결과가 이상합니다: {tuple(maps.shape)}")
    head_maps = maps[0].detach().float().cpu()

    # cached features -> forward_head가 full forward와 동일한지 검증.
    with torch.inference_mode():
        cached_output = core.forward_head(
            features,
            patch_prior=None,
        )
    cached_logits = extract_logits(cached_output).detach().float()
    diff = float((cached_logits - logits).abs().max().item())
    if diff > 1e-5:
        raise RuntimeError(
            "cached features -> forward_head가 원래 prediction과 일치하지 않습니다: "
            f"max_abs_diff={diff:.8g}"
        )

    return logits, features, head_maps, diff


def deleted_logits_from_features(
    core,
    features: torch.Tensor,
    index_sets: list[np.ndarray],
    *,
    replacement: str,
    batch_size: int,
) -> torch.Tensor:
    start, count = patch_slice(core, features)
    repl = replacement_token(
        features,
        patch_start=start,
        patch_count=count,
        mode=replacement,
    )[0, 0]

    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for offset in range(0, len(index_sets), int(batch_size)):
            chunk = index_sets[offset : offset + int(batch_size)]
            b = len(chunk)
            f = features.expand(b, -1, -1).clone()
            for row, idx_np in enumerate(chunk):
                idx = torch.as_tensor(
                    idx_np,
                    dtype=torch.long,
                    device=f.device,
                )
                f[row, start + idx] = repl

            out = core.forward_head(
                f,
                patch_prior=None,
            )
            outputs.append(extract_logits(out).detach().float())

    return torch.cat(outputs, dim=0)


def log_probability_margin(
    probabilities: torch.Tensor,
    target: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = probabilities.clamp_min(1e-12).log()
    target_logp = logp[:, int(target)]
    competitor = logp.clone()
    competitor[:, int(target)] = -torch.inf
    best_logp, best_index = competitor.max(dim=1)
    return target_logp - best_logp, best_index, target_logp


def load_fallback_thresholds(
    csv_path: Path,
    *,
    model_name: str,
    reliability_pct: float,
) -> tuple[float, float, pd.Series]:
    df = pd.read_csv(csv_path)
    required = {
        "model",
        "confidence_mode",
        "reliability_constraint_pct",
        "species_threshold",
        "genus_threshold",
        "status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"fallback CSV에 필요한 열이 없습니다: {sorted(missing)}"
        )

    mask = (
        df["model"].astype(str).str.lower().eq(model_name.lower())
        & df["confidence_mode"].astype(str).str.lower().eq("joint")
        & np.isclose(
            pd.to_numeric(df["reliability_constraint_pct"], errors="coerce"),
            float(reliability_pct),
            atol=1e-9,
            rtol=0.0,
        )
        & df["status"].astype(str).str.lower().eq("ok")
    )
    rows = df.loc[mask]
    if len(rows) != 1:
        raise RuntimeError(
            f"{model_name} reliability={reliability_pct} threshold 행이 "
            f"정확히 하나가 아닙니다: matches={len(rows)}"
        )
    row = rows.iloc[0]
    return float(row["species_threshold"]), float(row["genus_threshold"]), row


# ---------------------------------------------------------------------------
# Flat
# ---------------------------------------------------------------------------

class FlatRuntime:
    def __init__(self, project_root: Path, run_dir: Path, device: torch.device):
        from torchvision import datasets
        from data.dataset.imagefolder import JointImageTransform, load_taxonomy_manifest
        from model.factory import get_model
        from evaluation import hierarchy as hierarchy_eval

        self.project_root = project_root
        self.run_dir = run_dir
        self.device = device

        args_data = hierarchy_eval._load_yaml(run_dir / "args.yaml")
        if bool(args_data.get("hierarchical_prompt", False)):
            raise ValueError("Flat run에 hierarchical_prompt=True입니다")
        if bool(args_data.get("original_taxonomy_prompt", False)):
            raise ValueError("Flat run에 original_taxonomy_prompt=True입니다")

        cli = SimpleNamespace(batch_size=1, num_workers=0, device=str(device))
        params = hierarchy_eval._prepare_params(
            args_data, project_root, run_dir, cli
        )
        params.vis_attn = True
        self.params = params

        model, _, _ = get_model(params)
        checkpoint = hierarchy_eval._torch_load(run_dir / "model.pt")
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Flat checkpoint 구조 불일치: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model = model.to(device).eval()
        core_model(self.model).params.vis_attn = True
        self.transform = JointImageTransform(params, training=False)

        data_path = Path(str(params.data_path)).expanduser()
        if not data_path.is_absolute():
            data_path = project_root / data_path
        train_root = data_path / str(getattr(params, "train_split", "train"))
        if not train_root.is_dir() and (data_path / "imagefolder" / str(getattr(params, "train_split", "train"))).is_dir():
            train_root = data_path / "imagefolder" / str(getattr(params, "train_split", "train"))

        imagefolder = datasets.ImageFolder(str(train_root))
        taxonomy = load_taxonomy_manifest(
            params.taxonomy_manifest,
            imagefolder.classes,
            class_column=getattr(
                params,
                "taxonomy_class_column",
                "folder_name",
            ),
        )
        self.class_names = tuple(taxonomy.class_names)
        self.species_to_genus = tuple(int(x) for x in taxonomy.species_to_genus)
        counts = np.bincount(
            np.asarray(self.species_to_genus, dtype=np.int64),
            minlength=max(self.species_to_genus) + 1,
        )
        self.species_has_siblings = tuple(
            bool(counts[g] > 1) for g in self.species_to_genus
        )

    def image_tensor(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        tensor, _, _ = self.transform(
            image, bbox=None, bbox_coordinate_mode="normalized"
        )
        return tensor.unsqueeze(0).to(self.device)

    def baseline(self, batch: torch.Tensor, target: int):
        logits, features, head_maps, diff = capture_features_and_attention(
            self.model,
            batch,
            target_local_index=int(target),
            params=self.params,
        )
        probs = logits.softmax(dim=1)
        cam, source = aggregate_promptcam_maps(
            head_maps,
            core_model(self.model),
        )
        return probs, features, cam, source, diff

    def deleted_probabilities(
        self,
        features: torch.Tensor,
        index_sets: list[np.ndarray],
        *,
        replacement: str,
        batch_size: int,
    ) -> torch.Tensor:
        logits = deleted_logits_from_features(
            core_model(self.model),
            features,
            index_sets,
            replacement=replacement,
            batch_size=batch_size,
        )
        return logits.softmax(dim=1)


# ---------------------------------------------------------------------------
# Independent
# ---------------------------------------------------------------------------

def load_root_independent_evaluator(project_root: Path):
    del project_root
    from evaluation import independent

    return independent


class IndependentRuntime:
    def __init__(
        self,
        project_root: Path,
        run_root: Path,
        device: torch.device,
        *,
        fallback_csv: Path,
        reliability: float,
    ):
        from data.dataset.imagefolder import JointImageTransform

        self.project_root = project_root
        self.run_root = run_root
        self.device = device
        self.ind_eval = load_root_independent_evaluator(project_root)

        root_config = run_root / "configs" / "root.yaml"
        root_cfg = self.ind_eval._load_yaml(root_config)
        taxonomy_manifest = self.ind_eval._resolve_path(
            root_cfg.get("taxonomy_manifest"),
            project_root=project_root,
            base_dir=root_config.parent,
        )
        class_column = str(
            root_cfg.get("taxonomy_class_column", "folder_name")
        )
        self.tree = self.ind_eval._taxonomy_from_manifest(
            taxonomy_manifest,
            class_column,
        )
        specs = self.ind_eval._discover_node_specs(run_root, self.tree)
        self.nodes = {
            spec.key: self.ind_eval._load_node(
                spec,
                project_root=project_root,
                tree=self.tree,
                device=device,
                allow_order_fallback=False,
            )
            for spec in specs
        }
        self.ind_eval._validate_preprocessing(list(self.nodes.values()))

        for node in self.nodes.values():
            core_model(node.model).params.vis_attn = True
            node.params.vis_attn = True

        self.root_node = self.nodes["root"]
        self.transform = JointImageTransform(
            self.root_node.params,
            training=False,
        )

        self.class_names = tuple(self.tree.species_names)
        self.species_to_genus = tuple(int(x) for x in self.tree.species_to_genus)
        self.genus_to_family = tuple(int(x) for x in self.tree.genus_to_family)
        self.species_to_family = tuple(int(x) for x in self.tree.species_to_family)

        self.species_has_siblings = tuple(
            len(self.tree.species_by_genus[
                self.tree.genera[self.species_to_genus[s]]
            ]) > 1
            for s in range(len(self.class_names))
        )

        self.family_to_genera = {
            f: [
                g for g, parent in enumerate(self.genus_to_family)
                if parent == f
            ]
            for f in range(len(self.tree.families))
        }
        self.genus_to_species = {
            g: [
                s for s, parent in enumerate(self.species_to_genus)
                if parent == g
            ]
            for g in range(len(self.tree.genera))
        }

        self.family_index = {
            name: i for i, name in enumerate(self.tree.families)
        }
        self.genus_index = {
            name: i for i, name in enumerate(self.tree.genera)
        }
        self.species_index = {
            name: i for i, name in enumerate(self.tree.species_names)
        }

        self.species_threshold, self.genus_threshold, self.threshold_row = (
            load_fallback_thresholds(
                fallback_csv,
                model_name="Independent",
                reliability_pct=float(reliability),
            )
        )

    def image_tensor(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        tensor, _, _ = self.transform(
            image, bbox=None, bbox_coordinate_mode="normalized"
        )
        return tensor.unsqueeze(0).to(self.device)

    def capture_all_nodes(
        self,
        batch: torch.Tensor,
        *,
        target_species: int,
    ):
        target_species = int(target_species)
        target_genus = self.tree.genera[
            self.species_to_genus[target_species]
        ]
        target_species_name = self.tree.species_names[target_species]
        species_node_key = (
            f"genus__{self.ind_eval._slug(target_genus)}"
        )
        if species_node_key not in self.nodes:
            raise RuntimeError(
                f"target species contrast node가 없습니다: {species_node_key}"
            )

        node_probs: dict[str, torch.Tensor] = {}
        node_features: dict[str, torch.Tensor] = {}
        target_cam = None
        cam_source = None
        max_cached_diff = 0.0

        for key, node in self.nodes.items():
            if key == species_node_key:
                local_target = node.labels.index(target_species_name)
            else:
                local_target = 0

            logits, features, head_maps, diff = capture_features_and_attention(
                node.model,
                batch,
                target_local_index=int(local_target),
                params=node.params,
            )
            node_probs[key] = logits.softmax(dim=1)
            node_features[key] = features
            max_cached_diff = max(max_cached_diff, float(diff))

            if key == species_node_key:
                target_cam, cam_source = aggregate_promptcam_maps(
                    head_maps,
                    core_model(node.model),
                )

        if target_cam is None:
            raise RuntimeError("Independent target species CAM을 만들지 못했습니다")

        return (
            node_probs,
            node_features,
            target_cam,
            str(cam_source),
            max_cached_diff,
        )

    def deleted_node_probabilities(
        self,
        node_features: Mapping[str, torch.Tensor],
        index_sets: list[np.ndarray],
        *,
        replacement: str,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        expected_patch_count = None

        for key, node in self.nodes.items():
            features = node_features[key]
            start, count = patch_slice(
                core_model(node.model),
                features,
            )
            if expected_patch_count is None:
                expected_patch_count = count
            elif count != expected_patch_count:
                raise RuntimeError(
                    f"node별 patch count가 다릅니다: {key}={count}, "
                    f"expected={expected_patch_count}"
                )

            logits = deleted_logits_from_features(
                core_model(node.model),
                features,
                index_sets,
                replacement=replacement,
                batch_size=batch_size,
            )
            result[key] = logits.softmax(dim=1)

        return result

    def combine(self, node_probs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        root = node_probs["root"]
        batch = root.shape[0]
        device = root.device
        dtype = root.dtype

        F = len(self.tree.families)
        G = len(self.tree.genera)
        C = len(self.tree.species_names)

        family_probs = torch.empty(
            batch, F, device=device, dtype=dtype
        )
        root_label_to_local = {
            label: i for i, label in enumerate(self.root_node.labels)
        }
        for family_name, global_f in self.family_index.items():
            family_probs[:, global_f] = root[
                :, root_label_to_local[family_name]
            ]

        genus_cond = torch.ones(
            batch, G, device=device, dtype=dtype
        )
        for family_name in self.tree.families:
            children = self.tree.genera_by_family[family_name]
            if len(children) == 1:
                continue
            key = f"family__{self.ind_eval._slug(family_name)}"
            node = self.nodes[key]
            probs = node_probs[key]
            local_map = {
                label: i for i, label in enumerate(node.labels)
            }
            for genus_name in children:
                genus_cond[:, self.genus_index[genus_name]] = probs[
                    :, local_map[genus_name]
                ]

        species_cond = torch.ones(
            batch, C, device=device, dtype=dtype
        )
        for genus_name in self.tree.genera:
            children = self.tree.species_by_genus[genus_name]
            if len(children) == 1:
                continue
            key = f"genus__{self.ind_eval._slug(genus_name)}"
            node = self.nodes[key]
            probs = node_probs[key]
            local_map = {
                label: i for i, label in enumerate(node.labels)
            }
            for species_name in children:
                species_cond[:, self.species_index[species_name]] = probs[
                    :, local_map[species_name]
                ]

        s2g = torch.as_tensor(
            self.species_to_genus, device=device, dtype=torch.long
        )
        s2f = torch.as_tensor(
            self.species_to_family, device=device, dtype=torch.long
        )
        g2f = torch.as_tensor(
            self.genus_to_family, device=device, dtype=torch.long
        )

        genus_joint = (
            family_probs.index_select(1, g2f) * genus_cond
        )
        leaf = (
            family_probs.index_select(1, s2f)
            * genus_cond.index_select(1, s2g)
            * species_cond
        )
        leaf = leaf / leaf.sum(dim=1, keepdim=True).clamp_min(1e-12)

        family_pred = family_probs.argmax(dim=1)
        family_conf = family_probs.gather(
            1, family_pred[:, None]
        ).squeeze(1)

        genus_pred = torch.empty(
            batch, dtype=torch.long, device=device
        )
        genus_cond_conf = torch.empty(
            batch, dtype=dtype, device=device
        )
        for f, members_list in self.family_to_genera.items():
            mask = family_pred.eq(int(f))
            if not mask.any():
                continue
            members = torch.as_tensor(
                members_list, dtype=torch.long, device=device
            )
            local = genus_cond[mask].index_select(1, members)
            best, local_idx = local.max(dim=1)
            genus_pred[mask] = members[local_idx]
            genus_cond_conf[mask] = best

        species_pred_greedy = torch.empty(
            batch, dtype=torch.long, device=device
        )
        species_cond_conf = torch.empty(
            batch, dtype=dtype, device=device
        )
        for g, members_list in self.genus_to_species.items():
            mask = genus_pred.eq(int(g))
            if not mask.any():
                continue
            members = torch.as_tensor(
                members_list, dtype=torch.long, device=device
            )
            local = species_cond[mask].index_select(1, members)
            best, local_idx = local.max(dim=1)
            species_pred_greedy[mask] = members[local_idx]
            species_cond_conf[mask] = best

        genus_joint_conf = family_conf * genus_cond_conf
        species_joint_conf = (
            genus_joint_conf * species_cond_conf
        )

        emitted_depth = torch.ones(
            batch, dtype=torch.long, device=device
        )
        emitted_depth[
            genus_joint_conf.ge(float(self.genus_threshold))
        ] = 2
        emitted_depth[
            species_joint_conf.ge(float(self.species_threshold))
        ] = 3

        return {
            "family_probabilities": family_probs,
            "genus_conditional_probabilities": genus_cond,
            "species_conditional_probabilities": species_cond,
            "genus_probabilities": genus_joint,
            "species_probabilities": leaf,
            "family_prediction": family_pred,
            "genus_prediction": genus_pred,
            "species_prediction_greedy": species_pred_greedy,
            "family_confidence": family_conf,
            "genus_joint_confidence": genus_joint_conf,
            "species_joint_confidence": species_joint_conf,
            "emitted_depth": emitted_depth,
        }

    def decision_metrics(
        self,
        combined: Mapping[str, torch.Tensor],
        *,
        target_species: int,
    ) -> dict[str, torch.Tensor]:
        probs = combined["species_probabilities"].float()
        target_species = int(target_species)
        margin, best_competitor, target_logp = log_probability_margin(
            probs, target_species
        )

        leaf_pred = probs.argmax(dim=1)

        true_genus = int(self.species_to_genus[target_species])
        true_family = int(self.species_to_family[target_species])

        depth = combined["emitted_depth"]
        emitted_correct = torch.where(
            depth.eq(3),
            combined["species_prediction_greedy"].eq(target_species),
            torch.where(
                depth.eq(2),
                combined["genus_prediction"].eq(true_genus),
                combined["family_prediction"].eq(true_family),
            ),
        )

        return {
            "joint_species_prediction": leaf_pred,
            "target_probability": probs[:, target_species],
            "target_log_probability": target_logp,
            "best_competitor_species": best_competitor,
            "log_probability_margin": margin,
            "emitted_depth": depth,
            "emitted_correct": emitted_correct,
        }


def first_true_fraction(
    fractions: list[float],
    indicators: list[bool],
) -> float:
    for f, flag in zip(fractions, indicators):
        if bool(flag):
            return float(f)
    return math.nan


def depth_name(depth: int) -> str:
    return {1: "family", 2: "genus", 3: "species"}[int(depth)]


def summarize(curve: pd.DataFrame, model_name: str) -> pd.DataFrame:
    base_aggs = dict(
        n_images=("image_path", "size"),
        n_species=("species", "nunique"),
        top_flip_rate=("top_species_flip", "mean"),
        random_flip_rate=("random_species_flip_rate", "mean"),
        mean_top_minus_random_flip=(
            "top_minus_random_flip", "mean"
        ),
        mean_top_margin_drop=("top_margin_drop", "mean"),
        mean_random_margin_drop=("random_margin_drop_mean", "mean"),
        mean_top_minus_random_margin_drop=(
            "top_minus_random_margin_drop", "mean"
        ),
    )

    if model_name == "Independent":
        base_aggs.update(
            mean_baseline_depth=("baseline_emitted_depth", "mean"),
            mean_top_depth=("top_emitted_depth", "mean"),
            mean_random_depth=("random_emitted_depth_mean", "mean"),
            mean_top_minus_random_depth_loss=(
                "top_minus_random_depth_loss", "mean"
            ),
            baseline_species_emitted_n=(
                "baseline_species_emitted_correct", "sum"
            ),
            top_species_to_fallback_rate=(
                "top_species_to_fallback", "mean"
            ),
            random_species_to_fallback_rate=(
                "random_species_to_fallback_rate", "mean"
            ),
            mean_top_minus_random_species_fallback=(
                "top_minus_random_species_fallback", "mean"
            ),
            top_emitted_accuracy=("top_emitted_correct", "mean"),
            random_emitted_accuracy=(
                "random_emitted_correct_rate", "mean"
            ),
        )

    return (
        curve.groupby("fraction")
        .agg(**base_aggs)
        .reset_index()
    )


def run_flat(args, project_root, device, fractions):
    runtime = FlatRuntime(
        project_root,
        resolve(project_root, args.flat_run_dir),
        device,
    )
    test_root = resolve(project_root, args.test_root)

    items = []
    for s, name in enumerate(runtime.class_names):
        class_dir = test_root / name
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                items.append((s, name, p.resolve()))

    if args.max_images is not None:
        items = items[: int(args.max_images)]

    curve_rows = []
    minimum_rows = []
    incorrect_skip = 0
    singleton_skip = 0
    usable = 0
    printed_cam_source = False

    for image_no, (species_index, species_name, image_path) in enumerate(
        items, start=1
    ):
        if not runtime.species_has_siblings[species_index]:
            singleton_skip += 1
            continue

        batch = runtime.image_tensor(image_path)
        probs, features, cam, cam_source, cached_diff = runtime.baseline(
            batch, species_index
        )
        pred = int(probs[0].argmax().item())
        if pred != species_index:
            incorrect_skip += 1
            continue

        usable += 1
        if not printed_cam_source:
            print(f"[CAM head weighting] {cam_source}")
            print(f"[cached forward max diff] {cached_diff:.3g}")
            printed_cam_source = True

        baseline_margin = float(
            log_probability_margin(probs, species_index)[0][0].item()
        )
        patch_count = int(cam.size)
        order = np.argsort(cam)
        rng = image_rng(args.seed, image_path)
        random_perms = np.stack(
            [
                rng.permutation(patch_count).astype(np.int64)
                for _ in range(int(args.random_repeats))
            ],
            axis=0,
        )

        top_flip_flags = []

        for fraction in fractions:
            k = max(
                1,
                min(
                    patch_count,
                    int(round(fraction * patch_count)),
                ),
            )
            top_idx = order[-k:].copy()
            random_sets = [
                random_perms[r, :k].copy()
                for r in range(int(args.random_repeats))
            ]
            deleted_probs = runtime.deleted_probabilities(
                features,
                [top_idx] + random_sets,
                replacement=args.replacement,
                batch_size=args.deletion_batch,
            )

            deleted_pred = deleted_probs.argmax(dim=1)
            top_flip = bool(int(deleted_pred[0].item()) != species_index)
            random_flip = (
                deleted_pred[1:].detach().cpu().numpy()
                != int(species_index)
            )
            random_flip_rate = float(random_flip.mean())

            margins = log_probability_margin(
                deleted_probs, species_index
            )[0]
            top_margin = float(margins[0].item())
            random_margin = margins[1:].detach().float().cpu().numpy()

            top_margin_drop = baseline_margin - top_margin
            random_margin_drop = baseline_margin - random_margin

            curve_rows.append(
                {
                    "model": "Flat",
                    "image_path": str(image_path),
                    "species": species_name,
                    "species_index": species_index,
                    "fraction": fraction,
                    "deleted_patch_count": k,
                    "patch_count": patch_count,
                    "replacement": args.replacement,
                    "random_repeats": int(args.random_repeats),
                    "baseline_probability": float(
                        probs[0, species_index].item()
                    ),
                    "baseline_margin": baseline_margin,
                    "top_species_prediction": int(
                        deleted_pred[0].item()
                    ),
                    "top_species_flip": float(top_flip),
                    "random_species_flip_rate": random_flip_rate,
                    "top_minus_random_flip": (
                        float(top_flip) - random_flip_rate
                    ),
                    "top_margin": top_margin,
                    "random_margin_mean": float(random_margin.mean()),
                    "top_margin_drop": top_margin_drop,
                    "random_margin_drop_mean": float(
                        random_margin_drop.mean()
                    ),
                    "top_minus_random_margin_drop": float(
                        top_margin_drop
                        - random_margin_drop.mean()
                    ),
                    "cam_head_weight_source": cam_source,
                    "cached_full_max_probability_diff": cached_diff,
                }
            )
            top_flip_flags.append(top_flip)

        minimum_rows.append(
            {
                "model": "Flat",
                "image_path": str(image_path),
                "species": species_name,
                "species_index": species_index,
                "top_first_flip_fraction": first_true_fraction(
                    fractions, top_flip_flags
                ),
                "top_no_flip_through_max": bool(
                    not any(top_flip_flags)
                ),
                "max_fraction_tested": max(fractions),
            }
        )

        if image_no % 25 == 0 or image_no == len(items):
            print(
                f"[{image_no}/{len(items)}] usable={usable}, "
                f"incorrect-skip={incorrect_skip}, singleton-skip={singleton_skip}"
            )

    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(minimum_rows),
        {
            "usable": usable,
            "incorrect_skip": incorrect_skip,
            "singleton_skip": singleton_skip,
        },
    )


def run_independent(args, project_root, device, fractions):
    fallback_csv = resolve(project_root, args.fallback_results_csv)
    runtime = IndependentRuntime(
        project_root,
        resolve(project_root, args.independent_run_root),
        device,
        fallback_csv=fallback_csv,
        reliability=float(args.reliability),
    )
    print(
        "[fallback thresholds] "
        f"Independent reliability={args.reliability:.1f}% "
        f"tauS={runtime.species_threshold:.9f} "
        f"tauG={runtime.genus_threshold:.9f}"
    )

    test_root = resolve(project_root, args.test_root)
    items = []
    for s, name in enumerate(runtime.class_names):
        class_dir = test_root / name
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                items.append((s, name, p.resolve()))

    if args.max_images is not None:
        items = items[: int(args.max_images)]

    curve_rows = []
    minimum_rows = []
    incorrect_skip = 0
    singleton_skip = 0
    usable = 0
    printed_cam_source = False

    for image_no, (species_index, species_name, image_path) in enumerate(
        items, start=1
    ):
        if not runtime.species_has_siblings[species_index]:
            singleton_skip += 1
            continue

        batch = runtime.image_tensor(image_path)
        (
            base_node_probs,
            node_features,
            cam,
            cam_source,
            cached_diff,
        ) = runtime.capture_all_nodes(
            batch,
            target_species=species_index,
        )
        baseline_combined = runtime.combine(base_node_probs)
        baseline_metrics = runtime.decision_metrics(
            baseline_combined,
            target_species=species_index,
        )

        baseline_leaf_pred = int(
            baseline_metrics["joint_species_prediction"][0].item()
        )
        if baseline_leaf_pred != species_index:
            incorrect_skip += 1
            continue

        usable += 1
        if not printed_cam_source:
            print(f"[CAM head weighting] {cam_source}")
            print(f"[cached forward max diff] {cached_diff:.3g}")
            printed_cam_source = True

        baseline_prob = float(
            baseline_metrics["target_probability"][0].item()
        )
        baseline_margin = float(
            baseline_metrics["log_probability_margin"][0].item()
        )
        baseline_depth = int(
            baseline_metrics["emitted_depth"][0].item()
        )
        baseline_emitted_correct = bool(
            baseline_metrics["emitted_correct"][0].item()
        )
        baseline_species_emitted_correct = bool(
            baseline_depth == 3 and baseline_emitted_correct
        )

        patch_count = int(cam.size)
        order = np.argsort(cam)
        rng = image_rng(args.seed, image_path)
        random_perms = np.stack(
            [
                rng.permutation(patch_count).astype(np.int64)
                for _ in range(int(args.random_repeats))
            ],
            axis=0,
        )

        top_flip_flags = []
        top_fallback_flags = []

        for fraction in fractions:
            k = max(
                1,
                min(
                    patch_count,
                    int(round(fraction * patch_count)),
                ),
            )
            top_idx = order[-k:].copy()
            random_sets = [
                random_perms[r, :k].copy()
                for r in range(int(args.random_repeats))
            ]
            index_sets = [top_idx] + random_sets

            deleted_node_probs = runtime.deleted_node_probabilities(
                node_features,
                index_sets,
                replacement=args.replacement,
                batch_size=args.deletion_batch,
            )
            deleted_combined = runtime.combine(deleted_node_probs)
            deleted_metrics = runtime.decision_metrics(
                deleted_combined,
                target_species=species_index,
            )

            pred = (
                deleted_metrics["joint_species_prediction"]
                .detach()
                .cpu()
                .numpy()
            )
            top_flip = bool(int(pred[0]) != species_index)
            random_flip = pred[1:] != int(species_index)
            random_flip_rate = float(random_flip.mean())

            margins = (
                deleted_metrics["log_probability_margin"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            top_margin = float(margins[0])
            random_margin = margins[1:]
            top_margin_drop = baseline_margin - top_margin
            random_margin_drop = baseline_margin - random_margin

            depth = (
                deleted_metrics["emitted_depth"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            top_depth = int(depth[0])
            random_depth = depth[1:]
            top_depth_loss = baseline_depth - top_depth
            random_depth_loss = baseline_depth - random_depth

            emitted_correct = (
                deleted_metrics["emitted_correct"]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

            if baseline_species_emitted_correct:
                top_fb = float(top_depth < 3)
                random_fb = random_depth < 3
                random_fb_rate = float(random_fb.mean())
            else:
                top_fb = math.nan
                random_fb_rate = math.nan

            curve_rows.append(
                {
                    "model": "Independent",
                    "image_path": str(image_path),
                    "species": species_name,
                    "species_index": species_index,
                    "fraction": fraction,
                    "deleted_patch_count": k,
                    "patch_count": patch_count,
                    "replacement": args.replacement,
                    "random_repeats": int(args.random_repeats),
                    "fallback_reliability_pct": float(args.reliability),
                    "species_threshold": runtime.species_threshold,
                    "genus_threshold": runtime.genus_threshold,
                    "baseline_probability": baseline_prob,
                    "baseline_margin": baseline_margin,
                    "baseline_emitted_depth": baseline_depth,
                    "baseline_emitted_rank": depth_name(baseline_depth),
                    "baseline_emitted_correct": baseline_emitted_correct,
                    "baseline_species_emitted_correct": (
                        baseline_species_emitted_correct
                    ),
                    "top_species_prediction": int(pred[0]),
                    "top_species_flip": float(top_flip),
                    "random_species_flip_rate": random_flip_rate,
                    "top_minus_random_flip": (
                        float(top_flip) - random_flip_rate
                    ),
                    "top_margin": top_margin,
                    "random_margin_mean": float(random_margin.mean()),
                    "top_margin_drop": top_margin_drop,
                    "random_margin_drop_mean": float(
                        random_margin_drop.mean()
                    ),
                    "top_minus_random_margin_drop": float(
                        top_margin_drop
                        - random_margin_drop.mean()
                    ),
                    "top_emitted_depth": top_depth,
                    "top_emitted_rank": depth_name(top_depth),
                    "random_emitted_depth_mean": float(
                        random_depth.mean()
                    ),
                    "top_depth_loss": float(top_depth_loss),
                    "random_depth_loss_mean": float(
                        random_depth_loss.mean()
                    ),
                    "top_minus_random_depth_loss": float(
                        top_depth_loss
                        - random_depth_loss.mean()
                    ),
                    "top_emitted_correct": float(
                        emitted_correct[0]
                    ),
                    "random_emitted_correct_rate": float(
                        emitted_correct[1:].mean()
                    ),
                    "top_species_to_fallback": top_fb,
                    "random_species_to_fallback_rate": random_fb_rate,
                    "top_minus_random_species_fallback": (
                        top_fb - random_fb_rate
                        if not math.isnan(top_fb)
                        else math.nan
                    ),
                    "cam_head_weight_source": cam_source,
                    "cached_full_max_probability_diff": cached_diff,
                }
            )

            top_flip_flags.append(top_flip)
            top_fallback_flags.append(
                bool(
                    baseline_species_emitted_correct
                    and top_depth < 3
                )
            )

        minimum_rows.append(
            {
                "model": "Independent",
                "image_path": str(image_path),
                "species": species_name,
                "species_index": species_index,
                "baseline_emitted_depth": baseline_depth,
                "baseline_species_emitted_correct": (
                    baseline_species_emitted_correct
                ),
                "top_first_flip_fraction": first_true_fraction(
                    fractions, top_flip_flags
                ),
                "top_no_flip_through_max": bool(
                    not any(top_flip_flags)
                ),
                "top_first_fallback_fraction": (
                    first_true_fraction(
                        fractions,
                        top_fallback_flags,
                    )
                    if baseline_species_emitted_correct
                    else math.nan
                ),
                "max_fraction_tested": max(fractions),
            }
        )

        if image_no % 25 == 0 or image_no == len(items):
            print(
                f"[{image_no}/{len(items)}] usable={usable}, "
                f"incorrect-skip={incorrect_skip}, singleton-skip={singleton_skip}"
            )

    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(minimum_rows),
        {
            "usable": usable,
            "incorrect_skip": incorrect_skip,
            "singleton_skip": singleton_skip,
            "species_threshold": runtime.species_threshold,
            "genus_threshold": runtime.genus_threshold,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["flat", "independent"],
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--flat-run-dir")
    parser.add_argument("--independent-run-root")
    parser.add_argument(
        "--fallback-results-csv",
        default="output/hierarchical_fallback_comparison_dense/fallback_results.csv",
    )
    parser.add_argument("--reliability", type=float, default=94.0)
    parser.add_argument(
        "--test-root",
        default="data/dataset/imagefolder/test",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[
            0.05, 0.10, 0.20, 0.30, 0.40,
            0.50, 0.60, 0.70, 0.80, 0.90,
        ],
    )
    parser.add_argument(
        "--replacement",
        choices=["mean", "zero"],
        default="mean",
    )
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--deletion-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다")

    fractions = sorted(set(float(x) for x in args.fractions))
    if any(x <= 0 or x > 1 for x in fractions):
        raise ValueError("fractions는 (0,1] 범위여야 합니다")

    if args.model == "flat":
        if not args.flat_run_dir:
            raise ValueError("--model flat에는 --flat-run-dir이 필요합니다")
        model_name = "Flat"
        curve, minima, meta = run_flat(
            args, project_root, device, fractions
        )
    else:
        if not args.independent_run_root:
            raise ValueError(
                "--model independent에는 --independent-run-root가 필요합니다"
            )
        model_name = "Independent"
        curve, minima, meta = run_independent(
            args, project_root, device, fractions
        )

    if curve.empty:
        raise RuntimeError("분석 결과가 없습니다")

    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_path = output_dir / "decision_deletion_per_image_fraction.csv"
    minima_path = output_dir / "decision_minimal_fraction_per_image.csv"
    summary_path = output_dir / "decision_deletion_summary.csv"

    curve.to_csv(curve_path, index=False)
    minima.to_csv(minima_path, index=False)
    summary = summarize(curve, model_name)
    summary.to_csv(summary_path, index=False)

    print()
    print(f"===== {model_name.upper()} DECISION-DELETION SUMMARY =====")
    print(summary.to_string(index=False))
    print()
    print(meta)
    print(f"[저장] {curve_path}")
    print(f"[저장] {minima_path}")
    print(f"[저장] {summary_path}")


if __name__ == "__main__":
    main()
