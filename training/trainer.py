"""학습, EMA teacher, Semantic SnapMix, 평가, checkpoint 저장을 통합 관리한다."""

from __future__ import annotations

import copy
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler

from training.semantic_snapmix import (
    eta_for_epoch,
    hierarchical_promptcam_spm,
    promptcam_spm,
    semantic_snapmix,
    should_apply_snapmix,
    unchanged_snapmix_batch,
    uniform_spm,
)
from training.loss import (
    HierarchicalTaxonomicCriterion,
    RANK_FAMILY,
    RANK_GENUS,
    RANK_SPECIES,
)
from training.optimizer import make_optimizer
from utils.misc import AverageMeter
from utils.setup_logging import get_logger


logger = get_logger("Prompt_CAM")
torch.backends.cudnn.benchmark = False


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """DDP wrapper가 있으면 내부 원본 모델을 반환한다."""
    return model.module if hasattr(model, "module") else model




def _configure_flat_snapmix_teacher_attention(
    teacher: torch.nn.Module,
    *,
    semantic_snapmix: bool,
    hierarchical: bool,
) -> None:
    """비계층 Semantic SnapMix teacher가 마지막 self-attention을 반환하게 한다.

    비계층 Prompt-CAM의 SPM은 마지막 block의 prompt-to-patch attention을
    사용한다. Student의 일반 학습 forward에는 attention 반환이 필요 없으므로
    EMA teacher에만 ``vis_attn=True``를 적용한다.
    """
    if not semantic_snapmix or hierarchical:
        return
    teacher_params = getattr(teacher, "params", None)
    if teacher_params is None:
        raise RuntimeError(
            "비계층 Semantic SnapMix teacher는 params.vis_attn 설정을 제공해야 합니다"
        )
    teacher_params.vis_attn = True


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """모든 process의 tensor를 합산한 뒤 world size로 나눈다."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _squeeze_logits(logits: torch.Tensor) -> torch.Tensor:
    """로짓의 불필요한 마지막 singleton 차원을 제거한다."""
    if logits.ndim == 3 and logits.shape[-1] == 1:
        return logits.squeeze(-1)
    if logits.ndim != 2:
        raise ValueError(f"로짓은 [B,C] 또는 [B,C,1]이어야 하지만 {tuple(logits.shape)}입니다")
    return logits


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    """각 표본의 정답이 상위 k개 예측 안에 포함되는지 계산한다."""
    if targets.numel() == 0:
        return 0
    k = min(k, logits.shape[1])
    return int(logits.topk(k, dim=1).indices.eq(targets[:, None]).any(dim=1).sum().item())


def _metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, float]:
    """confusion matrix에서 accuracy와 macro-F1을 계산한다."""
    confusion = confusion.to(dtype=torch.float64)
    total = confusion.sum()
    if total <= 0:
        return {"top1": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    active = support > 0
    recall = true_positive / support.clamp_min(1.0)
    precision = true_positive / predicted.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "top1": float((true_positive.sum() / total * 100.0).item()),
        "balanced_accuracy": float((recall[active].mean() * 100.0).item()) if active.any() else 0.0,
        "macro_f1": float((f1[active].mean() * 100.0).item()) if active.any() else 0.0,
    }


def _move_batch(batch, device: torch.device) -> Dict[str, torch.Tensor | list[str]]:
    """batch 안의 tensor만 지정한 device로 비동기 이동한다."""
    if isinstance(batch, dict):
        result = {}
        for key, value in batch.items():
            result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        return result
    samples, targets = batch
    targets = targets.to(device, non_blocking=True)
    return {
        "image": samples.to(device, non_blocking=True),
        "species_target": targets,
        "genus_target": targets,
        "family_target": targets,
        "rank_target": torch.zeros_like(targets),
        "bbox": torch.zeros(targets.shape[0], 4, device=device),
        "bbox_valid": torch.zeros(targets.shape[0], dtype=torch.bool, device=device),
    }


class Trainer:
    """학습·평가·EMA·SnapMix·checkpoint 수명주기를 관리한다."""
    def __init__(self, model, tune_parameters, params) -> None:
        """모델, criterion, optimizer, scheduler, AMP, EMA teacher 상태를 초기화한다."""
        self.params = params
        self.model = model
        self.device = params.device
        self.num_classes = int(params.class_num)
        self.hierarchical = bool(getattr(params, "hierarchical_prompt", False))
        self.semantic_snapmix = bool(getattr(params, "semantic_snapmix", False))
        self.selection_metric = str(getattr(params, "selection_metric", "macro_f1"))
        allowed = {
            "top1",
            "balanced_accuracy",
            "macro_f1",
            "genus_accuracy",
            "family_accuracy",
            "rank_accuracy",
        }
        if self.selection_metric not in allowed:
            raise ValueError(f"지원하지 않는 selection_metric입니다: {self.selection_metric}")

        if self.hierarchical:
            self.hierarchical_criterion = HierarchicalTaxonomicCriterion(params).to(self.device)
            self.species_to_genus = torch.as_tensor(
                params.species_to_genus,
                dtype=torch.long,
                device=self.device,
            )
            self.genus_to_family = torch.as_tensor(
                params.genus_to_family,
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.train_criterion = nn.CrossEntropyLoss(reduction="none")
            self.eval_criterion = nn.CrossEntropyLoss(reduction="sum")

        self.optimizer = make_optimizer(tune_parameters, params)
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=int(params.epoch),
            warmup_t=int(params.warmup_epoch),
            lr_min=float(params.lr_min),
            warmup_lr_init=float(params.warmup_lr_init),
        )
        self.total_epoch = int(params.epoch)

        amp_name = str(getattr(params, "amp_dtype", "none")).lower()
        if amp_name not in {"none", "float16", "bfloat16"}:
            raise ValueError("amp_dtype은 none, float16, bfloat16 중 하나여야 합니다")
        self.amp_name = amp_name
        self.amp_enabled = self.device.type == "cuda" and amp_name != "none"
        self.amp_dtype = torch.float16 if amp_name == "float16" else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        student = _unwrap(self.model)
        self.teacher = copy.deepcopy(student).to(self.device).eval()
        _configure_flat_snapmix_teacher_attention(
            self.teacher,
            semantic_snapmix=self.semantic_snapmix,
            hierarchical=self.hierarchical,
        )
        # teacher는 역전파하지 않고 EMA 식으로만 갱신한다.
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        teacher_parameters = dict(self.teacher.named_parameters())
        self._ema_parameter_pairs = tuple(
            (teacher_parameters[name], student_parameter)
            for name, student_parameter in student.named_parameters()
            if student_parameter.requires_grad
        )

        self.start_epoch = 0
        self.best_value = float("-inf")
        self.best_metrics: Dict[str, float] = {}
        self.patience_counter = 0

        # true이면 model.pt와 last.pt를 디스크에 저장한다.
        self.store_checkpoint = bool(
            getattr(
                params,
                "store_ckp",
                False,
            )
        )

        # checkpoint 저장 여부와 무관하게 현재 실행의 최선 모델을 메모리에 보관한다.
        self._best_student_state: Optional[
            Dict[str, torch.Tensor]
        ] = None
        resume_path = getattr(params, "resume", None)
        if resume_path not in (None, "", "null"):
            self._load_resume(Path(str(resume_path)).expanduser().resolve())

    def is_main_process(self) -> bool:
        """현재 process가 로그와 checkpoint를 담당하는 rank 0인지 반환한다."""
        return (not getattr(self.params, "distributed", False)) or dist.get_rank() == 0

    def _autocast(self):
        """설정된 precision에 맞는 autocast context를 반환한다."""
        if not self.amp_enabled:
            return nullcontext()
        return torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def _model_forward(
        self,
        model: torch.nn.Module,
        samples: torch.Tensor,
        *,
        patch_prior: Optional[torch.Tensor] = None,
    ):
        """계층 모델과 기존 Prompt-CAM 모델의 서로 다른 forward signature를 통일한다."""
        if self.hierarchical:
            output, attention = model(samples, patch_prior=patch_prior)
        else:
            output, attention = model(samples)
        if isinstance(output, dict):
            return output, attention
        return _squeeze_logits(output), attention

    def _patch_prior(
        self,
        batch: Dict[str, torch.Tensor | list[str]],
        *,
        excluded_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """criterion을 통해 현재 batch의 bbox patch prior를 생성한다."""
        if not self.hierarchical:
            return None
        return self.hierarchical_criterion.build_patch_prior(
            batch, excluded_mask=excluded_mask
        )

    @torch.no_grad()
    def _update_teacher(self) -> None:
        """student 파라미터의 지수이동평균으로 teacher를 갱신한다."""
        decay = float(getattr(self.params, "ema_decay", 0.999))
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema_decay는 0 <= decay < 1을 만족해야 합니다")
        for teacher_param, student_param in self._ema_parameter_pairs:
            teacher_param.mul_(decay).add_(student_param.detach(), alpha=1.0 - decay)

    def _snapmix_batch(self, batch: Dict[str, torch.Tensor | list[str]], epoch: int):
        """EMA teacher CAM으로 현재 batch에 적용할 Semantic SnapMix 결과를 만든다."""
        samples = batch["image"]
        targets = batch["species_target"]
        # 초기에는 균등 면적 질량을, 후반에는 CAM 의미 질량을 더 강하게 사용한다.
        eta = eta_for_epoch(
            epoch,
            int(getattr(self.params, "eta_start_epoch", 5)),
            int(getattr(self.params, "eta_end_epoch", 15)),
        )
        student = _unwrap(self.model)
        patch_size = getattr(student, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(self.params, "patch_size", None)
        if patch_size is None:
            raise ValueError("활성 백본은 patch_size를 제공해야 합니다")
        probability = float(getattr(self.params, "mix_probability", 0.5))
        beta = float(getattr(self.params, "mix_beta", 1.0))
        if beta <= 0.0:
            raise ValueError("mix_beta는 양수여야 합니다")
        eligible_mask = (
            batch["rank_target"].eq(RANK_SPECIES) if self.hierarchical else None
        )
        apply_mix = should_apply_snapmix(
            int(samples.shape[0]),
            probability,
            eligible_mask,
        )
        if not apply_mix:
            return (
                unchanged_snapmix_batch(
                    samples,
                    targets,
                    genus_targets=(
                        batch["genus_target"]
                        if self.hierarchical
                        else None
                    ),
                    family_targets=(
                        batch["family_target"]
                        if self.hierarchical
                        else None
                    ),
                ),
                eta,
            )

        with self._autocast():
            if self.hierarchical:
                if eta == 0.0:
                    species_spm = uniform_spm(samples)
                    genus_spm = species_spm
                    family_spm = species_spm
                else:
                    (
                        species_spm,
                        genus_spm,
                        family_spm,
                        _,
                    ) = hierarchical_promptcam_spm(
                        self.teacher,
                        samples,
                        targets,
                        batch["genus_target"],
                        batch["family_target"],
                        patch_size=patch_size,
                        eta=eta,
                        eps=float(getattr(self.params, "spm_eps", 1e-8)),
                        patch_prior=self._patch_prior(batch),
                    )
                mix = semantic_snapmix(
                    samples,
                    targets,
                    species_spm,
                    probability=probability,
                    beta=beta,
                    eps=float(getattr(self.params, "spm_eps", 1e-8)),
                    genus_targets=batch["genus_target"],
                    genus_spm=genus_spm,
                    family_targets=batch["family_target"],
                    family_spm=family_spm,
                    eligible_mask=eligible_mask,
                    apply=True,
                )
            else:
                if eta == 0.0:
                    spm = uniform_spm(samples)
                else:
                    prefix_count = int(getattr(student, "num_prefix_tokens", 1))
                    spm, _ = promptcam_spm(
                        self.teacher,
                        samples,
                        targets,
                        prompt_count=int(self.params.vpt_num),
                        prefix_token_count=prefix_count,
                        patch_size=patch_size,
                        eta=eta,
                        head_reduction=str(getattr(self.params, "spm_head_reduction", "mean")),
                        eps=float(getattr(self.params, "spm_eps", 1e-8)),
                    )
                mix = semantic_snapmix(
                    samples,
                    targets,
                    spm,
                    probability=probability,
                    beta=beta,
                    eps=float(getattr(self.params, "spm_eps", 1e-8)),
                    apply=True,
                )
        return mix, eta

    def train_one_epoch(self, epoch: int, loader) -> OrderedDict:
        """한 epoch 동안 forward, loss, backward, optimizer step, metric 집계를 수행한다."""
        meters: Dict[str, AverageMeter] = {"loss": AverageMeter(), "primary_top1": AverageMeter()}
        applied_samples = 0
        total_samples = 0
        self.model.train()
        self.teacher.eval()

        if getattr(self.params, "distributed", False) and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        num_updates = epoch * len(loader)
        current_eta = (
            eta_for_epoch(
                epoch,
                int(getattr(self.params, "eta_start_epoch", 5)),
                int(getattr(self.params, "eta_end_epoch", 15)),
            )
            if self.semantic_snapmix
            else 0.0
        )
        if self.is_main_process():
            logger.info(
                "학습 에폭 %d / %d, 학습률=%s, 계층=%s, 의미 기반 SnapMix=%s, eta=%.4f",
                epoch + 1,
                self.total_epoch,
                self.scheduler._get_lr(epoch),
                self.hierarchical,
                self.semantic_snapmix,
                current_eta,
            )

        for raw_batch in loader:
            batch = _move_batch(raw_batch, self.device)
            mix = None
            train_samples = batch["image"]
            if self.semantic_snapmix:
                mix, _ = self._snapmix_batch(batch, epoch)
                train_samples = mix.images
                if mix.applied_mask is not None:
                    applied_samples += int(mix.applied_mask.sum().item())

            # set_to_none은 불필요한 gradient zero 연산과 메모리 쓰기를 줄인다.
            self.optimizer.zero_grad(set_to_none=True)
            excluded_from_bbox = (
                mix.applied_mask if mix is not None and mix.applied_mask is not None else None
            )
            patch_prior = self._patch_prior(batch, excluded_mask=excluded_from_bbox)
            with self._autocast():
                output, _ = self._model_forward(
                    self.model, train_samples, patch_prior=patch_prior
                )
                if self.hierarchical:
                    regularization = _unwrap(self.model).hierarchical_regularization()
                    loss, components = self.hierarchical_criterion(
                        output, batch, mix=mix, regularization=regularization
                    )
                    logits = output["species_logits"]
                else:
                    logits = output
                    if mix is None:
                        loss = self.train_criterion(logits, batch["species_target"]).mean()
                    else:
                        loss_a = self.train_criterion(logits, mix.target_a)
                        loss_b = self.train_criterion(logits, mix.target_b)
                        loss = (
                            loss_a * mix.weight_a.to(loss_a.dtype)
                            + loss_b * mix.weight_b.to(loss_b.dtype)
                        ).mean()
                    components = {"loss_total": loss.detach()}

            if not torch.isfinite(loss):
                raise FloatingPointError(f"에폭 {epoch + 1}에서 학습 손실이 유한하지 않습니다")
            if self.scaler.is_enabled():
                # AMP 사용 시 scaled loss로 backward한 뒤 overflow를 검사하며 optimizer step을 수행한다.
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            # student가 갱신된 직후 동일 step의 EMA teacher를 갱신한다.
            self._update_teacher()

            batch_size = int(batch["species_target"].shape[0])
            total_samples += batch_size
            meters["loss"].update(float(loss.detach().item()), batch_size)
            primary = (
                logits.detach().argmax(dim=1).eq(batch["species_target"]).float().mean().item() * 100.0
            )
            meters["primary_top1"].update(primary, batch_size)
            for name, value in components.items():
                if name not in meters:
                    meters[name] = AverageMeter()
                meters[name].update(float(value.item()), batch_size)

            num_updates += 1
            self.scheduler.step_update(num_updates=num_updates, metric=meters["loss"].avg)

        metrics = OrderedDict(
            (name, round(meter.avg, 6)) for name, meter in sorted(meters.items())
        )
        metrics["eta"] = round(current_eta, 6)
        metrics["mixed_sample_fraction"] = round(applied_samples / max(1, total_samples), 6)
        if self.is_main_process():
            logger.info("에폭 %d 학습 지표: %s", epoch + 1, dict(metrics))
        return metrics

    @torch.no_grad()
    def eval_classifier(self, loader, prefix: str) -> OrderedDict:
        """주어진 loader에서 손실과 분류 지표를 gradient 없이 평가한다."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_samples = torch.tensor(0.0, device=self.device)
        top5_correct = torch.tensor(0.0, device=self.device)
        species_samples = torch.tensor(0.0, device=self.device)
        genus_correct = torch.tensor(0.0, device=self.device)
        genus_samples = torch.tensor(0.0, device=self.device)
        family_correct = torch.tensor(0.0, device=self.device)
        family_samples = torch.tensor(0.0, device=self.device)
        rank_correct = torch.tensor(0.0, device=self.device)
        rank_samples = torch.tensor(0.0, device=self.device)
        taxonomy_distance_sum = torch.tensor(0.0, device=self.device)
        # 전체 클래스의 예측–정답 빈도를 누적해 macro-F1을 계산한다.
        confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.float64, device=self.device
        )

        for raw_batch in loader:
            batch = _move_batch(raw_batch, self.device)

            # 검증과 테스트에서는 정답 경계 상자를 모델 입력으로 사용하지 않는다.
            # 실제 이미지 정보만으로 추론해야 하므로 patch_prior는 항상 None이다.
            with self._autocast():
                output, _ = self._model_forward(
                    self.model,
                    batch["image"],
                    patch_prior=None,
                )
                if self.hierarchical:
                    regularization = _unwrap(self.model).hierarchical_regularization()
                    loss, _ = self.hierarchical_criterion(
                        output, batch, regularization=regularization
                    )
                    logits = output["species_logits"]
                else:
                    logits = output
                    loss = F.cross_entropy(logits, batch["species_target"])
            batch_size = int(batch["species_target"].shape[0])
            total_loss += loss.float() * batch_size
            total_samples += batch_size

            if self.hierarchical:
                rank = batch["rank_target"]
                species_mask = rank.eq(RANK_SPECIES)
                genus_mask = rank.le(RANK_GENUS)
                family_mask = rank.le(RANK_FAMILY)
                species_prediction = logits.argmax(dim=1)
                if species_mask.any():
                    true_species = batch["species_target"][species_mask]
                    pred_species = species_prediction[species_mask]
                    species_samples += species_mask.sum()
                    top5_correct += _topk_correct(logits[species_mask], true_species, 5)
                    indices = true_species * self.num_classes + pred_species
                    confusion += torch.bincount(
                        indices,
                        minlength=self.num_classes * self.num_classes,
                    ).reshape(self.num_classes, self.num_classes)

                    true_genus_for_distance = self.species_to_genus[true_species]
                    pred_genus_for_distance = self.species_to_genus[pred_species]
                    true_family_for_distance = self.genus_to_family[
                        true_genus_for_distance
                    ]
                    pred_family_for_distance = self.genus_to_family[
                        pred_genus_for_distance
                    ]
                    distance = torch.full_like(
                        true_species,
                        3,
                        dtype=torch.long,
                    )
                    distance[
                        true_family_for_distance.eq(pred_family_for_distance)
                    ] = 2
                    distance[
                        true_genus_for_distance.eq(pred_genus_for_distance)
                    ] = 1
                    distance[true_species.eq(pred_species)] = 0
                    taxonomy_distance_sum += distance.sum()
                genus_prediction = output["genus_logits"].argmax(dim=1)
                genus_correct += genus_prediction[genus_mask].eq(
                    batch["genus_target"][genus_mask]
                ).sum()
                genus_samples += genus_mask.sum()
                family_prediction = output["family_probabilities"].argmax(dim=1)
                family_correct += family_prediction[family_mask].eq(
                    batch["family_target"][family_mask]
                ).sum()
                family_samples += family_mask.sum()
                if bool(getattr(self.params, "identifiability_enabled", False)):
                    rank_prediction = output["rank_logits"].argmax(dim=1)
                    rank_correct += rank_prediction.eq(rank).sum()
                    rank_samples += batch_size
            else:
                predictions = logits.argmax(dim=1)
                targets = batch["species_target"]
                species_samples += batch_size
                top5_correct += _topk_correct(logits, targets, 5)
                indices = targets * self.num_classes + predictions
                confusion += torch.bincount(
                    indices,
                    minlength=self.num_classes * self.num_classes,
                ).reshape(self.num_classes, self.num_classes)

        tensors = [
            total_loss,
            total_samples,
            top5_correct,
            species_samples,
            genus_correct,
            genus_samples,
            family_correct,
            family_samples,
            rank_correct,
            rank_samples,
            taxonomy_distance_sum,
            confusion,
        ]
        for tensor in tensors:
            reduce_tensor(tensor)

        metrics = _metrics_from_confusion(confusion)
        metrics["loss"] = float((total_loss / total_samples.clamp_min(1.0)).item())
        metrics["top5"] = float((top5_correct / species_samples.clamp_min(1.0) * 100.0).item())
        metrics["genus_accuracy"] = float(
            (genus_correct / genus_samples.clamp_min(1.0) * 100.0).item()
        )
        metrics["family_accuracy"] = float(
            (family_correct / family_samples.clamp_min(1.0) * 100.0).item()
        )
        metrics["rank_accuracy"] = float(
            (rank_correct / rank_samples.clamp_min(1.0) * 100.0).item()
        ) if rank_samples.item() > 0 else 0.0
        metrics["mean_taxonomic_distance"] = float(
            (taxonomy_distance_sum / species_samples.clamp_min(1.0)).item()
        ) if self.hierarchical else 0.0
        ordered = OrderedDict(
            (key, round(metrics[key], 6))
            for key in [
                "loss",
                "top1",
                "top5",
                "balanced_accuracy",
                "macro_f1",
                "genus_accuracy",
                "family_accuracy",
                "rank_accuracy",
                "mean_taxonomic_distance",
            ]
        )
        if self.is_main_process():
            logger.info("추론(%s): %s", prefix, dict(ordered))
        return ordered

    @torch.no_grad()
    def _capture_best_student(self) -> None:
        """현재 학생 모델의 상태를 CPU 메모리에 독립적으로 복사한다."""
        student = _unwrap(self.model)

        self._best_student_state = OrderedDict(
            (
                name,
                value.detach().cpu().clone(),
            )
            for name, value in student.state_dict().items()
        )

    def _checkpoint_payload(self, epoch: int, metrics: Dict[str, float]) -> Dict:
        """재개 가능한 student·teacher·optimizer·scheduler 상태를 딕셔너리로 묶는다."""
        student = _unwrap(self.model)
        return {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "ema_state_dict": self.teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_value": self.best_value,
            "best_metrics": self.best_metrics,
            "current_metrics": dict(metrics),
            "selection_metric": self.selection_metric,
            "class_to_idx": dict(getattr(self.params, "class_to_idx", {})),
            "taxonomy": dict(getattr(self.params, "taxonomy", {})),
            "taxonomy_node": dict(getattr(self.params, "taxonomy_node", {})),
            "config": dict(vars(self.params)),
        }

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """store_ckp가 활성화된 경우에만 rank 0에서 checkpoint를 저장한다."""
        if not self.store_checkpoint:
            return

        if not self.is_main_process():
            return

        output_dir = Path(
            self.params.output_dir
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self._checkpoint_payload(
                epoch,
                metrics,
            ),
            output_dir / filename,
        )

    def _load_resume(self, path: Path) -> None:
        """중단 checkpoint에서 모델과 학습 상태를 복구한다."""
        if not path.is_file():
            raise FileNotFoundError(f"재개 체크포인트가 존재하지 않습니다: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        student = _unwrap(self.model)
        student.load_state_dict(checkpoint["model_state_dict"])
        self.teacher.load_state_dict(checkpoint.get("ema_state_dict", student.state_dict()))
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_value = float(checkpoint.get("best_value", float("-inf")))
        self.best_metrics = dict(checkpoint.get("best_metrics", {}))
        if self.is_main_process():
            logger.info("%s의 에폭 %d부터 학습을 재개했습니다", path, self.start_epoch)

    def _load_best_student(self) -> None:
        """메모리 또는 model.pt에서 검증 기준 최선 학생 가중치를 복원한다."""
        student = _unwrap(self.model)

        # 현재 실행에서 선택된 최선 모델이 있으면 메모리 상태를 우선 사용한다.
        if self._best_student_state is not None:
            student.load_state_dict(
                self._best_student_state
            )
            return

        # 재개 실행 등으로 메모리 상태가 없을 때만 기존 model.pt를 확인한다.
        path = (
            Path(self.params.output_dir)
            / "model.pt"
        )

        if not path.is_file():
            return

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        student.load_state_dict(
            checkpoint["model_state_dict"]
        )

    def train_classifier(self, train_loader, val_loader, test_loader):
        """validation으로 모델을 선택한 뒤 test를 정확히 한 번 평가한다."""
        if val_loader is None:
            raise RuntimeError(
                "체크포인트 선택과 조기 종료에는 검증 로더가 필요합니다. "
                "테스트 로더를 검증용으로 대신 사용하면 안 됩니다."
            )

        last_train_metrics = OrderedDict()
        last_val_metrics = OrderedDict()
        patience = int(getattr(self.params, "early_patience", 0))
        eval_freq = int(getattr(self.params, "eval_freq", 1))

        if eval_freq <= 0:
            raise ValueError("eval_freq는 양의 정수여야 합니다")

        for epoch in range(self.start_epoch, self.total_epoch):
            last_train_metrics = self.train_one_epoch(epoch, train_loader)
            self.scheduler.step(epoch + 1)
            should_evaluate = (epoch % eval_freq == 0) or (epoch == self.total_epoch - 1)
            if not should_evaluate:
                self._save_checkpoint("last.pt", epoch + 1, {})
                continue
            # checkpoint 선택과 early stopping에는 validation set만 사용한다.
            last_val_metrics = self.eval_classifier(
                val_loader,
                "val",
            )
            current_value = float(last_val_metrics[self.selection_metric])
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_metrics = dict(last_val_metrics)
                self.patience_counter = 0

                # 디스크 저장 여부와 무관하게 현재 최선 학생 모델을 메모리에 보관한다.
                self._capture_best_student()

                # store_ckp=True인 경우에만 model.pt가 실제로 저장된다.
                self._save_checkpoint(
                    "model.pt",
                    epoch + 1,
                    last_val_metrics,
                )
            else:
                self.patience_counter += 1
            self._save_checkpoint("last.pt", epoch + 1, last_val_metrics)
            if patience > 0 and self.patience_counter >= patience:
                if self.is_main_process():
                    logger.info("평가 %d회 후 조기 종료합니다", self.patience_counter)
                break

        # validation 기준으로 저장한 최선 모델을 불러온다.
        self._load_best_student()

        # test set은 모델 선택이 모두 끝난 후 정확히 한 번만 평가한다.
        if test_loader is not None:
            final_metrics = self.eval_classifier(
                test_loader,
                "test",
            )
        else:
            final_metrics = self.eval_classifier(
                val_loader,
                "val-final",
            )

        return (
            last_train_metrics,
            self.best_metrics,
            final_metrics,
        )

    def load_weight(self):
        """외부 checkpoint의 모델 가중치를 현재 모델에 로드한다."""
        self._load_best_student()

    @torch.no_grad()
    def collect_logits(self, loader):
        """정답 bbox를 사용하지 않고 loader 전체의 로짓과 정답을 수집한다."""
        self.model.eval()

        all_logits = []
        all_targets = []

        for raw_batch in loader:
            batch = _move_batch(
                raw_batch,
                self.device,
            )

            # 보정, 분석 및 외부 평가용 로짓에도
            # 정답 bbox 기반 patch prior를 사용하지 않는다.
            with self._autocast():
                output, _ = self._model_forward(
                    self.model,
                    batch["image"],
                    patch_prior=None,
                )

            logits = (
                output["species_logits"]
                if isinstance(output, dict)
                else output
            )

            all_logits.append(logits.cpu())
            all_targets.append(batch["species_target"].cpu())

        return (
            torch.cat(all_logits).numpy(),
            torch.cat(all_targets).numpy(),
        )
