"""공개 URL에서 고정 Wild30 ImageFolder 데이터셋을 복원한다.

저장소에는 제3자 이미지 파일 자체가 아니라 메타데이터, 체크섬과 고정된
학습/검증/테스트 배정만 보관한다. 이미 존재하는 정상 파일은 건너뛰므로 중단 후
안전하게 다시 실행할 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED_COLUMNS = {
    "download_url",
    "height",
    "relative_path",
    "sha256",
    "width",
}


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="고정된 Wild30 분할을 정확히 복원하고 모든 파일을 검증합니다."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=data_dir / "manifests" / "wild30_frozen.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_dir / "dataset" / "imagefolder",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--min-side", type=int, default=336)
    parser.add_argument("--delay-seconds", type=float, default=0.35)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--user-agent",
        default="PromptCAM-SnapMix-Wild30-reproducibility/1.0",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="이미지를 받지 않고 매니페스트만 검증합니다.",
    )
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"매니페스트를 찾을 수 없습니다: {args.manifest}")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality는 1에서 100 사이여야 합니다")
    if args.min_side <= 0 or args.delay_seconds < 0 or args.timeout_seconds <= 0:
        parser.error("크기와 제한 시간은 양수, 지연 시간은 0 이상이어야 합니다")
    return args


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"매니페스트에 필요한 열이 없습니다: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("매니페스트가 비어 있습니다")

    seen_paths: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        relative = PurePosixPath(row["relative_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 3
            or relative.parts[0] not in {"train", "val", "test"}
        ):
            raise ValueError(
                f"매니페스트 {line_number}행의 relative_path가 안전하지 않습니다: {relative}"
            )
        if row["relative_path"] in seen_paths:
            raise ValueError(f"relative_path가 중복되었습니다: {row['relative_path']}")
        seen_paths.add(row["relative_path"])
        if len(row["sha256"]) != 64:
            raise ValueError(f"매니페스트 {line_number}행의 SHA-256 값이 잘못되었습니다")

    return rows


def build_session(user_agent: str) -> Any:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def normalize_image(raw: bytes, min_side: int, quality: int) -> bytes:
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"이미지를 디코딩할 수 없습니다: {exc}") from exc

    if min(image.size) < min_side:
        raise ValueError(f"이미지가 너무 작습니다: {image.size}")

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=False)
    return output.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination_for(output_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    destination = output_root.joinpath(*relative.parts)
    resolved_root = output_root.resolve()
    resolved_destination = destination.resolve()
    if resolved_root not in resolved_destination.parents:
        raise ValueError(f"대상 경로가 출력 루트를 벗어납니다: {relative_path}")
    return destination


def restore(args: argparse.Namespace) -> None:
    rows = load_manifest(args.manifest)
    counts = {split: 0 for split in ("train", "val", "test")}
    for row in rows:
        counts[row["relative_path"].split("/", 1)[0]] += 1
    print(f"매니페스트: 이미지 {len(rows)}장({counts})")
    if args.validate_only:
        print("MANIFEST_VALIDATION=PASS")
        return

    session = build_session(args.user_agent)
    restored = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        destination = destination_for(args.output_root, row["relative_path"])
        expected_hash = row["sha256"].lower()
        if destination.is_file() and sha256_file(destination) == expected_hash:
            skipped += 1
            continue

        response = session.get(
            row["download_url"],
            timeout=(20.0, args.timeout_seconds),
        )
        response.raise_for_status()
        normalized = normalize_image(
            response.content,
            min_side=args.min_side,
            quality=args.jpeg_quality,
        )
        actual_hash = sha256_bytes(normalized)
        if actual_hash != expected_hash:
            raise RuntimeError(
                "체크섬이 일치하지 않습니다: "
                f"{row['relative_path']}: 예상 {expected_hash}, 실제 {actual_hash}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(normalized)
        os.replace(temporary, destination)
        restored += 1
        if index % 100 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] 복원={restored}, 건너뜀={skipped}")
        if args.delay_seconds:
            time.sleep(args.delay_seconds)

    print(f"DATASET_RESTORE=PASS 복원={restored} 건너뜀={skipped}")


if __name__ == "__main__":
    restore(parse_args())
