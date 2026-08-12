"""attention heatmap 생성, head 순위화, 원본 이미지 결합을 수행한다."""

import os
import torch
import warnings

import numpy as np
import cv2
import shutil

from PIL import Image


warnings.filterwarnings("ignore")

def combine_images(path, pred_class,resize_dim=(200,200)):
    """저장된 원본과 heatmap 이미지를 하나의 비교 이미지로 결합한다."""
    images = [os.path.join(path, image) for image in os.listdir(path) if image.endswith('.jpg')]
    images.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    imgs = [Image.open(image).resize(resize_dim) for image in images]

    widths, heights = zip(*(img.size for img in imgs))

    total_width = sum(widths)
    max_height = max(heights)
    merged_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in imgs:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width
    merged_image.save(path + "/" + "concatenated_prediction_"+str(pred_class)+".jpg")

    for image in images:
        #print(image)
        os.remove(image)

def SuperImposeHeatmap(
    attention,
    input_image,
    *,
    alpha: float = 0.5,
    eps: float = 1e-8,
):
    """비음수 CAM을 원본 이미지에 중첩한다.

    CAM은 softmax로 생성된 비음수 공간 질량이므로 최소값을 빼지 않는다.
    최대값으로만 정규화하여 미세한 노이즈가 전체 범위로 증폭되는 것을 막는다.
    """
    attention = np.asarray(
        attention,
        dtype=np.float32,
    )

    if attention.ndim != 2:
        raise ValueError(
            "어텐션은 2차원 맵이어야 합니다. "
            f"현재 형태={attention.shape}"
        )

    if not np.isfinite(attention).all():
        raise ValueError(
            "어텐션에 NaN 또는 Inf가 있습니다"
        )

    attention = np.clip(
        attention,
        a_min=0.0,
        a_max=None,
    )

    attention_resized = cv2.resize(
        attention,
        (
            input_image.shape[1],
            input_image.shape[0],
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    attention_resized = np.clip(
        attention_resized,
        a_min=0.0,
        a_max=None,
    )

    max_value = float(attention_resized.max())

    if max_value <= eps:
        attention_normalized = np.zeros_like(
            attention_resized,
            dtype=np.float32,
        )
    else:
        attention_normalized = (
            attention_resized / max_value
        )

    attention_normalized = cv2.GaussianBlur(
        attention_normalized,
        (9, 9),
        0,
    )

    attention_normalized = np.clip(
        attention_normalized,
        0.0,
        1.0,
    )

    heatmap = (
        attention_normalized * 255.0
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    input_bgr = cv2.cvtColor(
        input_image,
        cv2.COLOR_RGB2BGR,
    )

    result = cv2.addWeighted(
        input_bgr,
        alpha,
        heatmap,
        1.0 - alpha,
        0.0,
    )

    return result


def create_overlay_images(
    X,
    patch_size,
    attentions,
    output_folder,
):
    """원본 이미지와 최종 결합 CAM을 저장한다.

    attentions는 [B, K, P] 형식이다.
    계층 모델의 표준 출력에서는 K=1인 최종 species_cam을 받는다.
    """
    if attentions.ndim != 3:
        raise ValueError(
            "어텐션 형태는 [B, K, P]여야 합니다. "
            f"현재 형태={tuple(attentions.shape)}"
        )

    batch_size, map_count, patch_count = attentions.shape

    if batch_size != 1:
        raise ValueError(
            "시각화에는 선택 표본 하나가 필요합니다. "
            f"현재 배치 크기={batch_size}"
        )

    image_height = int(X.shape[-2])
    image_width = int(X.shape[-1])
    patch_size = int(patch_size)

    grid_height = image_height // patch_size
    grid_width = image_width // patch_size

    if grid_height * grid_width != patch_count:
        raise ValueError(
            "패치 수가 이미지 및 패치 크기와 일치하지 않습니다: "
            f"{grid_height}x{grid_width} != {patch_count}"
        )

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    image = (
        X[0]
        .detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    image_min = float(image.min())
    image_max = float(image.max())

    if image_max > image_min:
        image = (
            image - image_min
        ) / (
            image_max - image_min
        )
    else:
        image = np.zeros_like(image)

    image = np.clip(
        image * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            output_folder,
            "0.jpg",
        ),
        cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        ),
    )

    for map_index in range(map_count):
        attention_map = (
            attentions[0, map_index]
            .reshape(
                grid_height,
                grid_width,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        result_image = SuperImposeHeatmap(
            attention_map,
            image,
        )

        cv2.imwrite(
            os.path.join(
                output_folder,
                f"{map_index + 1}.jpg",
            ),
            result_image,
        )
