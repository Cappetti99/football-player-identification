from PIL import Image

from scripts.audit_jersey_region_ctc_preprocessing import (
    PREPROCESSING,
    preprocess_region,
    variant_name,
)


def test_preprocessing_variants_are_rgb_and_upscaled():
    image = Image.new("RGB", (10, 20), (20, 100, 200))
    for method in PREPROCESSING:
        output = preprocess_region(image, method, upscale=4)
        assert output.mode == "RGB"
        assert output.size == (40, 80)


def test_variant_name_is_stable():
    assert variant_name(0.25, "clahe", 4) == "pad0.25_clahe_x4"
