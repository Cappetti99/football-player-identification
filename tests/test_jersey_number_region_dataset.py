import pytest

from scripts.build_yolo_jersey_number_region_dataset import load_annotations


def test_number_region_requires_valid_normalized_box(tmp_path):
    image = tmp_path / "crop.jpg"
    image.write_bytes(b"placeholder")
    csv_path = tmp_path / "review.csv"
    csv_path.write_text(
        "audit_id,region_label,xmin,ymin,xmax,ymax,sequence,frame,crop_path\n"
        f'a,present,0.2,0.1,0.8,0.7,A,1,"{image}"\n'
    )
    rows, ignored = load_annotations(csv_path)
    assert rows[0]["box"] == (0.2, 0.1, 0.8, 0.7)
    assert not ignored

    csv_path.write_text(
        "audit_id,region_label,xmin,ymin,xmax,ymax,sequence,frame,crop_path\n"
        f'a,present,0.8,0.1,0.2,0.7,A,1,"{image}"\n'
    )
    with pytest.raises(ValueError):
        load_annotations(csv_path)
