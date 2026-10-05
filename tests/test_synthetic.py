import yaml

from shelfsense.cli import build_parser
from shelfsense.synthetic import make_dataset


def test_dataset_layout_and_labels_are_valid_yolo(tmp_path):
    yaml_path = make_dataset(tmp_path / "ds", n_images=10, seed=3)
    data = yaml.safe_load(yaml_path.read_text())
    assert data["names"] == {0: "product", 1: "defect"}

    root = tmp_path / "ds"
    for split in ("train", "val"):
        images = sorted((root / "images" / split).glob("*.jpg"))
        labels = sorted((root / "labels" / split).glob("*.txt"))
        assert images and [i.stem for i in images] == [l.stem for l in labels]
        for label_file in labels:
            for line in label_file.read_text().strip().splitlines():
                cls, *box = line.split()
                assert int(cls) in (0, 1)
                assert all(0.0 <= float(v) <= 1.0 for v in box)


def test_dataset_is_deterministic(tmp_path):
    a = make_dataset(tmp_path / "a", n_images=4, seed=7)
    b = make_dataset(tmp_path / "b", n_images=4, seed=7)
    la = (a.parent / "labels" / "train" / "train_0000.txt").read_text()
    lb = (b.parent / "labels" / "train" / "train_0000.txt").read_text()
    assert la == lb


def test_cli_parser():
    args = build_parser().parse_args(["detect", "--source", "0", "--no-show"])
    assert args.command == "detect" and args.no_show
    args = build_parser().parse_args(["train", "--data", "d.yaml", "--epochs", "3"])
    assert args.epochs == 3
