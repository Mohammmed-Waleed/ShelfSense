import pytest

from shelfsense import AnalyticsConfig, Detection, ShelfAnalyzer


def det(x1, y1, x2, y2, label="product"):
    return Detection(x1, y1, x2, y2, conf=0.9, cls_id=0, label=label)


def row(xs, y=100, w=50, h=80, label="product"):
    return [det(x, y, x + w, y + h, label) for x in xs]


def test_counts_per_label():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1))
    report = a.analyze(row([0, 60, 120]) + [det(0, 0, 10, 10, "defect")])
    assert report.counts == {"defect": 1, "product": 3}


def test_defect_raises_alert():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1))
    report = a.analyze(row([0, 60, 120]) + [det(0, 0, 10, 10, "defect")])
    assert [x.kind for x in report.alerts] == ["defect"]
    assert len(report.defects) == 1


def test_defects_are_not_products_for_gap_detection():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1))
    report = a.analyze(row([0, 60]) + [det(120, 100, 170, 180, "defect")])
    assert report.gaps == []  # only two real products: row too sparse


def test_gap_found_between_products():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1))
    # products at x=0,60,120 then a hole (one missing slot) then 300
    report = a.analyze(row([0, 60, 120, 300]))
    assert len(report.gaps) == 1
    g = report.gaps[0]
    assert g.x1 == pytest.approx(170) and g.x2 == pytest.approx(300)
    assert any(x.kind == "gap" for x in report.alerts)


def test_tight_row_has_no_gap():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1))
    assert a.analyze(row([0, 55, 110, 165, 220])).gaps == []


def test_gaps_are_per_row():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1))
    top = row([0, 60, 120, 180], y=50)
    bottom = row([0, 60, 120, 300], y=250)  # gap only here
    report = a.analyze(top + bottom)
    assert len(report.gaps) == 1
    assert report.gaps[0].y1 >= 250


def test_low_stock_alert():
    cfg = AnalyticsConfig(low_stock={"product": 5}, smooth_frames=1)
    report = ShelfAnalyzer(cfg).analyze(row([0, 60, 120]))
    assert any(x.kind == "low_stock" and "3/5" in x.message for x in report.alerts)


def test_low_stock_for_label_not_seen_at_all():
    cfg = AnalyticsConfig(low_stock={"bottle": 2}, smooth_frames=1)
    report = ShelfAnalyzer(cfg).analyze(row([0, 60, 120]))
    assert any("bottle low: 0/2" in x.message for x in report.alerts)


def test_smoothing_ignores_single_frame_flicker():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=5))
    full = row([0, 55, 110, 165])
    for _ in range(4):
        a.analyze(full)
    flicker = a.analyze(full[:1])  # detector briefly loses 3 products
    assert flicker.counts["product"] == 4


def test_product_classes_filter():
    cfg = AnalyticsConfig(product_classes=["bottle"], smooth_frames=1)
    report = ShelfAnalyzer(cfg).analyze(row([0, 55, 110], label="can"))
    assert report.gaps == []


def test_empty_frame():
    report = ShelfAnalyzer().analyze([])
    assert report.counts == {} and report.alerts == []


def test_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("low_stock: {product: 7}\ngap_factor: 2.0\n")
    cfg = AnalyticsConfig.from_yaml(p)
    assert cfg.low_stock == {"product": 7} and cfg.gap_factor == 2.0


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("gapp_factor: 2.0\n")
    with pytest.raises(ValueError, match="gapp_factor"):
        AnalyticsConfig.from_yaml(p)


def test_shipped_config_loads():
    from pathlib import Path

    cfg = AnalyticsConfig.from_yaml(Path(__file__).parent.parent / "configs" / "shelfsense.yaml")
    assert cfg.defect_classes == ["defect"]
