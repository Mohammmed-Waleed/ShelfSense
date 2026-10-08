import pytest

from shelfsense import AnalyticsConfig, Detection, ShelfAnalyzer


def det(x1, y1, x2, y2, label="product"):
    return Detection(x1, y1, x2, y2, conf=0.9, cls_id=0, label=label)


def row(xs, y=100, w=50, h=80, label="product"):
    return [det(x, y, x + w, y + h, label) for x in xs]


def test_counts_per_label():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=1))
    report = a.analyze(row([0, 60, 120]) + [det(0, 0, 10, 10, "defect")])
    assert report.counts == {"defect": 1, "product": 3}


def test_defect_raises_alert():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=1))
    report = a.analyze(row([0, 60, 120]) + [det(0, 0, 10, 10, "defect")])
    assert [x.kind for x in report.alerts] == ["defect"]
    assert len(report.defects) == 1


def test_defective_item_still_occupies_its_slot():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=1))
    report = a.analyze(row([0, 55]) + [det(110, 100, 160, 180, "defect")] + row([165, 220]))
    assert report.gaps == []
    assert report.counts == {"defect": 1, "product": 4}


def test_gap_must_persist_before_it_is_reported():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=3))
    holed = row([0, 55, 165, 220])
    assert a.analyze(holed).gaps == []
    assert a.analyze(holed).gaps == []
    assert len(a.analyze(holed).gaps) == 1


def test_one_frame_detector_flicker_is_not_a_gap():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=3))
    full, holed = row([0, 55, 110, 165, 220]), row([0, 55, 165, 220])
    for frame in (full, full, holed, full, full, holed, full):
        assert a.analyze(frame).gaps == []


def test_gap_found_between_products():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=1))
    # products at x=0,60,120 then a hole (one missing slot) then 300
    report = a.analyze(row([0, 60, 120, 300]))
    assert len(report.gaps) == 1
    g = report.gaps[0]
    assert g.x1 == pytest.approx(170) and g.x2 == pytest.approx(300)
    assert any(x.kind == "gap" for x in report.alerts)


def test_single_missing_product_is_a_gap():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=1))
    # 50px products on a 55px pitch; the one at x=110 is missing -> 65px hole
    report = a.analyze(row([0, 55, 165, 220]))
    assert len(report.gaps) == 1
    assert report.gaps[0].x1 == pytest.approx(105) and report.gaps[0].x2 == pytest.approx(165)


def test_tight_row_has_no_gap():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=1))
    assert a.analyze(row([0, 55, 110, 165, 220])).gaps == []


def test_gaps_are_per_row():
    a = ShelfAnalyzer(AnalyticsConfig(smooth_frames=1, gap_persist_frames=1))
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
