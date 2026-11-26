from detectit.train.finetune import best_epoch_row


def test_best_epoch_matches_the_checkpoint_ultralytics_keeps():
    rows = [
        {"epoch": "1", "metrics/mAP50-95(B)": "0.41"},
        {"epoch": "2", "metrics/mAP50-95(B)": "0.52"},
        {"epoch": "3", "metrics/mAP50-95(B)": "0.52"},
        {"epoch": "4", "metrics/mAP50-95(B)": "0.47"},
    ]
    assert best_epoch_row(rows)["epoch"] == "2"
