from digital_nose.dataset import load_dataset
from digital_nose.model import train_model


def test_training_pipeline_runs(tmp_path):
    dataset = load_dataset()
    artifacts, metrics = train_model(dataset, test_size=0.3, random_state=0)
    assert len(artifacts.classes_) >= 3
    assert "classification_report" in metrics
