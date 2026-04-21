from src.model_evaluation import comparison_table, evaluate_all
from src.model_training import MODEL_REGISTRY, train_all_models


def test_trains_all_registered_models(prepared_dataset):
    trained = train_all_models(prepared_dataset)
    assert set(trained.keys()) == set(MODEL_REGISTRY.keys())


def test_evaluation_metrics_in_range(prepared_dataset):
    trained = train_all_models(prepared_dataset)
    results = evaluate_all(trained, prepared_dataset)
    for r in results.values():
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.f1 <= 1.0
        assert r.confusion.shape == (
            len(prepared_dataset.class_names),
            len(prepared_dataset.class_names),
        )


def test_comparison_table_sorted_by_f1(prepared_dataset):
    trained = train_all_models(prepared_dataset)
    results = evaluate_all(trained, prepared_dataset)
    table = comparison_table(results)
    f1s = table["F1 (macro)"].tolist()
    assert f1s == sorted(f1s, reverse=True)
