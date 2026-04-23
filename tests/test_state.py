import threading

from app.state import AppState


def test_state_lock_is_public():
    state = AppState()
    assert state.lock is state._lock


def test_update_after_train_clears_predictions_and_tuning():
    state = AppState()
    state.last_prediction_df = "previous"
    state.tuning_results = {"M": "stale"}
    state.update_after_train(trained={"X": 1}, results={"X": 2}, best_name="X")
    assert state.last_prediction_df is None
    assert state.tuning_results == {}
    assert state.best_model_name == "X"


def test_append_live_caps_buffer():
    state = AppState()
    for i in range(150):
        state.append_live({"t": i}, max_size=50)
    snap = state.live_snapshot()
    assert len(snap) == 50
    assert snap[0]["t"] == 100


def test_concurrent_mutations_do_not_raise():
    state = AppState()
    errors: list[BaseException] = []

    def writer():
        try:
            for i in range(200):
                state.update_model(f"M{i}", trained=i, result=i)
        except BaseException as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(200):
                _ = state.live_snapshot()
                _ = dict(state.trained_models)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
