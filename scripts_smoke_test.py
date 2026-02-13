from src.synthetic_sensor_generator import generate_sensor_data
from src.feature_engineering import add_derived_features
from src.sensor_fusion import fuse_sensors
from src.preprocessing import prepare_training_data
from src.model_training import train_all_models, save_model
from src.model_evaluation import evaluate_all, comparison_table, best_model_name
from src.explainability import compute_shap_values, global_feature_importance
from src.report_generator import build_prediction_records, summarize_predictions

print(">> generating data")
df = generate_sensor_data(samples_per_class=200)
print(f"   shape={df.shape}, classes={df['fault_type'].nunique()}")

print(">> feature engineering + sensor fusion")
df = add_derived_features(df)
fused = fuse_sensors(df)
print(f"   fused shape={fused.shape}")

print(">> preparing training data")
prepared = prepare_training_data(fused)
print(f"   X_train={prepared.X_train.shape}, classes={len(prepared.class_names)}")

print(">> training all models")
trained = train_all_models(prepared)
for name, tm in trained.items():
    print(f"   {name:<22} train={tm.train_seconds:.2f}s predict={tm.predict_seconds:.3f}s")

print(">> evaluating")
results = evaluate_all(trained, prepared)
print(comparison_table(results).to_string(index=False))
best = best_model_name(results)
print(f">> best: {best}")

print(">> saving best model")
saved = save_model(trained[best], prepared)
print(f"   saved to {saved}")

print(">> computing SHAP global importance (sample=40)")
shap_res = compute_shap_values(
    trained[best].model, prepared.X_test, prepared.feature_names, prepared.class_names, max_samples=40
)
imp = global_feature_importance(shap_res)
print(imp.head(8).to_string(index=False))

print(">> building prediction records")
records = build_prediction_records(
    trained[best].y_pred, prepared.class_names, trained[best].y_proba
)
print(records.head())
print(summarize_predictions(records))

print(">> SMOKE TEST OK")
