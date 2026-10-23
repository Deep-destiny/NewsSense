import time
from src.research import train_and_save_nifty50_model

def execute_pipeline():
    print("=" * 60)
    print("  NEWSSENSE: NIFTY 50 MULTI-HORIZON RESEARCH MODELS")
    print("=" * 60)
    start_time = time.time()

    print("\n[1/3] Loading current NIFTY 50 constituents and maximum available adjusted prices...")
    print("[2/3] Training 1–10 year return and probability models with horizon-purged holdouts...")
    artifact = train_and_save_nifty50_model()

    elapsed = time.time() - start_time
    print(f"\n[3/3] Model saved in {elapsed:.2f}s.")
    print(f"      Available horizons: {artifact['supported_horizons']}")
    metrics = artifact["models_by_horizon"].get("10", {}).get("metrics")
    if metrics:
        print(f"      10-year return MAE: {metrics['return_mae']:.1%}")
        print(f"      10-year baseline MAE: {metrics['baseline_return_mae']:.1%}")
        print(f"      10-year probability Brier: {metrics['positive_return_brier']:.3f}")
        print(f"      10-year base-rate Brier: {metrics['baseline_positive_return_brier']:.3f}")
        loss_mae = metrics["conditional_loss_mae"]
        baseline_loss_mae = metrics["baseline_conditional_loss_mae"]
        if loss_mae is not None and baseline_loss_mae is not None:
            print(f"      10-year conditional-loss MAE: {loss_mae:.1%}")
            print(f"      10-year loss baseline MAE: {baseline_loss_mae:.1%}")
        print(f"      10-year model beats all baselines: {metrics['model_has_holdout_edge']}")
    else:
        print(f"      10-year model unavailable: {artifact['horizon_errors'].get('10', 'insufficient history')}")
    print()
    print("=" * 60)

if __name__ == "__main__":
    execute_pipeline()
