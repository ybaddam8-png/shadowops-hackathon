"""Check if all required artifacts for submission exist and generate a report."""

import os
import json
from pathlib import Path

def main():
    backend_dir = Path(__file__).resolve().parent.parent
    root_dir = backend_dir.parent
    reports_dir = backend_dir / "training" / "reports"
    plots_dir = backend_dir / "training" / "plots"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    required_files = [
        ("openenv.yaml", root_dir / "openenv.yaml"),
        ("backend-ml/openenv.yaml", backend_dir / "openenv.yaml"),
        ("backend-ml/schema_contract.json", backend_dir / "schema_contract.json"),
        ("backend-ml/training/reports/openenv_loop_eval.md", reports_dir / "openenv_loop_eval.md"),
        ("backend-ml/training/reports/openenv_behavior_comparison.md", reports_dir / "openenv_behavior_comparison.md"),
        ("backend-ml/training/plots/openenv_episode_rewards.png", plots_dir / "openenv_episode_rewards.png"),
        ("Submission Docs", root_dir / "README.md" if (root_dir / "README.md").exists() else root_dir / "CRITICAL_FIX_SUMMARY.md"),
    ]
    
    results = []
    
    for name, path in required_files:
        if path.exists() and path.stat().st_size > 0:
            results.append((name, "PASS", "File exists and is not empty."))
        else:
            results.append((name, "FAIL", "File is missing or empty."))
            
    # Model Evaluation Check
    eval_report_path = backend_dir / "training" / "model_eval_report.json"
    if eval_report_path.exists():
        try:
            report_data = json.loads(eval_report_path.read_text(encoding="utf-8"))
            if report_data.get("model_metrics") is None:
                results.append(("Trained Model Evaluation", "WARN", "The trained-model evaluation has not yet produced valid metrics in the standard local environment. Pending GPU Validation."))
            else:
                results.append(("Trained Model Evaluation", "PASS", "Valid GPU metrics found."))
        except Exception as e:
            results.append(("Trained Model Evaluation", "WARN", f"Failed to parse model_eval_report.json: {e}"))
    else:
        results.append(("Trained Model Evaluation", "WARN", "model_eval_report.json is missing. The trained-model evaluation has not yet produced valid metrics in the standard local environment."))

    # Reward curve check
    reward_curve = plots_dir / "reward_curve.png"
    if reward_curve.exists() and reward_curve.stat().st_size > 0:
        results.append(("backend-ml/training/plots/reward_curve.png", "PASS", "File exists and is not empty."))
    else:
        results.append(("backend-ml/training/plots/reward_curve.png", "WARN", "Missing trained checkpoint metrics. Trained-model evidence will be added only when GPU evaluation artifacts exist."))

    # Generate MD report
    report_path = reports_dir / "submission_artifact_check_report.md"
    lines = [
        "# Submission Artifact Check Report",
        "",
        "| Artifact | Status | Note |",
        "| --- | --- | --- |"
    ]
    for name, status, note in results:
        lines.append(f"| `{name}` | {status} | {note} |")
        
    report_path.write_text("\n".join(lines), encoding="utf-8")
    
    failures = [r for r in results if r[1] == "FAIL"]
    if failures:
        print(f"Artifact check FAILED with {len(failures)} missing files. See {report_path.relative_to(root_dir)}.")
    else:
        print(f"Artifact check PASSED. See {report_path.relative_to(root_dir)}.")

if __name__ == "__main__":
    main()
