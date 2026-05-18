from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score, \
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import mlflow
import os

def compute_metrics(y, predictions, custom_threshold=False):
    if not custom_threshold:
        return {"accuracy": accuracy_score(y, predictions),
            "recall": recall_score(y, predictions),
            "precision": precision_score(y, predictions),
            "f1_score": f1_score(y, predictions)}
    else:
        return {"accuracy_threshold": accuracy_score(y, predictions),
            "recall_threshold": recall_score(y, predictions),
            "precision_threshold": precision_score(y, predictions),
            "f1_score_threshold": f1_score(y, predictions)}

def plot_classification_curve(display_class, y_true, y_scores, model_name, title, xlabel, ylabel, save_path,
                              curve_name="Exoplanet", color="#2c3e50", legend_loc="lower right",
                              add_baseline=False, baseline_value=None, baseline_label=None):
    """
    Generic plotting function for ROC and Precision-Recall curves.
    """
    # Ensuring visuals folder exists, else it is created.
    os.makedirs("visuals", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    display_class.from_predictions(
        y_true,
        y_scores,
        name=f"{model_name}: {curve_name}",
        color=color,
        lw=2,
        ax=ax,
    )

    # Optional baseline / chance line
    if add_baseline and baseline_value is not None:
        if display_class == RocCurveDisplay:
            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="red",
                label=baseline_label,
            )

        elif display_class == PrecisionRecallDisplay:
            ax.axhline(
                y=baseline_value,
                linestyle="--",
                color="red",
                label=baseline_label,
            )

    ax.set_title(title, fontsize=15, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc=legend_loc, fontsize=10)

    # Save figure
    plt.savefig(save_path, bbox_inches="tight")
    mlflow.log_artifact(save_path, artifact_path='images')

    return fig, ax

def plot_metrics(y, predictions, prob_predictions, model_name):
    display_confusion_matrix = ConfusionMatrixDisplay.from_predictions(y, predictions, 
                                        display_labels=["Non-Exoplanet", "Exoplanet"])
    path_confusion = f"visuals/confusion_matrix_{model_name}.png"
    plt.savefig(path_confusion)
    mlflow.log_artifact(path_confusion, artifact_path='images')
    
    plot_classification_curve(display_class=RocCurveDisplay, y_true=y, y_scores=prob_predictions,
                              model_name=model_name, title="ROC Curve for Exoplanet Detection", xlabel="False Positive Rate", 
                              ylabel="True Positive Rate", save_path=f"visuals/rocauc_curve_{model_name}.png", 
                              legend_loc="lower right", add_baseline=True, baseline_value=0.5, 
                              baseline_label="Chance (AUC = 0.50)")
    
    baseline_precision = sum(y) / len(y)

    plot_classification_curve(display_class=PrecisionRecallDisplay, y_true=y, y_scores=prob_predictions,
                              model_name=model_name, title="Precision-Recall Curve for Exoplanet Detection", xlabel="Recall",
                              ylabel="Precision", save_path=f"visuals/pr_curve_{model_name}.png", legend_loc="lower left",
                              add_baseline=True, baseline_value=baseline_precision,
                              baseline_label=f"Baseline Precision = {baseline_precision:.2f}",
    )

    plt.close()