from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import mlflow

def compute_metrics(y, predictions):
    return {"accuracy": accuracy_score(y, predictions),
            "recall": recall_score(y, predictions),
            "precision": precision_score(y, predictions),
            "f1_score": f1_score(y, predictions)}

def plot_confusion_matrix_rocauc(y, predictions, prob_predictions, model_name):
    display_confusion_matrix = ConfusionMatrixDisplay.from_predictions(y, predictions, 
                                        display_labels=["Non-Exoplanet", "Exoplanet"])
    path_confusion = f"visuals/confusion_matrix_{model_name}.png"
    plt.savefig(path_confusion)
    mlflow.log_artifact(path_confusion, artifact_path='images')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    roc_disp = RocCurveDisplay.from_predictions(
        y, 
        prob_predictions, 
        name=f"{model_name}: Exoplanet", 
        color="#2c3e50", # Professional dark slate color
        lw=2,
        ax=ax
    )

    # Add a dashed line for "Random Guessing" (Chance level)
    ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Chance (AUC = 0.50)")

    # Customizing labels and title
    ax.set_title("ROC Curve for Exoplanet Detection", fontsize=15, pad=20)
    ax.set_xlabel("False Positive Rate (Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)

    # Add a grid for readability
    ax.grid(alpha=0.3, linestyle='--')

    # Place the legend in the bottom right (standard for ROC)
    ax.legend(loc="lower right", fontsize=10)
    path_rocauc = f"visuals/rocauc_curve_{model_name}.png"
    mlflow.log_artifact(path_rocauc, artifact_path='images')
    plt.savefig(path_rocauc)

    plt.close()