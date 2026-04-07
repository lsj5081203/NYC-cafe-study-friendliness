"""
Baseline models for urban sound classification.

MFCC features + SVM (RBF kernel) and Random Forest classifiers.
Uses UrbanSound8K's predefined 10-fold CV to avoid data leakage.
"""

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.dataset import get_fold_data, get_default_split, CLASS_NAMES
from src.audio_features import extract_mfcc_batch


def build_svm_pipeline():
    """Build an SVM pipeline with standard scaling.

    Uses an RBF kernel with C=10, matching the best configuration reported
    by Salamon et al. 2014 ("A Dataset and Taxonomy for Urban Sound
    Research") on UrbanSound8K.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ])


def build_rf_pipeline():
    """Build a Random Forest pipeline.

    No StandardScaler is included. Random Forest is scale-invariant (splits
    are based on rank order of feature values, not magnitude), so scaling
    would have no effect on model behavior and is intentionally omitted.
    """
    return Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )),
    ])


def train_baseline(X_train, y_train, model_type="rf"):
    """Train a baseline classifier.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features).
        y_train: Label array of shape (n_samples,).
        model_type: "svm" or "rf".

    Returns:
        Trained sklearn Pipeline.
    """
    if model_type == "svm":
        pipeline = build_svm_pipeline()
    elif model_type == "rf":
        pipeline = build_rf_pipeline()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_baseline(model, X_test, y_test):
    """Evaluate a trained baseline model.

    Args:
        model: Trained sklearn Pipeline.
        X_test: Feature matrix.
        y_test: True labels.

    Returns:
        dict with 'accuracy', 'report' (classification report string),
        'confusion_matrix', and 'predictions'.
    """
    y_pred = model.predict(X_test)
    labels = list(range(len(CLASS_NAMES)))
    target_names = [CLASS_NAMES[i] for i in labels]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, labels=labels, target_names=target_names, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "predictions": y_pred,
    }


def run_single_split(data_dir, model_type="rf", sr=22050, n_mfcc=40):
    """Run training and evaluation on the default train/val/test split.

    Uses folds 1-8 for training, fold 9 for validation (model selection),
    and fold 10 for final test evaluation (reported once, at the end).

    Args:
        data_dir: Path to UrbanSound8K root directory.
        model_type: "svm" or "rf".
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Trained model and evaluation results dict (evaluated on test fold).
    """
    split = get_default_split()

    print("Loading training data...")
    train_audios, y_train, _ = get_fold_data(data_dir, split["train"], sr=sr)
    print(f"  {len(train_audios)} training clips loaded.")

    print("Loading validation data...")
    val_audios, y_val, _ = get_fold_data(data_dir, split["val"], sr=sr)
    print(f"  {len(val_audios)} validation clips loaded.")

    print("Loading test data...")
    test_audios, y_test, _ = get_fold_data(data_dir, split["test"], sr=sr)
    print(f"  {len(test_audios)} test clips loaded.")

    print("Extracting MFCC features...")
    X_train = extract_mfcc_batch(train_audios, sr=sr, n_mfcc=n_mfcc)
    X_val = extract_mfcc_batch(val_audios, sr=sr, n_mfcc=n_mfcc)
    X_test = extract_mfcc_batch(test_audios, sr=sr, n_mfcc=n_mfcc)
    print(f"  Feature shape: {X_train.shape[1]} dimensions")

    print(f"Training {model_type.upper()} model...")
    model = train_baseline(X_train, y_train, model_type=model_type)

    val_results = evaluate_baseline(model, X_val, y_val)
    print(f"  Val accuracy: {val_results['accuracy']:.4f}")

    print("Evaluating on test set...")
    results = evaluate_baseline(model, X_test, y_test)
    print(f"  Test accuracy: {results['accuracy']:.4f}")
    print(results["report"])

    return model, results


def run_kfold_cv(data_dir, model_type="rf", sr=22050, n_mfcc=40):
    """Run full 10-fold cross-validation using UrbanSound8K's predefined folds.

    Trains a fresh model for each of the 10 folds (leaving one fold out as
    test each time) and reports per-fold and mean accuracy.

    Note: Trained models are NOT saved to disk. This function is for
    reporting cross-validated accuracy only. To obtain a model for
    inference, use run_single_split(), which returns the trained model.

    Args:
        data_dir: Path to UrbanSound8K root directory.
        model_type: "svm" or "rf".
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        List of per-fold accuracies and mean accuracy.
    """
    fold_accuracies = []

    for test_fold in range(1, 11):
        train_folds = [f for f in range(1, 11) if f != test_fold]

        train_audios, y_train, _ = get_fold_data(data_dir, train_folds, sr=sr)
        test_audios, y_test, _ = get_fold_data(data_dir, [test_fold], sr=sr)

        X_train = extract_mfcc_batch(train_audios, sr=sr, n_mfcc=n_mfcc)
        X_test = extract_mfcc_batch(test_audios, sr=sr, n_mfcc=n_mfcc)

        model = train_baseline(X_train, y_train, model_type=model_type)
        results = evaluate_baseline(model, X_test, y_test)

        fold_accuracies.append(results["accuracy"])
        print(f"Fold {test_fold:2d}: accuracy = {results['accuracy']:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies, ddof=1)
    print(f"\n10-Fold CV: {mean_acc:.4f} +/- {std_acc:.4f}")

    return fold_accuracies, mean_acc


def save_model(model, path):
    """Save a trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load a trained model from disk."""
    return joblib.load(path)
