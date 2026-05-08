"""
Baseline models for UrbanSound8K classification.

Uses MFCC features with SVM and Random Forest classifiers.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.audio_features import extract_mfcc_batch
from src.dataset import CLASS_NAMES, get_default_split, get_fold_data


def build_svm_pipeline():
    """Build the RBF-SVM baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ])


def build_rf_pipeline():
    """Build the Random Forest baseline."""
    return Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )),
    ])


def train_baseline(X_train, y_train, model_type="rf"):
    """Train either the Random Forest or SVM baseline."""
    if model_type == "svm":
        pipeline = build_svm_pipeline()
    elif model_type == "rf":
        pipeline = build_rf_pipeline()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_baseline(model, X_test, y_test):
    """Evaluate a trained baseline model."""
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
    """Train on folds 1-8, validate on fold 9, and test on fold 10."""
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
    """Run 10-fold cross-validation using UrbanSound8K's predefined folds."""
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
    """Save a trained sklearn model."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load a trained sklearn model."""
    return joblib.load(path)
