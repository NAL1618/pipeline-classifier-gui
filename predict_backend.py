import numpy as np
import scipy.io
import pywt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Load TensorFlow model once at import
model = load_model("cnn_pipe.h5")

LABELS = [
    "No Damage",          # 0
    "Local Corrosion",    # 1
    "General Corrosion",  # 2
    "Clamp",              # 3
    "Weld",               # 4
    "Pitting"             # 5
]

# ------------------ Pre‑processing ------------------

def denoise_signal(raw_signal, wavelet_name="db8"):
    coeffs = pywt.wavedec(raw_signal, wavelet_name)
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(raw_signal)))
    den    = [pywt.threshold(c, thresh, mode="soft") for c in coeffs]
    return pywt.waverec(den, wavelet_name)[:len(raw_signal)]

def preprocess(X):
    """Denoise + z‑norm across each sample."""
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out[i, j, :] = denoise_signal(X[i, j, :])
    out = (out - np.mean(out, axis=(1, 2), keepdims=True)) / (
        np.std(out, axis=(1, 2), keepdims=True) + 1e-8
    )
    return out

# ------------------ Inference helpers ------------------

def predict_from_array(X):
    """Predict a *single* pipeline sample, X shape (1, 46, 1200)."""
    X = preprocess(X)
    preds = model.predict(X)
    idx   = int(np.argmax(preds[0]))
    return {
        "prediction_index": idx,
        "prediction_name": LABELS[idx]
    }


def predict(mat_filepath):
    """Legacy helper: load .mat file, predict first pipeline sample."""
    mat = scipy.io.loadmat(mat_filepath)
    X = mat["X_n"]
    y = mat.get("Y_n", None)

    true_label = int(y[0]) if y is not None else None

    X     = preprocess(X)
    preds = model.predict(X)
    idx   = int(np.argmax(preds[0]))

    return {
        "prediction_index": idx,
        "prediction_name": LABELS[idx],
        "true_label_index": true_label,
        "waveform": X[0]  # (46, 1200)
    }


def predict_all(X, y_true):
    """Batch inference for confusion‑matrix stats."""
    X      = preprocess(X)
    preds  = model.predict(X)
    y_pred = np.argmax(preds, axis=1)
    y_true = y_true.reshape(-1)

    cm  = confusion_matrix(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)

    return {
        "confusion_matrix": cm,
        "f1_score": f1,
        "accuracy": acc,
        "y_true": y_true,
        "y_pred": y_pred
    }

