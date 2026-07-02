from pathlib import Path
import re
import zipfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from IPython.display import Audio, display
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
SAMPLE_RATE = 16000
JNV_URL = "https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver3.zip"


def download_jnv(data_root="/content/data", url=JNV_URL):
    data_root = Path(data_root)
    zip_path = data_root / "jnv_corpus_ver3.zip"
    jnv_dir = data_root / "JNV"
    data_root.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        import urllib.request

        urllib.request.urlretrieve(url, zip_path)

    if not jnv_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_root)

    wav_paths = sorted(jnv_dir.glob("*/*.wav"))
    if len(wav_paths) != 420:
        raise RuntimeError(f"Expected 420 wav files, but found {len(wav_paths)}")
    return wav_paths, jnv_dir


def build_manifest(wav_paths, manifest_path=None):
    pattern = re.compile(
        r"^(?P<speaker_id>[FM]\d)_(?P<label>angry|disgust|fear|happy|sad|surprise)_(?P<utterance_id>\d+)_(?P<session>[RF])\.wav$"
    )
    rows = []
    for path in wav_paths:
        match = pattern.match(Path(path).name)
        if not match:
            raise ValueError(f"Unexpected filename: {path}")
        item = match.groupdict()
        item["path"] = str(path)
        rows.append(item)

    df = pd.DataFrame(rows)
    df = df[["path", "label", "speaker_id", "session", "utterance_id"]]
    if manifest_path is not None:
        df.to_csv(manifest_path, index=False)
    return df


def load_demo_audio(df, label="happy", sample_rate=SAMPLE_RATE):
    row = df[df["label"] == label].iloc[0]
    y, sr = librosa.load(row["path"], sr=sample_rate, mono=True)
    return row, y, sr


def plot_waveform(y, sr):
    time = np.arange(len(y)) / sr
    plt.figure(figsize=(11, 3))
    plt.plot(time, y, linewidth=0.8)
    plt.title("Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_spectrogram(y, sr):
    stft = librosa.stft(y, n_fft=1024, hop_length=160, win_length=400)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.figure(figsize=(11, 4))
    librosa.display.specshow(
        stft_db,
        sr=sr,
        hop_length=160,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Linear-frequency Spectrogram")
    plt.tight_layout()
    plt.show()


def compute_mel_db(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmax=8000,
    )
    return librosa.power_to_db(mel, ref=np.max)


def plot_mel_spectrogram(y, sr):
    mel_db = compute_mel_db(y, sr)
    plt.figure(figsize=(11, 4))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=160,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()
    return mel_db


def plot_log_mel_dct_frame(mel_db, n_mfcc=20):
    mfcc_from_log_mel = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)
    frame_index = mel_db.shape[1] // 2
    log_mel_frame = mel_db[:, frame_index]
    mfcc_frame = mfcc_from_log_mel[:, frame_index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(log_mel_frame, marker="o", markersize=3)
    axes[0].set_title("One frame of log-Mel spectrum")
    axes[0].set_xlabel("Mel filter index")
    axes[0].set_ylabel("Power [dB]")
    axes[1].stem(np.arange(1, len(mfcc_frame) + 1), mfcc_frame)
    axes[1].set_title("DCT coefficients: MFCC")
    axes[1].set_xlabel("MFCC coefficient index")
    axes[1].set_ylabel("Coefficient value")
    plt.tight_layout()
    plt.show()


def compute_mfcc(y, sr, n_mfcc=20):
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmax=8000,
    )


def plot_mfcc(y, sr, n_mfcc=20):
    mfcc = compute_mfcc(y, sr, n_mfcc=n_mfcc)

    # C0 is strongly related to the overall log-energy level and often dominates
    # the color scale. For teaching, show C1.. as row-wise normalized changes.
    mfcc_view = mfcc[1:]
    row_mean = mfcc_view.mean(axis=1, keepdims=True)
    row_std = mfcc_view.std(axis=1, keepdims=True)
    mfcc_z = (mfcc_view - row_mean) / np.maximum(row_std, 1e-6)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        gridspec_kw={"height_ratios": [3, 1.2]},
        sharex=True,
    )
    img = librosa.display.specshow(
        mfcc_z,
        sr=sr,
        hop_length=160,
        x_axis="time",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
        ax=axes[0],
    )
    axes[0].set_title("MFCC variation over time: C1-C19, row-wise z-score")
    axes[0].set_ylabel("MFCC coefficient")
    axes[0].set_yticks(np.arange(0, n_mfcc - 1, 3))
    axes[0].set_yticklabels([f"C{i}" for i in range(1, n_mfcc, 3)])
    fig.colorbar(img, ax=axes[0], label="within-coefficient z-score")

    times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=160)
    for coef_index in [1, 2, 3, 4]:
        axes[1].plot(times, mfcc_z[coef_index - 1], linewidth=1.1, label=f"C{coef_index}")
    axes[1].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Selected low-order MFCC trajectories")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("z-score")
    axes[1].legend(ncol=4, loc="upper right")
    plt.tight_layout()
    plt.show()
    return mfcc


def display_mfcc_summary(mfcc):
    summary = pd.DataFrame(
        {
            "coefficient": [f"MFCC{i + 1}" for i in range(mfcc.shape[0])],
            "mean": mfcc.mean(axis=1),
            "std": mfcc.std(axis=1),
        }
    )
    display(summary.head())
    summary_long = summary.melt(id_vars="coefficient", value_vars=["mean", "std"])
    plt.figure(figsize=(12, 4))
    sns.barplot(data=summary_long, x="coefficient", y="value", hue="variable")
    plt.xticks(rotation=45)
    plt.title("MFCC summary statistics used for visualization")
    plt.tight_layout()
    plt.show()
    return summary


def draw_waveform_with_pitch(path, pitch_floor=75, pitch_ceiling=600):
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    samples = snd.values[0]
    times = np.linspace(snd.xmin, snd.xmax, len(samples))
    pitch_times = pitch.xs()
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan

    fig, ax1 = plt.subplots(figsize=(11, 3.5))
    ax1.plot(times, samples, linewidth=0.7, color="gray")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform and Praat pitch track")

    ax2 = ax1.twinx()
    ax2.plot(pitch_times, f0, "o-", markersize=3, linewidth=1.2, color="tab:red", label="F0")
    ax2.set_ylabel("F0 [Hz]")
    ax2.set_ylim(pitch_floor, pitch_ceiling)
    plt.tight_layout()
    plt.show()
    return summarize_f0_array(f0)


def summarize_f0_array(f0):
    voiced_f0 = f0[np.isfinite(f0)]
    if len(voiced_f0) == 0:
        return {"F0_mean": np.nan, "F0_std": np.nan, "F0_min": np.nan, "F0_max": np.nan, "voiced_ratio": 0.0}
    return {
        "F0_mean": float(np.mean(voiced_f0)),
        "F0_std": float(np.std(voiced_f0)),
        "F0_min": float(np.min(voiced_f0)),
        "F0_max": float(np.max(voiced_f0)),
        "voiced_ratio": float(np.isfinite(f0).mean()),
    }


def summarize_pitch(path, pitch_floor=75, pitch_ceiling=600):
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan
    return summarize_f0_array(f0)


def summarize_pitch_by_label(df, random_state=RANDOM_STATE, n_per_label=10):
    pitch_df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(n_per_label, len(x)), random_state=random_state))
        .reset_index(drop=True)
    )
    rows = []
    for _, row in pitch_df.iterrows():
        item = {"label": row["label"], "speaker_id": row["speaker_id"], "path": row["path"]}
        item.update(summarize_pitch(row["path"]))
        rows.append(item)

    stats = pd.DataFrame(rows)
    display(stats.groupby("label")[["F0_mean", "F0_std", "voiced_ratio"]].mean().round(2))
    plt.figure(figsize=(9, 4))
    sns.boxplot(data=stats, x="label", y="F0_mean")
    sns.stripplot(data=stats, x="label", y="F0_mean", color="black", alpha=0.5)
    plt.title("Praat F0 mean by emotion")
    plt.ylabel("F0 mean [Hz]")
    plt.tight_layout()
    plt.show()
    return stats


def draw_spectrogram_with_formants(path, maximum_formant=5500, max_frequency=8000):
    snd = parselmouth.Sound(str(path))
    spectrogram = snd.to_spectrogram(window_length=0.005, maximum_frequency=max_frequency)
    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=maximum_formant)

    values_db = 10 * np.log10(spectrogram.values)
    plt.figure(figsize=(11, 4))
    plt.pcolormesh(
        spectrogram.x_grid(),
        spectrogram.y_grid(),
        values_db,
        cmap="gray_r",
        shading="auto",
        vmin=values_db.max() - 70,
        vmax=values_db.max(),
    )
    times = np.arange(snd.xmin, snd.xmax, 0.01)
    for formant_index, color in zip([1, 2, 3], ["tab:red", "tab:orange", "tab:blue"]):
        values = [formant.get_value_at_time(formant_index, t) for t in times]
        plt.plot(times, values, "o", markersize=2, color=color, label=f"F{formant_index}")

    plt.ylim(0, max_frequency)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def summarize_formants(path, maximum_formant=5500):
    snd = parselmouth.Sound(str(path))
    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=maximum_formant)
    times = np.arange(snd.xmin, snd.xmax, 0.01)
    summary = {}
    for formant_index in [1, 2, 3]:
        values = np.array([formant.get_value_at_time(formant_index, t) for t in times], dtype=float)
        values = values[np.isfinite(values)]
        summary[f"F{formant_index}_median"] = np.nan if len(values) == 0 else float(np.median(values))
    return summary


def summarize_formants_by_label(df, random_state=RANDOM_STATE, n_per_label=10):
    sample_df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(n_per_label, len(x)), random_state=random_state))
        .reset_index(drop=True)
    )
    rows = []
    for _, row in sample_df.iterrows():
        item = {"label": row["label"], "speaker_id": row["speaker_id"], "path": row["path"]}
        item.update(summarize_formants(row["path"]))
        rows.append(item)
    stats = pd.DataFrame(rows)
    display(stats.groupby("label")[["F1_median", "F2_median", "F3_median"]].mean().round(1))
    return stats


def extract_mfcc_stats(path, target_sr=SAMPLE_RATE, n_mfcc=20):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if len(y) < target_sr // 10:
        y = np.pad(y, (0, target_sr // 10 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.concatenate([mfcc, delta, delta2], axis=0)
    return np.concatenate([feats.mean(axis=1), feats.std(axis=1)])


def build_mfcc_stat_dataset(df, sample_rate=SAMPLE_RATE):
    X = np.vstack([extract_mfcc_stats(path, target_sr=sample_rate) for path in df["path"]]).astype(np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    groups = df["speaker_id"].to_numpy()
    return X, y, groups, label_encoder


def _central_pca_view(pca_df, x_col="PC1", y_col="PC2", q=0.02, y_zoom=0.55):
    x_low, x_high = pca_df[x_col].quantile([q, 1 - q])
    y_low, y_high = pca_df[y_col].quantile([q, 1 - q])
    view_df = pca_df[
        pca_df[x_col].between(x_low, x_high)
        & pca_df[y_col].between(y_low, y_high)
    ].copy()

    x_center = float((x_low + x_high) / 2)
    y_center = float((y_low + y_high) / 2)
    x_half = float((x_high - x_low) / 2) * 1.12
    y_half = float((y_high - y_low) / 2) * y_zoom
    y_half = max(y_half, 1e-6)

    limits = {
        "xlim": (x_center - x_half, x_center + x_half),
        "ylim": (y_center - y_half, y_center + y_half),
        "shown": len(view_df),
        "total": len(pca_df),
    }
    return view_df, limits


def _finish_pca_plot(title, limits):
    plt.title(f"{title}\ncentral view: {limits['shown']}/{limits['total']} samples")
    plt.xlim(*limits["xlim"])
    plt.ylim(*limits["ylim"])
    plt.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.axvline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.gca().set_aspect("auto")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_pca_mfcc(df, X, random_state=RANDOM_STATE):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = df.copy()
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]
    view_df, limits = _central_pca_view(pca_df)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=view_df, x="PC1", y="PC2", hue="label", style="speaker_id", s=85, alpha=0.88)
    title = (
        f"MFCC PCA: PC1 {pca.explained_variance_ratio_[0] * 100:.1f}%, "
        f"PC2 {pca.explained_variance_ratio_[1] * 100:.1f}%"
    )
    _finish_pca_plot(title, limits)
    return X_scaled, pca_df, pca


def plot_lda_mfcc(df, X_scaled, y, label_encoder):
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    lda_df = df.copy()
    lda_df["LD1"] = X_lda[:, 0]
    lda_df["LD2"] = X_lda[:, 1]
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=lda_df, x="LD1", y="LD2", hue="label", style="speaker_id", s=80, alpha=0.85)
    plt.title("MFCC LDA: supervised axes using emotion labels")
    plt.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.axvline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return lda_df, lda


def plot_pca_by_speaker(pca_df):
    view_df, limits = _central_pca_view(pca_df)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=view_df, x="PC1", y="PC2", hue="speaker_id", style="label", s=85, alpha=0.88)
    _finish_pca_plot("MFCC PCA colored by speaker", limits)


def make_split(df, y, groups, mode="random", random_state=RANDOM_STATE):
    if mode == "speaker_holdout":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, y, groups=groups))
    elif mode == "random":
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=random_state,
            stratify=y,
        )
    else:
        raise ValueError(f"Unknown EVAL_MODE: {mode}")
    return train_idx, test_idx


def display_split_summary(df, y, groups, label_encoder, train_idx, test_idx, mode):
    print("evaluation mode:", mode)
    print("train speakers:", sorted(set(groups[train_idx])))
    print("test speakers:", sorted(set(groups[test_idx])))
    split_df = pd.DataFrame({"split": "train", "label": label_encoder.inverse_transform(y[train_idx])})
    split_df = pd.concat(
        [
            split_df,
            pd.DataFrame({"split": "test", "label": label_encoder.inverse_transform(y[test_idx])}),
        ],
        ignore_index=True,
    )
    display(pd.crosstab(split_df["label"], split_df["split"]))


def extract_mfcc_sequence(path, target_sr=SAMPLE_RATE, n_mfcc=20):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if len(y) < target_sr // 10:
        y = np.pad(y, (0, target_sr // 10 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features.T.astype(np.float32)


def build_sequence_dataset(df, train_idx, y, random_state=RANDOM_STATE, sample_rate=SAMPLE_RATE):
    sequences = [extract_mfcc_sequence(path, target_sr=sample_rate) for path in df["path"]]
    seq_lengths = np.array([seq.shape[0] for seq in sequences], dtype=np.int64)
    max_frames = int(seq_lengths.max())
    n_features = sequences[0].shape[1]

    frame_scaler = StandardScaler()
    frame_scaler.fit(np.concatenate([sequences[i] for i in train_idx], axis=0))

    X_seq = np.zeros((len(sequences), max_frames, n_features), dtype=np.float32)
    for i, seq in enumerate(sequences):
        scaled = frame_scaler.transform(seq)
        X_seq[i, : seq.shape[0], :] = scaled

    inner_train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=random_state,
        stratify=y[train_idx],
    )
    print("number of utterances:", len(sequences))
    print("frame length: min / median / max =", seq_lengths.min(), int(np.median(seq_lengths)), seq_lengths.max())
    print("feature dimension per frame:", n_features)
    print("train:", len(inner_train_idx), "validation:", len(val_idx))
    return X_seq, seq_lengths, n_features, inner_train_idx, val_idx, frame_scaler


class MFCCLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=64, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def masked_mean_max_pooling(self, outputs, lengths):
        frame_ids = torch.arange(outputs.size(1), device=outputs.device).unsqueeze(0)
        mask = frame_ids < lengths.unsqueeze(1)
        mask_float = mask.unsqueeze(-1).to(outputs.dtype)
        mean_pool = (outputs * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        max_pool = outputs.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=x.size(1))
        return self.classifier(self.masked_mean_max_pooling(outputs, lengths))


def make_loader(X_seq, seq_lengths, y, indices, batch_size=32, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X_seq[indices], dtype=torch.float32),
        torch.tensor(seq_lengths[indices], dtype=torch.long),
        torch.tensor(y[indices], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_torch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, lengths, yb in loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            yb = yb.to(device)
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total


def train_lstm(X_seq, seq_lengths, y, train_idx, val_idx, label_encoder, random_state=RANDOM_STATE):
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_seq.shape[-1]
    model = MFCCLSTM(n_features=n_features, n_classes=len(label_encoder.classes_)).to(device)
    print("device:", device)
    print(model)

    train_loader = make_loader(X_seq, seq_lengths, y, train_idx, shuffle=True)
    val_loader = make_loader(X_seq, seq_lengths, y, val_idx)
    class_counts = np.bincount(y[train_idx], minlength=len(label_encoder.classes_))
    class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
    display(pd.DataFrame({"label": label_encoder.classes_, "train_count": class_counts, "loss_weight": class_weights}))

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    wait = 0
    history = []

    for epoch in range(1, 121):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(yb)
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += len(yb)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate_torch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "loss": train_loss, "accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    display(history_df.tail())
    plt.figure(figsize=(10, 4))
    plt.plot(history_df["epoch"], history_df["loss"], label="train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return model, history_df, device


def evaluate_lstm(model, X_seq, seq_lengths, y, train_idx, test_idx, label_encoder, device):
    test_loader = make_loader(X_seq, seq_lengths, y, test_idx)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for xb, lengths, _ in test_loader:
            logits = model(xb.to(device), lengths.to(device))
            all_logits.append(logits.cpu())

    pred = torch.cat(all_logits, dim=0).argmax(dim=1).numpy()
    majority_class = np.bincount(y[train_idx], minlength=len(label_encoder.classes_)).argmax()
    majority_pred = np.full_like(y[test_idx], majority_class)
    result = pd.DataFrame(
        [
            {
                "model": "majority baseline",
                "accuracy": accuracy_score(y[test_idx], majority_pred),
                "macro_f1": f1_score(y[test_idx], majority_pred, average="macro", zero_division=0),
            },
            {
                "model": "LSTM on MFCC sequence",
                "accuracy": accuracy_score(y[test_idx], pred),
                "macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
            },
        ]
    )
    display(result.round(3))

    prediction_summary = pd.DataFrame(
        {
            "label": label_encoder.classes_,
            "test_count": np.bincount(y[test_idx], minlength=len(label_encoder.classes_)),
            "predicted_count": np.bincount(pred, minlength=len(label_encoder.classes_)),
        }
    )
    display(prediction_summary)
    if np.count_nonzero(prediction_summary["predicted_count"].to_numpy()) <= 2:
        print("WARNING: predictions are concentrated in one or two labels. This model should not be interpreted as having learned emotion categories.")

    print(classification_report(y[test_idx], pred, target_names=label_encoder.classes_, zero_division=0))
    cm = confusion_matrix(y[test_idx], pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix: LSTM on MFCC sequence")
    plt.tight_layout()
    plt.show()
    return pred, result, prediction_summary, cm
