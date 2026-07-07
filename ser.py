"""Japanese SER notebook helper functions.

このファイルは，Colabノートブックを短く保つために，長い処理を関数として
まとめたものである。授業で読むときは，上から順にノートブックの節と対応している
と考えればよい。

大まかな流れ:
1. JNVコーパスをダウンロードし，ファイル一覧表を作る。
2. 波形，スペクトログラム，MFCC，F0，フォルマントを可視化する。
3. PCA/LDAで固定長MFCC特徴量を2次元に可視化する。
4. MFCC時系列をPyTorchのLSTMに入力して，感情分類を試す。

使っている主なライブラリ:
- librosa: 音声ファイルの読み込み，スペクトログラム，メルスペクトログラム，
  MFCC，ΔMFCCを計算するための音声処理ライブラリ。
- parselmouth: Praatの機能をPythonから使うためのライブラリ。F0やフォルマントの
  推定に使う。
- numpy / pandas: 数値配列と表形式データを扱うための基本ライブラリ。
- matplotlib / seaborn: 波形，スペクトログラム，散布図，混同行列などを描くための
  可視化ライブラリ。
- japanize_matplotlib: Matplotlibで日本語フォントを使えるようにするライブラリ。
- scikit-learn: PCA，LDA，データ分割，標準化，評価指標を使うための機械学習
  ライブラリ。
- PyTorch: LSTMモデルを作り，学習・評価するための深層学習ライブラリ。
"""

# 標準ライブラリ。ファイルパス，正規表現，zip展開に使う。
from pathlib import Path
import re
import zipfile

# 音声処理ライブラリ。
# librosaはwavの読み込み，STFT，メルスペクトログラム，MFCCの計算を担当する。
import librosa
import librosa.display

# 可視化と数値処理の基本ライブラリ。
# numpyは配列計算，pandasは表，matplotlib/seabornは図の描画に使う。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import japanize_matplotlib  # noqa: F401  # Matplotlibの日本語表示を有効にする。
except ImportError:
    pass

# PraatをPythonから呼び出すためのライブラリ。
# F0（ピッチ）やフォルマントの推定で使う。
import parselmouth
import seaborn as sns

# 深層学習ライブラリ。
# torchはテンソル計算，nnはニューラルネットワーク部品，
# DataLoaderはデータをミニバッチに分けて取り出すために使う。
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Colab/Jupyter上で音声再生ボタンや表を表示するための道具。
from IPython.display import Audio, clear_output, display

# scikit-learnの機械学習・評価用部品。
# PCA/LDAは可視化，metricsは評価指標，model_selectionはデータ分割，
# preprocessingはラベル変換と標準化に使う。
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
SAMPLE_RATE = 16000
JNV_URL = "https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver3.zip"


# ノートブック「JNV のダウンロードと展開」に対応する処理。
# Colab上にJNVのzipを保存し，展開済みwavファイルの一覧を返す。
def download_jnv(data_root="/content/data", url=JNV_URL):
    """JNVコーパスをダウンロードして展開する。

    引数:
        data_root: Colab上でデータを置くディレクトリ。
        url: JNVコーパスのzipファイルURL。

    返り値:
        wav_paths: 展開後に見つかったwavファイルのパス一覧。
        jnv_dir: JNVディレクトリのパス。

    学生向けメモ:
        この関数は「データを手元に用意する」だけで，音声認識はまだ行わない。
    """
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


# ノートブック「manifest の生成」に対応する処理。
# JNVのファイル名から話者ID，感情ラベル，発話番号，セッションを取り出す。
def build_manifest(wav_paths, manifest_path=None):
    """wavファイル一覧から，機械学習で使いやすい表を作る。

    JNVのファイル名には，話者ID，感情ラベル，発話番号，セッション種別が
    埋め込まれている。この関数は，それらを取り出してpandasのDataFrameにする。

    例:
        F1_angry_00_F.wav -> speaker_id=F1, label=angry, utterance_id=00, session=F
    """
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


# ノートブックのデモ表示用に，指定した感情ラベルの先頭サンプルを読み込む。
def load_demo_audio(df, label="happy", sample_rate=SAMPLE_RATE):
    """指定した感情ラベルの音声を1つ読み込む。

    ノートブックでは，波形やスペクトログラムを説明するための代表例として使う。
    返り値の y は音声波形そのもの，sr はサンプリング周波数である。
    """
    row = df[df["label"] == label].iloc[0]
    y, sr = librosa.load(row["path"], sr=sample_rate, mono=True)
    return row, y, sr


# ここから「音声波形・スペクトログラム・MFCC」に対応する可視化関数。
# 波形，周波数成分，メル尺度，MFCCの順に，音声の表現を段階的に確認する。
def plot_waveform(y, sr):
    """音声波形を描く。

    横軸は時間，縦軸は振幅である。音声を最も素朴に表示した図なので，
    最初に「音は時間方向に並んだ数値列である」ことを確認するために使う。
    """
    time = np.arange(len(y)) / sr
    plt.figure(figsize=(11, 3))
    plt.plot(time, y, linewidth=0.8)
    plt.title("波形")
    plt.xlabel("時間 [秒]")
    plt.ylabel("振幅")
    plt.tight_layout()
    plt.show()


def plot_spectrogram(y, sr):
    """線形周波数スペクトログラムを描く。

    波形だけでは分かりにくい「どの時刻に，どの周波数成分が強いか」を
    色で表示する。後で見るメルスペクトログラムやMFCCの出発点になる。
    """
    # STFTは，音声を短い時間窓に分け，各時刻の周波数成分を求める処理。
    # ノートブックでは「スペクトログラム」の定義に対応する。
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
    plt.title("線形周波数スペクトログラム")
    plt.tight_layout()
    plt.show()


def plot_fft_sine_demo(sample_rate=8000, duration=0.05):
    """人工的な正弦波を使って，FFTの意味を可視化する。

    2つの周波数の正弦波を足し合わせると，時間波形だけでは成分が見えにくい。
    FFTを使うと，どの周波数が含まれているかがピークとして現れる。
    ノートブックでは「波形を周波数成分に分ける」という直観を作るために使う。
    """
    t = np.arange(int(sample_rate * duration)) / sample_rate
    low_freq = 300
    high_freq = 900
    signal = np.sin(2 * np.pi * low_freq * t) + 0.6 * np.sin(2 * np.pi * high_freq * t)

    freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(t * 1000, signal, linewidth=1.2)
    axes[0].set_title("人工波形: 300 Hz + 900 Hz")
    axes[0].set_xlabel("時間 [ミリ秒]")
    axes[0].set_ylabel("振幅")

    axes[1].plot(freqs, spectrum, linewidth=1.2)
    axes[1].set_xlim(0, 1500)
    axes[1].set_title("FFTの振幅スペクトル")
    axes[1].set_xlabel("周波数 [Hz]")
    axes[1].set_ylabel("振幅")
    axes[1].axvline(low_freq, color="tab:red", linestyle="--", alpha=0.7)
    axes[1].axvline(high_freq, color="tab:red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def _dct_type2_ortho_matrix(n):
    """NumPyだけでDCT-IIの直交行列を作る。

    scipyを追加依存にしないため，教材用の小さなDCTは行列積として計算する。
    librosaのMFCCで使われるDCT-IIの直交正規化と同じ考え方である。
    """
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    basis = np.cos(np.pi * (i + 0.5) * k / n)
    basis[0, :] *= np.sqrt(1 / n)
    basis[1:, :] *= np.sqrt(2 / n)
    return basis


def plot_stft_spectrogram_demo(sample_rate=4000):
    """周波数が時間とともに変わる人工信号で，短時間FFTを説明する。

    音声全体に1回だけFFTをかけると「どの周波数が含まれるか」は分かるが，
    「いつ含まれたか」は分からない。短い時間窓をずらしながらFFTを繰り返すと，
    時間と周波数の両方を持つスペクトログラムになる。
    """
    segment_duration = 0.45
    freqs_by_segment = [300, 700, 1100]
    signal_parts = []
    for freq in freqs_by_segment:
        t_seg = np.arange(int(sample_rate * segment_duration)) / sample_rate
        signal_parts.append(np.sin(2 * np.pi * freq * t_seg))
    signal = np.concatenate(signal_parts)
    t = np.arange(len(signal)) / sample_rate

    fft_freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
    whole_spectrum = np.abs(np.fft.rfft(signal))
    stft = librosa.stft(signal, n_fft=512, hop_length=64, win_length=256)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 1, 1.5]})
    axes[0].plot(t, signal, linewidth=0.8)
    axes[0].set_title("人工信号: 時間とともに周波数が変化")
    axes[0].set_xlabel("時間 [秒]")
    axes[0].set_ylabel("振幅")

    axes[1].plot(fft_freqs, whole_spectrum, linewidth=1.1)
    axes[1].set_xlim(0, 1500)
    axes[1].set_title("全体に1回だけFFT: 時間順序は失われる")
    axes[1].set_xlabel("周波数 [Hz]")
    axes[1].set_ylabel("振幅")

    img = librosa.display.specshow(
        stft_db,
        sr=sample_rate,
        hop_length=64,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=axes[2],
    )
    axes[2].set_ylim(0, 1500)
    axes[2].set_title("短時間FFT: 時間と周波数を保ったスペクトログラム")
    fig.colorbar(img, ax=axes[2], format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


def hz_to_mel(frequencies):
    """Hzをメル尺度に変換する。

    メル尺度では，低い周波数ではHzの違いが大きく反映され，高い周波数では
    同じHz差でも相対的に小さく扱われる。
    """
    frequencies = np.asarray(frequencies, dtype=float)
    return 2595 * np.log10(1 + frequencies / 700)


def _make_tone_pair(first_freq, second_freq, sample_rate=16000, duration=0.55, gap=0.18):
    """2つの純音を短い無音を挟んで並べる。"""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    fade_len = max(1, int(sample_rate * 0.02))
    fade = np.ones_like(t)
    fade[:fade_len] = np.linspace(0, 1, fade_len)
    fade[-fade_len:] = np.linspace(1, 0, fade_len)
    first = 0.35 * np.sin(2 * np.pi * first_freq * t) * fade
    second = 0.35 * np.sin(2 * np.pi * second_freq * t) * fade
    silence = np.zeros(int(sample_rate * gap))
    return np.concatenate([first, silence, second])


def interactive_mel_scale_demo(sample_rate=16000):
    """メル尺度の曲線と，同じHz差の低音・高音ペアを聴き比べる。"""
    import ipywidgets as widgets

    low_base = widgets.IntSlider(value=200, min=80, max=600, step=10, description="低音[Hz]")
    high_base = widgets.IntSlider(value=2000, min=800, max=5000, step=100, description="高音[Hz]")
    delta = widgets.IntSlider(value=50, min=20, max=120, step=10, description="差[Hz]")
    output = widgets.Output()

    def update(_=None):
        low_pair = (low_base.value, low_base.value + delta.value)
        high_pair = (high_base.value, high_base.value + delta.value)
        freq_axis = np.linspace(0, max(6000, high_pair[1] + 500), 600)
        mel_axis = hz_to_mel(freq_axis)
        low_mel = hz_to_mel(low_pair)
        high_mel = hz_to_mel(high_pair)

        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(freq_axis, mel_axis, color="black", linewidth=1.5)
            ax.plot(low_pair, low_mel, "o-", color="tab:blue", label="低音側")
            ax.plot(high_pair, high_mel, "o-", color="tab:red", label="高音側")
            ax.set_title("Hzとメル尺度の対応")
            ax.set_xlabel("周波数 [Hz]")
            ax.set_ylabel("メル尺度")
            ax.legend()
            plt.tight_layout()
            plt.show()

            display(
                pd.DataFrame(
                    [
                        {
                            "音域": "低音側",
                            "周波数1 [Hz]": low_pair[0],
                            "周波数2 [Hz]": low_pair[1],
                            "Hz差": delta.value,
                            "メル差": low_mel[1] - low_mel[0],
                        },
                        {
                            "音域": "高音側",
                            "周波数1 [Hz]": high_pair[0],
                            "周波数2 [Hz]": high_pair[1],
                            "Hz差": delta.value,
                            "メル差": high_mel[1] - high_mel[0],
                        },
                    ]
                ).round(2)
            )
            print("低音側")
            display(Audio(_make_tone_pair(*low_pair, sample_rate=sample_rate), rate=sample_rate))
            print("高音側")
            display(Audio(_make_tone_pair(*high_pair, sample_rate=sample_rate), rate=sample_rate))

    for widget in (low_base, high_base, delta):
        widget.observe(update, names="value")
    display(widgets.VBox([low_base, high_base, delta]), output)
    update()


def compute_mel_db(y, sr):
    """音声から対数メルスペクトログラムを計算する。

    メル尺度は，人間の聴覚に近い周波数の並べ方である。
    dB表示にすることで，音の強さの大きな差を圧縮して見やすくする。
    """
    # メルスペクトログラムは，人間の聴覚に近い周波数尺度で表したスペクトログラム。
    # power_to_dbで対数スケールに変換し，値の広い範囲を扱いやすくする。
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
    """対数メルスペクトログラムを描き，計算結果も返す。

    返された mel_db は，次のDCT説明図で再利用する。
    """
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
    plt.title("メルスペクトログラム")
    plt.tight_layout()
    plt.show()
    return mel_db


def plot_log_mel_dct_frame(mel_db, n_mfcc=20):
    """1時刻分の対数メルスペクトルと，DCT後のMFCCを並べて描く。

    初学者には，ここを「横に並んだメルフィルタの値を，DCTで別の座標に
    写している」と読むとよい。音声全体ではなく，中央付近の1フレームだけを
    取り出して説明用の図にしている。
    """
    # ノートブックの「対数メルスペクトルにDCTをかける」説明に対応する図。
    # 左はDCT前の1フレーム，右はDCT後のMFCC係数を示す。
    mfcc_from_log_mel = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)
    frame_index = mel_db.shape[1] // 2
    log_mel_frame = mel_db[:, frame_index]
    mfcc_frame = mfcc_from_log_mel[:, frame_index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(log_mel_frame, marker="o", markersize=3)
    axes[0].set_title("対数メルスペクトルの1フレーム")
    axes[0].set_xlabel("メルフィルタ番号")
    axes[0].set_ylabel("パワー [dB]")
    axes[1].stem(np.arange(1, len(mfcc_frame) + 1), mfcc_frame)
    axes[1].set_title("DCT係数: MFCC")
    axes[1].set_xlabel("MFCC係数番号")
    axes[1].set_ylabel("係数値")
    plt.tight_layout()
    plt.show()

def plot_dct_reconstruction_demo(mel_db, keep_coeffs=(4, 8, 20)):
    """対数メルスペクトルをDCTし，低次係数だけで再構成する様子を描く。

    DCT係数の低次成分だけを残して逆DCTすると，細かな山谷が減り，
    なめらかなスペクトル包絡に近い曲線になる。これは「低次MFCCが
    スペクトルの大まかな形を表しやすい」という説明に対応する。
    """
    frame_index = mel_db.shape[1] // 2
    log_mel_frame = mel_db[:, frame_index]
    dct_matrix = _dct_type2_ortho_matrix(len(log_mel_frame))
    coeffs = dct_matrix @ log_mel_frame

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    mel_bins = np.arange(len(log_mel_frame))
    axes[0].plot(mel_bins, log_mel_frame, color="black", linewidth=1.5, label="元の対数メルスペクトル")
    for keep in keep_coeffs:
        kept = np.zeros_like(coeffs)
        kept[:keep] = coeffs[:keep]
        reconstructed = dct_matrix.T @ kept
        axes[0].plot(mel_bins, reconstructed, linewidth=1.2, label=f"低次 {keep} 個で再構成")
    axes[0].set_title("低次DCT係数だけで滑らかな包絡に近づく")
    axes[0].set_xlabel("メルフィルタ番号")
    axes[0].set_ylabel("パワー [dB]")
    axes[0].legend()

    show_n = min(30, len(coeffs))
    markerline, stemlines, baseline = axes[1].stem(np.arange(show_n), coeffs[:show_n])
    markerline.set_markersize(4)
    stemlines.set_linewidth(1.0)
    baseline.set_linewidth(0.8)
    axes[1].axvspan(-0.5, keep_coeffs[0] - 0.5, color="tab:orange", alpha=0.18, label="低次係数")
    axes[1].set_title("対数メルスペクトル1フレームのDCT係数")
    axes[1].set_xlabel("DCT係数番号")
    axes[1].set_ylabel("係数値")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def compute_mfcc(y, sr, n_mfcc=20):
    """音声波形からMFCCを計算する。

    返り値は2次元配列で，行がMFCC係数，列が時間フレームである。
    つまり，1つの音声が「時間ごとに並んだ特徴量」に変換される。
    """
    # librosa.feature.mfcc は，メルフィルタバンク，対数化，DCTを内部で行う。
    # 返り値の形は「MFCC係数数 × 時間フレーム数」である。
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
    """MFCCの時間変化を，初学者が見やすい形で可視化する。

    MFCCは係数ごとに値の範囲が違うため，そのままヒートマップにすると
    色の差が分かりにくい。この関数ではC0を除き，各係数を標準化して，
    「その係数が発話中のどこで相対的に大きいか」を見せる。
    """
    mfcc = compute_mfcc(y, sr, n_mfcc=n_mfcc)

    # C0は対数メルスペクトルをDCTしたときの直流成分に相当し，
    # スペクトル全体の平均的な大きさを表す。
    # 音声のエネルギーに近い情報を含むが，時間波形のエネルギーそのものではない。
    # そのため値の範囲が大きくなりやすく，色の範囲を支配しやすい。
    # 教材としてはC1以降を係数ごとに標準化し，時間変化を見やすくする。
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
    axes[0].set_title("MFCCの時間変化: C1-C19を係数ごとに標準化")
    axes[0].set_ylabel("MFCC係数")
    axes[0].set_yticks(np.arange(0, n_mfcc - 1, 3))
    axes[0].set_yticklabels([f"C{i}" for i in range(1, n_mfcc, 3)])
    fig.colorbar(img, ax=axes[0], label="係数内zスコア")

    times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=160)
    for coef_index in [1, 2, 3, 4]:
        axes[1].plot(times, mfcc_z[coef_index - 1], linewidth=1.1, label=f"C{coef_index}")
    axes[1].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    axes[1].set_title("低次MFCC係数の時間変化")
    axes[1].set_xlabel("時間 [秒]")
    axes[1].set_ylabel("zスコア")
    axes[1].legend(ncol=4, loc="upper right")
    plt.tight_layout()
    plt.show()
    return mfcc


def _make_sample_dropdowns(df, prefix="", default_label="happy"):
    """JNVサンプルを選ぶためのドロップダウン群を作る。"""
    import ipywidgets as widgets

    label_options = sorted(df["label"].unique())
    speaker_options = sorted(df["speaker_id"].unique())
    session_options = sorted(df["session"].unique())
    utterance_options = sorted(df["utterance_id"].unique())
    return {
        "label": widgets.Dropdown(
            options=label_options,
            value=default_label if default_label in label_options else label_options[0],
            description=f"{prefix}感情",
        ),
        "speaker_id": widgets.Dropdown(options=speaker_options, value=speaker_options[0], description=f"{prefix}話者"),
        "session": widgets.Dropdown(options=session_options, value=session_options[0], description=f"{prefix}区分"),
        "utterance_id": widgets.Dropdown(options=utterance_options, value=utterance_options[0], description=f"{prefix}番号"),
    }


def _selected_sample_row(df, controls):
    """ドロップダウンで選ばれた条件に合うサンプルを1つ返す。"""
    query = (
        (df["label"] == controls["label"].value)
        & (df["speaker_id"] == controls["speaker_id"].value)
        & (df["session"] == controls["session"].value)
        & (df["utterance_id"] == controls["utterance_id"].value)
    )
    matched = df[query]
    if matched.empty:
        return None
    return matched.iloc[0]


def _mfcc_zscore_for_display(y, sr, n_mfcc=20):
    """比較表示用に，C1以降のMFCCを係数ごとに標準化する。"""
    mfcc = compute_mfcc(y, sr, n_mfcc=n_mfcc)
    mfcc_view = mfcc[1:]
    row_mean = mfcc_view.mean(axis=1, keepdims=True)
    row_std = mfcc_view.std(axis=1, keepdims=True) + 1e-8
    return (mfcc_view - row_mean) / row_std


def show_sample_explorer(df, default_label="happy", sample_rate=SAMPLE_RATE):
    """感情・話者・区分・番号を選び，音声と特徴量をまとめて観察する。

    ノートブックの基本セルは固定のサンプルで説明を進めるが，この関数では
    学生が自分でサンプルを選び，音声再生，波形，スペクトログラム，MFCC，
    必要に応じてF0やフォルマントを見比べられるようにする。
    """
    import ipywidgets as widgets

    controls = _make_sample_dropdowns(df, default_label=default_label)
    show_pitch = widgets.Checkbox(value=False, description="F0も表示")
    show_formants = widgets.Checkbox(value=False, description="フォルマントも表示")
    output = widgets.Output()

    def update(_=None):
        row = _selected_sample_row(df, controls)
        with output:
            clear_output(wait=True)
            if row is None:
                print("該当するサンプルがない。選択を変えること。")
                return
            y, sr = librosa.load(row["path"], sr=sample_rate, mono=True)
            print(row[["label", "speaker_id", "session", "utterance_id", "path"]].to_dict())
            display(Audio(y, rate=sr))
            plot_waveform(y, sr)
            plot_spectrogram(y, sr)
            mel_db = plot_mel_spectrogram(y, sr)
            plot_log_mel_dct_frame(mel_db)
            plot_mfcc(y, sr)
            if show_pitch.value:
                pitch_stats = draw_waveform_with_pitch(row["path"])
                display(pd.DataFrame([pitch_stats]).round(2))
            if show_formants.value:
                formant_stats = draw_spectrogram_with_formants(row["path"])
                display(pd.DataFrame([formant_stats]).round(1))

    widgets_to_watch = list(controls.values()) + [show_pitch, show_formants]
    for widget in widgets_to_watch:
        widget.observe(update, names="value")

    ui = widgets.VBox(
        [
            widgets.HBox([controls["label"], controls["speaker_id"]]),
            widgets.HBox([controls["session"], controls["utterance_id"]]),
            widgets.HBox([show_pitch, show_formants]),
        ]
    )
    display(ui, output)
    update()


def compare_samples(df, default_left="happy", default_right="sad", sample_rate=SAMPLE_RATE):
    """2つのサンプルを左右に並べ，音声・波形・スペクトログラム・MFCCを比較する。"""
    import ipywidgets as widgets

    left = _make_sample_dropdowns(df, prefix="左", default_label=default_left)
    right = _make_sample_dropdowns(df, prefix="右", default_label=default_right)
    output = widgets.Output()

    def _plot_pair(rows, waves, sr):
        titles = [
            f"{row['label']} / {row['speaker_id']} / {row['session']} / {row['utterance_id']}"
            for row in rows
        ]
        fig, axes = plt.subplots(3, 2, figsize=(13, 8))
        max_duration = max(len(y) / sr for y in waves)
        for col, (row, y, title) in enumerate(zip(rows, waves, titles)):
            time = np.arange(len(y)) / sr
            axes[0, col].plot(time, y, linewidth=0.8)
            axes[0, col].set_xlim(0, max_duration)
            axes[0, col].set_title(title)
            axes[0, col].set_xlabel("時間 [秒]")
            axes[0, col].set_ylabel("振幅")

            stft = librosa.stft(y, n_fft=1024, hop_length=160, win_length=400)
            stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            librosa.display.specshow(
                stft_db,
                sr=sr,
                hop_length=160,
                x_axis="time",
                y_axis="hz",
                cmap="magma",
                ax=axes[1, col],
            )
            axes[1, col].set_title("スペクトログラム")

            mfcc_z = _mfcc_zscore_for_display(y, sr)
            librosa.display.specshow(
                mfcc_z,
                sr=sr,
                hop_length=160,
                x_axis="time",
                cmap="coolwarm",
                vmin=-2.5,
                vmax=2.5,
                ax=axes[2, col],
            )
            axes[2, col].set_title("MFCC（C1以降を係数ごとに標準化）")
            axes[2, col].set_ylabel("MFCC係数")
        plt.tight_layout()
        plt.show()

    def update(_=None):
        rows = [_selected_sample_row(df, left), _selected_sample_row(df, right)]
        with output:
            clear_output(wait=True)
            if rows[0] is None or rows[1] is None:
                print("該当するサンプルがない。選択を変えること。")
                return
            waves = [librosa.load(row["path"], sr=sample_rate, mono=True)[0] for row in rows]
            print("左")
            display(Audio(waves[0], rate=sample_rate))
            print("右")
            display(Audio(waves[1], rate=sample_rate))
            _plot_pair(rows, waves, sample_rate)

    for widget in list(left.values()) + list(right.values()):
        widget.observe(update, names="value")

    ui = widgets.VBox(
        [
            widgets.HTML("<b>左のサンプル</b>"),
            widgets.HBox([left["label"], left["speaker_id"], left["session"], left["utterance_id"]]),
            widgets.HTML("<b>右のサンプル</b>"),
            widgets.HBox([right["label"], right["speaker_id"], right["session"], right["utterance_id"]]),
        ]
    )
    display(ui, output)
    update()


def display_mfcc_summary(mfcc):
    """MFCCを平均と標準偏差で要約して表示する。

    PCA/LDAの節では，1つの発話を1本の固定長ベクトルにする必要がある。
    ここでは「各MFCC係数の平均」と「各MFCC係数の標準偏差」を使って，
    時間方向の情報を大きく圧縮した特徴量の考え方を示す。
    """
    # PCA/LDA用の固定長特徴量の考え方を示すため，
    # 各MFCC係数を平均と標準偏差で要約して表示する。
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
    plt.title("可視化に使うMFCC要約統計量")
    plt.tight_layout()
    plt.show()
    return summary


# ここから「ピッチ・基本周波数」に対応する処理。
# Praatの推定を使い，F0の時間変化と感情ラベルごとの傾向を見る。
def draw_waveform_with_pitch(path, pitch_floor=75, pitch_ceiling=600):
    """波形の上にPraatで推定したF0を重ねて描く。

    F0は声帯振動の基本周波数で，声の高さと関係する。
    波形だけでは声の高さの時間変化が分かりにくいため，右軸にF0を重ねる。
    """
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    samples = snd.values[0]
    times = np.linspace(snd.xmin, snd.xmax, len(samples))
    pitch_times = pitch.xs()
    f0 = pitch.selected_array["frequency"].astype(float)
    # Praatでは無声区間のF0が0として返る。平均などを計算しやすいようNaNにする。
    f0[f0 == 0] = np.nan

    fig, ax1 = plt.subplots(figsize=(11, 3.5))
    ax1.plot(times, samples, linewidth=0.7, color="gray")
    ax1.set_xlabel("時間 [秒]")
    ax1.set_ylabel("振幅")
    ax1.set_title("波形とPraatによるピッチ軌跡")

    ax2 = ax1.twinx()
    ax2.plot(pitch_times, f0, "o-", markersize=3, linewidth=1.2, color="tab:red", label="F0")
    ax2.set_ylabel("F0 [Hz]")
    ax2.set_ylim(pitch_floor, pitch_ceiling)
    plt.tight_layout()
    plt.show()
    return summarize_f0_array(f0)


def summarize_f0_array(f0):
    """F0系列から平均，標準偏差，最小値，最大値などを計算する。

    f0には，時刻ごとのF0推定値が入っている。
    無声区間ではF0が定義しにくいため，NaNを除いて有声区間だけを要約する。
    """
    # NaNを除いた有声区間だけからF0の統計量を計算する。
    # voiced_ratioは，F0が推定されたフレームの割合である。
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
    """1つの音声ファイルについてF0統計量を返す。

    図を描かずに数値だけを得たいときに使う。
    感情ごとのF0分布を比べる処理から呼び出される。
    """
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan
    return summarize_f0_array(f0)


def summarize_pitch_by_label(df, random_state=RANDOM_STATE, n_per_label=10):
    """感情ラベルごとにF0の傾向を要約する。

    すべての音声を使うのではなく，各感情から同じ程度の数を取り出す。
    これにより，授業中でも短時間で「怒りや驚きではF0が高いのか」などを
    概観できる。
    """
    # 全データを処理すると時間がかかるため，各感情から最大10個ずつ抽出する。
    # random_stateを固定し，授業で毎回同じサンプルが選ばれるようにする。
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
    plt.title("感情ラベルごとのPraat F0平均")
    plt.ylabel("F0平均 [Hz]")
    plt.tight_layout()
    plt.show()
    return stats


# ここから「Praat によるスペクトログラム・フォルマントの描画」に対応する処理。
# フォルマントは母音らしい有声区間で解釈しやすいが，推定値は無音にも出ることがある。
def draw_spectrogram_with_formants(path, maximum_formant=5500, max_frequency=8000):
    """スペクトログラムにF1, F2, F3の推定値を重ねて描く。

    フォルマントは，声道の共鳴によって強く現れる周波数である。
    ただし，Praatの推定値は無音区間や雑音的な区間にも点として表示される
    ことがあるため，背景のスペクトログラムと一緒に解釈する。
    """
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
        # F1, F2, F3をスペクトログラム上に重ねる。
        # 点があることは「安定した母音がある」ことを必ずしも意味しない。
        values = [formant.get_value_at_time(formant_index, t) for t in times]
        plt.plot(times, values, "o", markersize=2, color=color, label=f"F{formant_index}")

    plt.ylim(0, max_frequency)
    plt.xlabel("時間 [秒]")
    plt.ylabel("周波数 [Hz]")
    plt.title("推定フォルマント軌跡（無音・無声区間にも点が出る場合がある）")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def summarize_formants(path, maximum_formant=5500):
    """1つの音声ファイルについてF1, F2, F3の中央値を返す。

    フォルマント推定には外れ値が出ることがあるため，平均ではなく中央値を使う。
    この値は厳密な音声学的測定ではなく，教材用の概観として扱う。
    """
    snd = parselmouth.Sound(str(path))
    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=maximum_formant)
    times = np.arange(snd.xmin, snd.xmax, 0.01)
    summary = {}
    for formant_index in [1, 2, 3]:
        # 外れ値の影響を受けにくくするため，平均ではなく中央値を使う。
        values = np.array([formant.get_value_at_time(formant_index, t) for t in times], dtype=float)
        values = values[np.isfinite(values)]
        summary[f"F{formant_index}_median"] = np.nan if len(values) == 0 else float(np.median(values))
    return summary


def summarize_formants_by_label(df, random_state=RANDOM_STATE, n_per_label=10):
    """感情ラベルごとにフォルマントの概略を要約する。

    JNVは母音をきれいに発声したデータだけではないので，結果は
    「声道共鳴らしいものを観察する練習」として読む。
    """
    # ピッチと同様に，各感情から少数のサンプルを取り出して概観する。
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


# ここから「MFCCと特徴量」に対応する処理。
# PCA/LDA用には，時間方向を平均・標準偏差でまとめた固定長ベクトルを作る。
def extract_mfcc_stats(path, target_sr=SAMPLE_RATE, n_mfcc=20):
    """1つの音声から，PCA/LDA用の固定長MFCC特徴量を作る。

    PCAやLDAでは，各発話を同じ長さのベクトルとして扱いたい。
    そこで，時間方向に並んだMFCCを平均と標準偏差で要約する。

    注意:
        この要約では「いつ声が変化したか」という時間順序は失われる。
        その限界を補うため，後半ではLSTMで時系列そのものを扱う。
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if len(y) < target_sr // 10:
        # delta特徴量を安定して計算するため，極端に短い音声は最低0.1秒にそろえる。
        y = np.pad(y, (0, target_sr // 10 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # MFCC，ΔMFCC，ΔΔMFCCを縦に連結し，平均と標準偏差を固定長特徴量にする。
    feats = np.concatenate([mfcc, delta, delta2], axis=0)
    return np.concatenate([feats.mean(axis=1), feats.std(axis=1)])


def build_mfcc_stat_dataset(df, sample_rate=SAMPLE_RATE):
    """全音声について固定長MFCC特徴量とラベル番号を作る。

    Xは機械学習に入力する数値行列，yは感情ラベルを整数に変換した配列である。
    groupsには話者IDを入れておき，未知話者評価をするときに使う。
    """
    X = np.vstack([extract_mfcc_stats(path, target_sr=sample_rate) for path in df["path"]]).astype(np.float32)
    # LabelEncoderは，"happy" のような文字列ラベルを 0, 1, 2, ... の整数に変換する。
    # 多くの機械学習モデルは，正解ラベルを文字列ではなく整数として受け取る。
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    groups = df["speaker_id"].to_numpy()
    return X, y, groups, label_encoder


# ここから「MFCCのPCA/LDAによる可視化」に対応する処理。
# PCAは感情ラベルを使わず，LDAは感情ラベルを使って2次元に写す。
def _central_pca_view(pca_df, x_col="PC1", y_col="PC2", q=0.02, y_zoom=0.55):
    """PCA散布図を見やすくするため，中心付近の点だけを描画対象にする。

    PCAの計算そのものは全サンプルで行う。ここでは描画範囲だけを調整する。
    外れ値があると大半の点が細い帯に見えるため，教育用の図として読みにくくなる。
    """
    # PCAの計算は全サンプルで行うが，表示は中心部分に絞る。
    # 外れ値で図全体がつぶれて見えることを避けるための表示上の工夫。
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
    """PCA散布図の共通の仕上げを行う。

    表示している点数をタイトルに出し，中心部分だけを表示していることが
    図から分かるようにする。
    """
    plt.title(f"{title}\n中心部分を表示: {limits['shown']}/{limits['total']} サンプル")
    plt.xlim(*limits["xlim"])
    plt.ylim(*limits["ylim"])
    plt.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.axvline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.gca().set_aspect("auto")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_pca_mfcc(df, X, random_state=RANDOM_STATE):
    """固定長MFCC特徴量をPCAで2次元に可視化する。

    PCAは感情ラベルを使わず，データ全体のばらつきが大きい方向を探す。
    そのため，感情よりも話者差や録音条件の違いが強く見えることがある。
    """
    # StandardScalerは，各特徴量の平均を0，標準偏差を1にそろえる。
    # 値の大きい特徴量だけがPCAに強く影響することを避ける。
    scaler = StandardScaler()
    # PCAでは各特徴量のスケール差が結果に影響するため，平均0・分散1に標準化する。
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
        f"MFCCのPCA: PC1 {pca.explained_variance_ratio_[0] * 100:.1f}%, "
        f"PC2 {pca.explained_variance_ratio_[1] * 100:.1f}%"
    )
    _finish_pca_plot(title, limits)
    return X_scaled, pca_df, pca


def plot_lda_mfcc(df, X_scaled, y, label_encoder):
    """固定長MFCC特徴量をLDAで2次元に可視化する。

    LDAは感情ラベルを使って軸を作るため，PCAより分かれて見えやすい。
    ただし，これは可視化であり，未知データに対する分類性能を直接示すものではない。
    """
    # LDAはラベルを使う可視化なので，分離して見えても未知データ性能そのものではない。
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    lda_df = df.copy()
    lda_df["LD1"] = X_lda[:, 0]
    lda_df["LD2"] = X_lda[:, 1]
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=lda_df, x="LD1", y="LD2", hue="label", style="speaker_id", s=80, alpha=0.85)
    plt.title("MFCCのLDA: 感情ラベルを使った教師あり軸")
    plt.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.axvline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return lda_df, lda


def plot_pca_by_speaker(pca_df):
    """PCAの同じ点を，感情ではなく話者IDで色分けして描く。

    これにより，PCAで見えているばらつきが感情差なのか話者差なのかを
    比較しやすくする。
    """
    view_df, limits = _central_pca_view(pca_df)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=view_df, x="PC1", y="PC2", hue="speaker_id", style="label", s=85, alpha=0.88)
    _finish_pca_plot("MFCCのPCA（話者で色分け）", limits)


# ここから「MFCCの時系列を使うLSTM」に対応する処理。
# 固定長の平均ベクトルではなく，時間フレームの並びをそのままモデルに渡す。
def make_split(df, y, groups, mode="random", random_state=RANDOM_STATE):
    """データを学習用とテスト用に分ける。

    random:
        感情ラベルの比率を保ちながら，発話をランダムに分ける。
    speaker_holdout:
        話者が重ならないように分ける。未知話者への評価なので難しくなる。

    ノートブックでは，まず基本的な動作を見るためにrandomを既定値にしている。
    """
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
    """学習用データとテスト用データの内訳を表示する。

    機械学習では，分割後に各感情がどちらにも入っているかを確認することが重要である。
    ここで偏りが大きいと，後のaccuracyやmacro F1の解釈が難しくなる。
    """
    # 学習用データとテスト用データに，どの感情が何個入ったかを確認する。
    # speaker_holdoutでは，未知話者評価になっているかも同時に見る。
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
    """1つの音声をLSTM用のMFCC時系列に変換する。

    固定長特徴量ではなく，各時刻の特徴量を順番に並べた配列を返す。
    返り値の形は「時間フレーム数 × 特徴量数」である。
    特徴量数は，MFCC 20個，ΔMFCC 20個，ΔΔMFCC 20個の合計60個になる。
    """
    # LSTM用の入力。平均せず，「時間フレーム × 特徴量」の表として返す。
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if len(y) < target_sr // 10:
        y = np.pad(y, (0, target_sr // 10 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # 各時刻の特徴量は MFCC + ΔMFCC + ΔΔMFCC。
    # 転置して「時間フレーム × 特徴量」にする点がLSTM入力との対応で重要。
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features.T.astype(np.float32)


def build_sequence_dataset(df, train_idx, y, random_state=RANDOM_STATE, sample_rate=SAMPLE_RATE):
    """全音声をLSTMに入力できる3次元配列にまとめる。

    LSTMに複数の発話をまとめて入力するには，配列の形をそろえる必要がある。
    そこで，短い発話の後ろに0を追加して，すべてを最長発話と同じ長さにする。

    返り値:
        X_seq: 発話数 × 最大時間フレーム数 × 特徴量数 の配列。
        seq_lengths: padding前の本当の長さ。LSTMでpaddingを無視するために使う。
        frame_scaler: 学習データから計算した標準化器。
    """
    sequences = [extract_mfcc_sequence(path, target_sr=sample_rate) for path in df["path"]]
    seq_lengths = np.array([seq.shape[0] for seq in sequences], dtype=np.int64)
    max_frames = int(seq_lengths.max())
    n_features = sequences[0].shape[1]

    frame_scaler = StandardScaler()
    # LSTM入力でも，特徴量ごとの値の大きさをそろえる。
    # ただし，テストデータの情報を学習時に使わないように注意する。
    # テストデータの情報を使わないよう，標準化は学習データのフレームだけでfitする。
    frame_scaler.fit(np.concatenate([sequences[i] for i in train_idx], axis=0))

    # 長さの異なる発話を1つの配列にまとめるため，短い発話の末尾を0でpaddingする。
    # 後でpack_padded_sequenceを使うので，padding部分はLSTMの有効な入力として扱わない。
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
    """MFCC時系列を感情ラベルに分類する小さなLSTMモデル。

    入力:
        x: 発話数 × 時間フレーム数 × 特徴量数
        lengths: 各発話の本当の時間フレーム数

    出力:
        各感情ラベルに対するスコア。最も高いスコアのラベルを予測結果とする。

    読み方:
        1. LSTMが時間フレームを順に読む。
        2. 各時刻のLSTM出力を平均と最大値で1本のベクトルにまとめる。
        3. 最後の全結合層で感情ラベルのスコアに変換する。
    """
    # MFCC時系列を入力し，感情ラベルを出力する小さな双方向LSTM。
    # ノートブックでは，実用モデルではなく時系列情報を残す基本モデルとして扱う。
    def __init__(self, n_features, n_classes, hidden_size=64, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        # batch_first=Trueなので，入力の形は「発話数 × 時間 × 特徴量」になる。
        # bidirectional=Trueでは，前から読むLSTMと後ろから読むLSTMを両方使う。
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        # mean poolingとmax poolingを連結するため，入力次元はlstm_output_sizeの2倍。
        # 最後にn_classes個のスコアを出す。
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def masked_mean_max_pooling(self, outputs, lengths):
        """paddingを除いて，LSTM出力を発話単位のベクトルにまとめる。

        outputsには，各時刻のLSTM出力が入っている。
        しかし，短い発話では後ろの方がpaddingなので，その部分を計算から除く。
        mean_poolは発話全体の平均的な特徴，max_poolは目立つ特徴を拾う。
        """
        # LSTMの各時刻の出力を，発話全体の1本のベクトルにまとめる。
        # paddingされた時刻を平均や最大値に含めないよう，lengthsからmaskを作る。
        frame_ids = torch.arange(outputs.size(1), device=outputs.device).unsqueeze(0)
        mask = frame_ids < lengths.unsqueeze(1)
        mask_float = mask.unsqueeze(-1).to(outputs.dtype)
        mean_pool = (outputs * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        max_pool = outputs.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=1)

    def forward(self, x, lengths):
        """モデルの順伝播。

        PyTorchでは，model(x, lengths) と呼ばれたときにこの関数が実行される。
        pack_padded_sequenceを使うことで，LSTMはpaddingされた0の部分を読まない。
        """
        # pack_padded_sequenceにより，LSTMがpadding部分を読まないようにする。
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=x.size(1))
        return self.classifier(self.masked_mean_max_pooling(outputs, lengths))


def make_loader(X_seq, seq_lengths, y, indices, batch_size=32, shuffle=False):
    """指定されたデータだけを取り出し，ミニバッチで読める形にする。

    DataLoaderは，学習時に「32個ずつ取り出す」といった処理を担当する。
    shuffle=Trueにすると，毎エポックで学習データの順番を入れ替える。
    """
    # numpy配列をPyTorchのTensorDatasetにまとめ，ミニバッチ単位で取り出せるようにする。
    dataset = TensorDataset(
        torch.tensor(X_seq[indices], dtype=torch.float32),
        torch.tensor(seq_lengths[indices], dtype=torch.long),
        torch.tensor(y[indices], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_torch(model, loader, criterion, device):
    """検証データに対するlossとaccuracyを計算する。

    学習中にこの関数を使い，「モデルが学習データだけに覚え込みすぎていないか」を
    検証データで確認する。
    """
    # 学習中の検証用。勾配計算を止め，lossとaccuracyだけを計算する。
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
    """LSTMモデルを学習する。

    ここでは，学習データをミニバッチに分け，誤差を計算し，PyTorchのoptimizerで
    パラメータを少しずつ更新する。検証lossがしばらく改善しない場合は早めに止める。

    初学者向けの見方:
        コードの細部よりも，history_dfのlossとval_lossがどう変化するかを見る。
        val_lossが下がらなくなったら，それ以上学習しても汎化性能は上がりにくい。
    """
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_seq.shape[-1]
    model = MFCCLSTM(n_features=n_features, n_classes=len(label_encoder.classes_)).to(device)
    print("device:", device)
    print(model)

    # DataLoaderは，配列全体からミニバッチを順に取り出すためのPyTorchの仕組み。
    train_loader = make_loader(X_seq, seq_lengths, y, train_idx, shuffle=True)
    val_loader = make_loader(X_seq, seq_lengths, y, val_idx)
    class_counts = np.bincount(y[train_idx], minlength=len(label_encoder.classes_))
    # 感情ラベルの数に偏りがあるため，少ないクラスの誤りを少し重く扱う。
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
        # 1 epochは，学習データ全体を一通り使ってパラメータを更新する単位。
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for xb, lengths, yb in train_loader:
            # xb: 音声特徴量，lengths: padding前の長さ，yb: 正解ラベル。
            xb = xb.to(device)
            lengths = lengths.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            # loss.backward()で，各パラメータをどちらに動かせばlossが下がるかを計算する。
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # optimizer.step()で，計算した勾配に基づいてパラメータを更新する。
            optimizer.step()
            train_loss_sum += loss.item() * len(yb)
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += len(yb)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate_torch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "loss": train_loss, "accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc})

        if val_loss < best_val_loss:
            # 検証lossが最も低い時点の重みを保存する。
            # これにより，学習しすぎた最後の状態ではなく，よかった状態を使える。
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                # 検証lossが改善しない状態が続いたら早めに学習を止める。
                break

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    display(history_df.tail())
    plt.figure(figsize=(10, 4))
    plt.plot(history_df["epoch"], history_df["loss"], label="学習loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="検証loss")
    plt.xlabel("エポック")
    plt.ylabel("損失")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return model, history_df, device


def evaluate_lstm(model, X_seq, seq_lengths, y, train_idx, test_idx, label_encoder, device):
    """テストデータでLSTMを評価し，授業で読むための結果を表示する。

    表示するもの:
        多数派ベースライン: 最も多い感情だけを予測する単純な基準。
        accuracy / macro F1: 全体の正解率と，感情ごとのバランスを見た指標。
        predicted_count: モデルの予測が特定ラベルに偏っていないかを見る表。
        confusion matrix: どの感情をどの感情と間違えたかを見る表。
    """
    # ノートブック「実行後の結果の見方」に対応する評価処理。
    # 多数派ベースライン，accuracy，macro F1，予測分布，混同行列をまとめて表示する。
    test_loader = make_loader(X_seq, seq_lengths, y, test_idx)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for xb, lengths, _ in test_loader:
            logits = model(xb.to(device), lengths.to(device))
            all_logits.append(logits.cpu())

    # logitsは各感情ラベルのスコア。argmaxで最もスコアの高いラベル番号を選ぶ。
    pred = torch.cat(all_logits, dim=0).argmax(dim=1).numpy()
    # 学習データで最も多い感情だけを常に予測する単純な基準。
    # LSTMがこれを上回らない場合，感情を学習できたとは言いにくい。
    majority_class = np.bincount(y[train_idx], minlength=len(label_encoder.classes_)).argmax()
    majority_pred = np.full_like(y[test_idx], majority_class)
    result = pd.DataFrame(
        [
            {
                "model": "多数派ベースライン",
                "accuracy": accuracy_score(y[test_idx], majority_pred),
                "macro_f1": f1_score(y[test_idx], majority_pred, average="macro", zero_division=0),
            },
            {
                "model": "MFCC時系列LSTM",
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
    # 予測が1〜2種類の感情に集中している場合は，混同行列が1列に潰れやすい。
    if np.count_nonzero(prediction_summary["predicted_count"].to_numpy()) <= 2:
        print("警告: 予測が1〜2種類のラベルに集中しているため，感情カテゴリを十分に学習したとは解釈しにくい．")

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
    plt.xlabel("予測ラベル")
    plt.ylabel("正解ラベル")
    plt.title("混同行列: MFCC時系列を用いたLSTM")
    plt.tight_layout()
    plt.show()
    return pred, result, prediction_summary, cm
