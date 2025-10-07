from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cmocean
import colorcet as cc
import librosa
import librosa.display
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.transform import resize

matplotlib.rcParams.update({
    "svg.fonttype": "path",
    "figure.figsize": (5, 5),
    "figure.dpi": 300,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "font.size": 20,
    "font.weight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False
})


def resolve_cmap(cmap: Union[str, matplotlib.colors.Colormap]) -> matplotlib.colors.Colormap:
    if isinstance(cmap, matplotlib.colors.Colormap):
        return cmap
    try:
        return matplotlib.colormaps.get_cmap(cmap)
    except Exception:
        pass
    try:
        return getattr(cc.cm, cmap)
    except Exception:
        pass
    try:
        return getattr(cmocean.cm, cmap)
    except Exception:
        pass
    return matplotlib.colormaps.get_cmap('viridis')


def _min_max_norm(im: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    if vmin is None:
        vmin = np.percentile(im, 0.01)
    if vmax is None:
        vmax = np.percentile(im, 99.99)
    denom = max(vmax - vmin, 1e-12)
    return np.clip((im - vmin) / denom, 0.0, 1.0)


def find_audio_and_csv(
    folder: str,
    csv_name: Optional[str] = None,
    recursive: bool = False,
    csv_suffix: str = "_new.csv"
) -> Tuple[List[str], Optional[str]]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    wavs, csvs = [], []
    search_path = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for root, _, files in search_path:
        for f in files:
            fp = os.path.join(root, f)
            fl = f.lower()
            if fl.endswith(".wav"):
                wavs.append(fp)
            elif fl.endswith(csv_suffix.lower()):
                csvs.append(fp)
    wavs.sort(key=lambda p: os.path.basename(p).lower())
    if csv_name:
        chosen_csv = csv_name if os.path.isabs(csv_name) else os.path.join(folder, csv_name)
        if not os.path.isfile(chosen_csv):
            raise FileNotFoundError(f"Specified CSV file not found: {chosen_csv}")
        return wavs, chosen_csv
    preferred = [c for c in csvs if "annotation" in os.path.basename(c).lower()]
    return wavs, (preferred[0] if preferred else (csvs[0] if csvs else None))


def estimate_common_freq_bounds(audio_files: List[str]) -> Tuple[float, float]:
    if not audio_files:
        return 0.0, 8000.0
    min_nyquist = float('inf')
    for f in audio_files:
        try:
            sr = librosa.get_samplerate(f)
        except Exception:
            _, sr = librosa.load(f, sr=None, mono=True)
        if sr:
            min_nyquist = min(min_nyquist, sr / 2.0)
    return 0.0, min_nyquist if min_nyquist != float('inf') else 8000.0


def _balanced_pack_rows(durations: Sequence[float], rows: int) -> List[List[int]]:
    N = len(durations)
    if rows <= 0:
        return [[]]
    if rows >= N:
        return [[i] for i in range(N)]

    indexed_durations = sorted(enumerate(durations), key=lambda x: x[1], reverse=True)
    packed = [[] for _ in range(rows)]
    row_time = [0.0] * rows
    for original_index, duration in indexed_durations:
        lightest_row = min(range(rows), key=lambda r: row_time[r])
        packed[lightest_row].append(original_index)
        row_time[lightest_row] += duration
    return packed


def _get_crop_window(
    audio_file: str,
    df_annotation: Optional[pd.DataFrame],
    padding_ms: float = 200.0,
    target_sr: Optional[int] = None,
) -> Tuple[float, float]:
    start_time, end_time = 0.0, 0.0
    found_annotation = False
    if df_annotation is not None and {"filename", "onset", "offset"}.issubset(df_annotation.columns):
        base = os.path.basename(audio_file)
        base_no_ext = os.path.splitext(base)[0]
        matches = df_annotation[
            df_annotation["filename"].astype(str).str.contains(base, regex=False, case=False) |
            df_annotation["filename"].astype(str).str.contains(base_no_ext, regex=False, case=False)
        ]
        if matches.empty:
            indiv_csv = os.path.splitext(audio_file)[0] + "_new.csv"
            if os.path.exists(indiv_csv):
                try:
                    matches = pd.read_csv(indiv_csv)
                except Exception:
                    matches = pd.DataFrame()
        if not matches.empty and {"onset", "offset"}.issubset(matches.columns):
            min_onset = float(matches["onset"].min())
            max_offset = float(matches["offset"].max())
            padding_secs = padding_ms / 1000.0
            start_time = max(0.0, min_onset - padding_secs)
            end_time = max_offset + padding_secs
            found_annotation = True
    if not found_annotation:
        try:
            end_time = librosa.get_duration(path=audio_file)
        except Exception:
            y, sr = librosa.load(audio_file, sr=target_sr)
            end_time = len(y) / sr if sr > 0 else 0.0
    return start_time, end_time


def _parse_aspect(s: str) -> float:
    if ":" in s or "x" in s.lower():
        sep = ":" if ":" in s else "x"
        a, b = s.lower().split(sep)
        return float(a) / float(b)
    return float(s)


def compute_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    center: bool,
    win_length: Optional[int],
    window: str,
    spec_type: str,
    n_mels: int,
    fmin: float,
    fmax: float,
    ref_db_max: bool = True
) -> Tuple[np.ndarray, float]:
    S = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, center=center, win_length=win_length, window=window)
    power_spec = np.abs(S) ** 2

    if spec_type == "mel":
        if fmax is None or fmax <= 0:
            fmax = sr / 2
        melfb = librosa.filters.mel(sr=sr, n_mels=n_mels, n_fft=n_fft, fmin=fmin or 0.0, fmax=fmax)
        mel_spec = np.matmul(melfb, power_spec)
        db_spec = librosa.power_to_db(mel_spec, ref=np.max if ref_db_max else 1.0)
    else:
        db_spec = librosa.power_to_db(power_spec, ref=np.max if ref_db_max else 1.0)

    duration_sec = (db_spec.shape[1] * hop_length) / sr
    return db_spec, duration_sec


def parse_bird_from_filename(path: str) -> str:
    base_no_ext = os.path.splitext(os.path.basename(path))[0]
    return base_no_ext.split("_")[0] if "_" in base_no_ext else base_no_ext


def parse_age_from_filename(path: str, hatch_dates: Dict[str, int]) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r'^(?P<bird>[A-Za-z]\d+)[_\-](?P<date>\d+)', base)
    if not m:
        return None
    bird = m.group("bird")
    date_num = int(m.group("date"))
    if bird in hatch_dates:
        return date_num - int(hatch_dates[bird])
    return None


def plot_auto_grid(
    audio_files: List[str],
    annotation_csv: Optional[str],
    *,
    # Spectrogram controls
    spec_type: str = "linear",  # "linear" or "mel"
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    ref_db_max: bool = True,
    db_floor: float = -80.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    # Frequency/display
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    force_common_freq_bounds: bool = True,
    y_axis: str = "linear",  # "linear" or "mel" (display only)
    # Layout
    rows: Optional[int] = None,  # override row count
    gap: Union[float, Tuple[float, float]] = (0.2, 1.0),
    seed: Optional[int] = 0,
    time_zoom: float = 1.0,
    target_aspect: float = 9 / 16,
    target_sr: Optional[int] = None,
    # Sorting
    sort_by: Optional[str] = None,  # 'age','filename','duration','random'
    sort_ascending: bool = True,
    # Titles
    title_mode: str = "bird_dph",  # "bird","bird_dph","basename","none"
    title_color: str = "white",
    title_bg: str = "black",
    show_titles: bool = True,
    # Waveform
    plot_waveform: bool = False,
    waveform_color: str = "black",
    waveform_height_ratio: float = 0.3,
    # Annotations
    show_annotations: bool = True,
    annotation_color: str = "black",
    annotation_box_style: str = "auto",  # "auto","brackets","boxes"
    annotation_expand_mode: str = "callmark",  # "none","expand","callmark"
    # Add highlight options
    annotation_highlight: bool = False,
    annotation_highlight_color: str = "green",
    annotation_highlight_lw: float = 4.0,
    annotation_highlight_y: float = 1.02,
    # Gaps/background
    gap_color: Union[str, Tuple[float, float, float]] = "white",
    # Color mapping
    cmap: Union[str, matplotlib.colors.Colormap] = "CET_L20",
    # Colorbar
    show_colorbar: bool = True,
    # Scalebar
    scalebar_sec: float = 0.5,
    scalebar_pos: str = "right",  # "right" or "left"
    # Output
    save_file_name: str = "plot.png",
    dpi: int = 300,
    transparent: bool = False,
    # Age/Hatch date
    hatch_dates: Optional[Dict[str, int]] = None
) -> None:
    assert spec_type in ("linear", "mel")
    assert y_axis in ("linear", "mel")
    assert annotation_expand_mode in ("none", "expand", "callmark")
    assert annotation_box_style in ("auto", "brackets", "boxes")
    assert scalebar_pos in ("right", "left")

    if not audio_files:
        raise ValueError("No audio files provided.")

    rng = random.Random(seed)
    cmap_resolved = resolve_cmap(cmap)
    df_annotation = pd.read_csv(annotation_csv) if (annotation_csv and os.path.isfile(annotation_csv)) else None

    # SR and STFT params
    sr0 = librosa.get_samplerate(audio_files[0])
    final_sr = target_sr if target_sr else sr0
    hop_length = hop_length if hop_length is not None else max(1, n_fft // 16)
    win_length = win_length if win_length is not None else None  # librosa will default to n_fft if None

    # Ages (optional)
    hatch_dates = hatch_dates or {}

    # Crop windows and metadata
    clip_metadata = []
    for f in audio_files:
        s, e = _get_crop_window(f, df_annotation, padding_ms=200.0, target_sr=final_sr)
        meta = {
            "path": f, "start_sec": s, "end_sec": e, "duration": e - s,
            "dph": parse_age_from_filename(f, hatch_dates),
            "bird": parse_bird_from_filename(f)
        }
        clip_metadata.append(meta)

    # Sorting
    indices = list(range(len(audio_files)))
    if sort_by == "age":
        indices.sort(key=lambda i: (clip_metadata[i]["dph"] is None, clip_metadata[i]["dph"], clip_metadata[i]["bird"]))
    elif sort_by == "filename":
        indices.sort(key=lambda i: os.path.basename(clip_metadata[i]["path"]).lower())
    elif sort_by == "duration":
        indices.sort(key=lambda i: clip_metadata[i]["duration"])
    elif sort_by == "random":
        rng.shuffle(indices)
    else:
        # default: stable by filename
        indices.sort(key=lambda i: os.path.basename(clip_metadata[i]["path"]).lower())

    if not sort_ascending:
        indices = list(reversed(indices))

    clip_metadata = [clip_metadata[i] for i in indices]
    audio_files = [audio_files[i] for i in indices]

    durations = [m["duration"] for m in clip_metadata]

    # Row packing
    if len(audio_files) > 1:
        TARGET_ASPECT_RATIO = float(target_aspect)
        WIDTH_MULTIPLIER = 2.0 * float(time_zoom)
        HEIGHT_PER_ROW = 4.0
        avg_gap = (gap[0] + gap[1]) / 2 if isinstance(gap, tuple) else float(gap)
        best_rows = 1
        min_cost = float("inf")
        for r in range(1, len(audio_files) + 1):
            packed_indices = _balanced_pack_rows(durations, r)
            row_widths_sec = [sum(durations[i] for i in row) + (len(row) - 1) * avg_gap for row in packed_indices if row]
            if not row_widths_sec:
                continue
            max_duration_sec = max(row_widths_sec)
            est_plot_width = max_duration_sec * WIDTH_MULTIPLIER
            est_plot_height = r * HEIGHT_PER_ROW
            if est_plot_height <= 0:
                continue
            current_aspect = est_plot_width / est_plot_height
            cost = abs(current_aspect - TARGET_ASPECT_RATIO)
            if cost < min_cost:
                min_cost = cost
                best_rows = r
        effective_rows = rows if (rows and rows > 0) else best_rows
    else:
        effective_rows = 1

    row_indices = _balanced_pack_rows(durations, rows=effective_rows)
    effective_rows = len(row_indices)

    # Gaps per row
    if isinstance(gap, tuple):
        all_row_gaps = [[rng.uniform(*gap) for _ in range(len(r) - 1)] for r in row_indices]
    else:
        all_row_gaps = [[float(gap) for _ in range(len(r) - 1)] for r in row_indices]

    # Frequency bounds
    if force_common_freq_bounds or fmin is None or fmax is None:
        auto_fmin, auto_fmax = estimate_common_freq_bounds(audio_files)
        fmin = auto_fmin if fmin is None else fmin
        fmax = auto_fmax if fmax is None else fmax
    if fmax is not None:
        # ensure <= Nyquist of final_sr
        fmax = min(fmax, final_sr / 2.0)

    # Compute specs per clip
    all_specs_data = []
    for meta in clip_metadata:
        y, sr = librosa.load(
            meta["path"], sr=final_sr, mono=True,
            offset=meta["start_sec"], duration=meta["duration"]
        )
        db_spec, spec_duration = compute_spectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, center=center,
            win_length=win_length, window=window, spec_type=spec_type,
            n_mels=n_mels, fmin=fmin or 0.0, fmax=fmax or (sr / 2.0), ref_db_max=ref_db_max
        )
        all_specs_data.append({"raw_spec": db_spec, "duration": spec_duration})

    # Global min/max for colormap
    global_max = 0.0 if ref_db_max else (vmax if vmax is not None else None)
    global_min = db_floor if vmin is None else vmin

    # Row assembly
    num_freq_bins = all_specs_data[0]["raw_spec"].shape[0] if all_specs_data else (1 + n_fft // 2)
    row_final_data = []
    max_row_duration = 0.0
    for r_num, indices_in_row in enumerate(row_indices):
        row_parts, time_cursor = [], 0.0
        gaps_in_row = all_row_gaps[r_num]
        for j, idx in enumerate(indices_in_row):
            data = all_specs_data[idx]
            row_parts.append(data["raw_spec"])
            time_cursor += data["duration"]
            if j < len(gaps_in_row):
                gap_sec = gaps_in_row[j]
                gap_cols = max(1, int(round(gap_sec * final_sr / hop_length)))
                gap_spec = np.full((num_freq_bins, gap_cols), global_min - 20)
                row_parts.append(gap_spec)
                time_cursor += gap_sec
        full_row_spec = np.hstack(row_parts) if row_parts else np.full((num_freq_bins, 1), global_min - 20)
        row_final_data.append({'spec': full_row_spec, 'duration': time_cursor, 'indices': indices_in_row})
        max_row_duration = max(max_row_duration, time_cursor)

    # Figure sizing
    TARGET_ASPECT_RATIO = float(target_aspect)
    min_height = max(4.0, 2.6 * effective_rows)
    fig_width_by_time = max(14.0, max_row_duration * 2.0 * float(time_zoom))
    fig_width = max(fig_width_by_time, TARGET_ASPECT_RATIO * min_height)
    fig_height = fig_width / TARGET_ASPECT_RATIO

    # Axes creation
    if plot_waveform:
        total_rows = effective_rows * 2
        fig, axes = plt.subplots(nrows=total_rows, ncols=1, figsize=(fig_width, fig_height * (1 + waveform_height_ratio)), sharex=True, dpi=dpi)
    else:
        fig, axes = plt.subplots(nrows=effective_rows, ncols=1, figsize=(fig_width, fig_height), sharey=True, squeeze=False, dpi=dpi)
    axes = axes.flatten()
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12, hspace=0.35)

    # Annotation timing adjustment
    half_window_sec = ((win_length if win_length is not None else n_fft) / (2.0 * final_sr))
    callmark_shift_sec = ((n_fft - hop_length) / 2.0) / final_sr
    # For librosa center=True, display x-axis already matches center-of-frame; we typically prefer expand mode

    # Y-axis display type
    display_y_axis = "mel" if y_axis == "mel" else "linear"

    # Plot rows
    for r in range(effective_rows):
        if plot_waveform:
            ax_wave = axes[2 * r]
            ax_spec = axes[2 * r + 1]
        else:
            ax_spec = axes[r]

        # Waveform
        if plot_waveform:
            time_cursor = 0.0
            for j, idx in enumerate(row_final_data[r]['indices']):
                meta = clip_metadata[idx]
                y_full, sr = librosa.load(meta["path"], sr=final_sr, mono=True, offset=meta["start_sec"], duration=meta["duration"])
                times = np.linspace(0, len(y_full) / sr, len(y_full), endpoint=False)
                ax_wave.plot(times + time_cursor, y_full, color=waveform_color, lw=0.8)
                time_cursor += meta["duration"]
                if j < len(all_row_gaps[r]):
                    time_cursor += all_row_gaps[r][j]
            ax_wave.set_xlim(0, max_row_duration)
            ax_wave.set_ylim(-1, 1)
            ax_wave.axis("off")

        # Spectrogram
        ax_spec.set_facecolor(gap_color)
        for spine in ax_spec.spines.values():
            spine.set_visible(False)
        ax_spec.set_xlim(0, max_row_duration)

        row_data = row_final_data[r]
        full_row_spec = row_data['spec']

        librosa.display.specshow(
            full_row_spec, ax=ax_spec, sr=final_sr, hop_length=hop_length,
            x_axis='time', y_axis=display_y_axis, fmin=fmin or 0.0, fmax=fmax or (final_sr / 2.0),
            cmap=cmap_resolved, vmin=global_min, vmax=global_max
        )

        # Y ticks
        if display_y_axis == "linear":
            if fmin is not None and fmax is not None:
                custom_ticks = np.linspace(fmin, fmax, num=4, dtype=int)
                ax_spec.set_yticks(custom_ticks)
        ax_spec.set_ylabel(None)

        # Titles and annotations per segment
        time_cursor = 0.0
        indices_in_row = row_data['indices']
        gaps_in_row = all_row_gaps[r]
        for j, idx in enumerate(indices_in_row):
            meta = clip_metadata[idx]
            data = all_specs_data[idx]

            # Title
            if show_titles:
                base_no_ext = os.path.splitext(os.path.basename(meta["path"]))[0]
                bird = meta["bird"]
                title = ""
                if title_mode == "none":
                    title = ""
                elif title_mode == "bird":
                    title = f"{bird}"
                elif title_mode == "bird_dph":
                    if meta["dph"] is not None:
                        title = f"{bird} ({meta['dph']})"
                    else:
                        title = f"{bird}"
                elif title_mode == "basename":
                    title = base_no_ext
                else:
                    title = f"{bird}"

                if title:
                    ax_spec.text(
                        time_cursor, 1.06, title,
                        fontsize=20, fontweight="bold",
                        ha="left", va="bottom", color=title_color,
                        bbox=dict(facecolor=title_bg, pad=2),
                        transform=ax_spec.get_xaxis_transform(),
                        clip_on=False
                    )

            # Annotations
            if show_annotations:
                matches = pd.DataFrame()
                if df_annotation is not None and all(c in df_annotation.columns for c in ["filename", "onset", "offset"]):
                    base, base_no_ext = os.path.basename(meta["path"]), os.path.splitext(os.path.basename(meta["path"]))[0]
                    matches = df_annotation[
                        df_annotation["filename"].astype(str).str.contains(base, regex=False, case=False) |
                        df_annotation["filename"].astype(str).str.contains(base_no_ext, regex=False, case=False)
                    ]
                if matches.empty and os.path.exists(os.path.splitext(meta["path"])[0] + "_new.csv"):
                    try:
                        matches = pd.read_csv(os.path.splitext(meta["path"])[0] + "_new.csv")
                    except Exception:
                        matches = pd.DataFrame()

                if not matches.empty and {"onset", "offset"}.issubset(matches.columns):
                    has_freq_box = {"minFrequency", "maxFrequency"}.issubset(matches.columns)
                    for _, row in matches.iterrows():
                        base_onset = float(row["onset"]) - meta["start_sec"] + time_cursor
                        base_offset = float(row["offset"]) - meta["start_sec"] + time_cursor

                        # Timing mode
                        if annotation_expand_mode == "expand":
                            onset = base_onset - half_window_sec
                            offset = base_offset + half_window_sec
                        elif annotation_expand_mode == "callmark":
                            onset = base_onset - callmark_shift_sec
                            offset = base_offset + callmark_shift_sec
                        else:  # "none"
                            onset = base_onset
                            offset = base_offset

                        onset = max(time_cursor, onset)
                        offset = min(time_cursor + data["duration"], offset)
                        if not (offset > time_cursor and onset < (time_cursor + data["duration"])):
                            continue

                        # Optional green highlight bar above the axis
                        if annotation_highlight:
                            ax_spec.plot(
                                [onset, offset], [annotation_highlight_y, annotation_highlight_y],
                                color=annotation_highlight_color, linewidth=annotation_highlight_lw,
                                transform=ax_spec.get_xaxis_transform(), clip_on=False
                            )

                        draw_boxes = annotation_box_style in ("boxes",) or (annotation_box_style == "auto" and has_freq_box)
                        if draw_boxes and has_freq_box and pd.notna(row["minFrequency"]) and pd.notna(row["maxFrequency"]):
                            min_freq = float(row["minFrequency"])
                            max_freq = float(row["maxFrequency"])
                            if (fmin is None or max_freq > fmin) and (fmax is None or min_freq < fmax):
                                rect = patches.Rectangle(
                                    (onset, min_freq), offset - onset, max_freq - min_freq,
                                    linewidth=1, linestyle=':', edgecolor='white', facecolor='none'
                                )
                                ax_spec.add_patch(rect)
                        else:
                            # Brackets on time axis
                            xform = ax_spec.get_xaxis_transform()
                            ax_spec.plot([onset, offset], [1.02, 1.02], color=annotation_color, lw=1.5,
                                         transform=xform, clip_on=False)
                            ax_spec.plot([onset, onset], [1.02, 1.0 - 0.04], color=annotation_color, lw=1.5,
                                         transform=xform, clip_on=False)
                            ax_spec.plot([offset, offset], [1.02, 1.0 - 0.04], color=annotation_color, lw=1.5,
                                         transform=xform, clip_on=False)

            time_cursor += data["duration"] + (gaps_in_row[j] if j < len(gaps_in_row) else 0)

        ax_spec.set_xticks([])
        ax_spec.set_xlabel(None)

    fig.supylabel("Frequency (Hz)" if display_y_axis == "linear" else "Mel", fontsize=20, fontweight="bold", x=0.01)
    last_ax = axes[-1] if axes.size else None

    # Scalebar
    if scalebar_sec and last_ax is not None:
        transform = last_ax.get_xaxis_transform()
        y = -0.18
        if scalebar_pos == "right":
            x0 = max_row_duration - scalebar_sec
            x1 = max_row_duration
        else:
            x0 = 0.0
            x1 = scalebar_sec
        last_ax.plot([x0, x1], [y, y], color='black', lw=4, transform=transform, clip_on=False)
        last_ax.text((x0 + x1) / 2.0, y - 0.06, f"{scalebar_sec} s", color='black', ha='center', va='top',
                     fontsize=20, fontweight="bold", transform=transform)

    # Colorbar
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap_resolved, norm=plt.Normalize(vmin=global_min, vmax=global_max))
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.015, pad=0.04)
        cbar.set_label('Power (dB)', fontsize=20, fontweight="bold")

    plt.savefig(save_file_name, pad_inches=0.2, transparent=transparent)
    plt.close(fig)


def plot_folder(
    folder: str,
    *,
    csv_name: Optional[str] = None,
    output: Optional[str] = None,
    recursive: bool = False,
    csv_suffix: str = "_new.csv",
    # Forwarded options (subset; see CLI for more)
    spec_type: str = "linear",
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    ref_db_max: bool = True,
    db_floor: float = -80.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    force_common_freq_bounds: bool = True,
    y_axis: str = "linear",
    rows: Optional[int] = None,
    gap: Union[float, Tuple[float, float]] = (0.2, 1.0),
    seed: Optional[int] = 0,
    time_zoom: float = 1.0,
    target_aspect: float = 9/16,
    target_sr: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_ascending: bool = True,
    title_mode: str = "bird_dph",
    title_color: str = "white",
    title_bg: str = "black",
    show_titles: bool = True,
    plot_waveform: bool = False,
    waveform_color: str = "black",
    waveform_height_ratio: float = 0.3,
    show_annotations: bool = True,
    annotation_color: str = "black",
    annotation_box_style: str = "auto",
    annotation_expand_mode: str = "expand",
    annotation_highlight: bool = False,
    annotation_highlight_color: str = "green",
    annotation_highlight_lw: float = 4.0,
    annotation_highlight_y: float = 1.02,
    gap_color: Union[str, Tuple[float, float, float]] = "white",
    cmap: Union[str, matplotlib.colors.Colormap] = "CET_L20",
    show_colorbar: bool = True,
    scalebar_sec: float = 0.5,
    scalebar_pos: str = "right",
    output_format: str = "svg",
    dpi: int = 300,
    transparent: bool = False,
    hatch_dates_file: Optional[str] = None,
    hatch_dates: Optional[Dict[str, int]] = None
) -> str:
    audio_files, ann_csv = find_audio_and_csv(folder, csv_name=csv_name, recursive=recursive, csv_suffix=csv_suffix)
    if not audio_files:
        raise ValueError(f"No WAV files found in {folder}")
    if output is None:
        parent = os.path.abspath(folder)
        output = os.path.join(parent, f"{os.path.basename(os.path.normpath(folder))}_plot.{output_format}")

    if hatch_dates is None and hatch_dates_file and os.path.isfile(hatch_dates_file):
        with open(hatch_dates_file, "r") as f:
            try:
                hatch_dates = json.load(f)
            except Exception:
                hatch_dates = {}

    plot_auto_grid(
        audio_files=audio_files, annotation_csv=ann_csv,
        spec_type=spec_type, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, center=center, ref_db_max=ref_db_max,
        db_floor=db_floor, vmin=vmin, vmax=vmax, fmin=fmin, fmax=fmax,
        force_common_freq_bounds=force_common_freq_bounds, y_axis=y_axis,
        rows=rows, gap=gap, seed=seed, time_zoom=time_zoom, target_aspect=target_aspect,
        target_sr=target_sr, sort_by=sort_by, sort_ascending=sort_ascending,
        title_mode=title_mode, title_color=title_color, title_bg=title_bg, show_titles=show_titles,
        plot_waveform=plot_waveform, waveform_color=waveform_color, waveform_height_ratio=waveform_height_ratio,
        show_annotations=show_annotations, annotation_color=annotation_color,
        annotation_box_style=annotation_box_style, annotation_expand_mode=annotation_expand_mode,
        annotation_highlight=annotation_highlight,
        annotation_highlight_color=annotation_highlight_color,
        annotation_highlight_lw=annotation_highlight_lw,
        annotation_highlight_y=annotation_highlight_y,
        gap_color=gap_color, cmap=cmap, show_colorbar=show_colorbar, scalebar_sec=scalebar_sec,
        scalebar_pos=scalebar_pos, save_file_name=output, dpi=dpi, transparent=transparent,
        hatch_dates=hatch_dates
    )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Pack spectrograms into a rectangular, time-true grid with rich controls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    io = parser.add_argument_group("Input/Output")
    io.add_argument("folder", help="Folder containing .wav files; optionally an annotation CSV.")
    io.add_argument("--csv", dest="csv_name", default=None, help="Specific annotation CSV file path or name.")
    io.add_argument("--csv-suffix", default="_new.csv", help="Per-file CSV suffix to auto-load if master CSV not matched.")
    io.add_argument("--output", default=None, help="Output image file path.")
    io.add_argument("--format", default="svg", help="Output file format (e.g., 'png', 'svg', 'pdf').")
    io.add_argument("--recursive", action="store_true", help="Search for audio files recursively.")
    io.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    io.add_argument("--transparent", action="store_true", help="Save with transparent background.")

    spec = parser.add_argument_group("Spectrogram")
    spec.add_argument("--spec-type", choices=["linear", "mel"], default="linear")
    spec.add_argument("--n-mels", type=int, default=128)
    spec.add_argument("--n-fft", type=int, default=2048)
    spec.add_argument("--hop-length", type=int, default=None)
    spec.add_argument("--win-length", type=int, default=None)
    spec.add_argument("--window", type=str, default="hann")
    spec.add_argument("--center", action="store_true", default=True)
    spec.add_argument("--no-center", dest="center", action="store_false")
    spec.add_argument("--ref-db-max", action="store_true", default=True, help="Reference dB to max power.")
    spec.add_argument("--no-ref-db-max", dest="ref_db_max", action="store_false")
    spec.add_argument("--db-floor", type=float, default=-80.0, help="Display floor for dB.")
    spec.add_argument("--vmin", type=float, default=None, help="Override display vmin.")
    spec.add_argument("--vmax", type=float, default=None, help="Override display vmax.")
    spec.add_argument("--target-sr", type=int, default=None, help="Resample all audio to this sample rate.")
    spec.add_argument("--fmin", type=float, default=None)
    spec.add_argument("--fmax", type=float, default=None)
    spec.add_argument("--force-common-freq-bounds", action="store_true", default=True)
    spec.add_argument("--no-force-common-freq-bounds", dest="force_common_freq_bounds", action="store_false")
    spec.add_argument("--y-axis", choices=["linear", "mel"], default="linear")

    layout = parser.add_argument_group("Layout")
    layout.add_argument("--rows", type=int, default=None, help="Override number of rows (auto if None).")
    layout.add_argument("--gap", type=float, default=None, help="Fixed gap in seconds (overrides --gap-range).")
    layout.add_argument("--gap-range", type=float, nargs=2, default=[0.2, 1.0], help="Random gap range (min max) in seconds.")
    layout.add_argument("--seed", type=int, default=0, help="Random seed for gap sampling and random sort.")
    layout.add_argument("--time-zoom", type=float, default=1.0, help="Multiply horizontal space per second.")
    layout.add_argument("--aspect", type=str, default="9:16", help="Target aspect ratio, e.g. '9:16' or '1.777'.")

    sort = parser.add_argument_group("Sorting")
    sort.add_argument("--sort-by", choices=["age", "filename", "duration", "random"], default=None)
    sort.add_argument("--sort-desc", action="store_true", help="Sort descending instead of ascending.")

    style = parser.add_argument_group("Style")
    style.add_argument("--cmap", default="CET_L20", help="Colormap name (matplotlib/colorcet/cmocean).")
    style.add_argument("--gap-color", default="white", help="Background/gap color.")
    style.add_argument("--show-colorbar", action="store_true", default=True)
    style.add_argument("--no-colorbar", dest="show_colorbar", action="store_false")

    titles = parser.add_argument_group("Titles")
    titles.add_argument("--title-mode", choices=["bird", "bird_dph", "basename", "none"], default="bird")
    titles.add_argument("--title-color", default="white")
    titles.add_argument("--title-bg", default="black")
    titles.add_argument("--no-titles", dest="show_titles", action="store_false")

    waves = parser.add_argument_group("Waveform")
    waves.add_argument("--waveform", action="store_true", default=False)
    waves.add_argument("--waveform-color", default="black")
    waves.add_argument("--waveform-height-ratio", type=float, default=0.3)

    ann = parser.add_argument_group("Annotations")
    ann.add_argument("--show-annotations", action="store_true", default=True)
    ann.add_argument("--no-annotations", dest="show_annotations", action="store_false")
    ann.add_argument("--annotation-color", default="black")
    ann.add_argument("--annotation-style", choices=["auto", "brackets", "boxes"], default="auto")
    ann.add_argument("--annotation-expand-mode", choices=["none", "expand", "callmark"], default="callmark",
                     help="'expand' pads annotations by ±half-window; 'callmark' uses (n_fft - hop)/2; 'none' leaves untouched.")
    ann.add_argument("--annotation-highlight", action="store_true", default=True,
                     help="Draw a colored bar above each annotation window.")
    ann.add_argument("--annotation-highlight-color", default="green")
    ann.add_argument("--annotation-highlight-lw", type=float, default=4.0)
    ann.add_argument("--annotation-highlight-y", type=float, default=1.02,
                     help="Vertical position in axes fraction units (1.0 is top).")

    scale = parser.add_argument_group("Scalebar")
    scale.add_argument("--scalebar-sec", type=float, default=0.5)
    scale.add_argument("--scalebar-pos", choices=["left", "right"], default="right")

    age = parser.add_argument_group("Age/Hatch Dates")
    age.add_argument("--hatch-dates-json", default=None, help="JSON file with bird->hatch_date mapping (serial date ints).")

    args = parser.parse_args()

    gap_spec = args.gap if args.gap is not None else tuple(sorted(args.gap_range))
    target_aspect = _parse_aspect(args.aspect)

    output_path = plot_folder(
        folder=args.folder, csv_name=args.csv_name, output=args.output, recursive=args.recursive,
        csv_suffix=args.csv_suffix, spec_type=args.spec_type, n_mels=args.n_mels, n_fft=args.n_fft,
        hop_length=args.hop_length, win_length=args.win_length, window=args.window, center=args.center,
        ref_db_max=args.ref_db_max, db_floor=args.db_floor, vmin=args.vmin, vmax=args.vmax,
        fmin=args.fmin, fmax=args.fmax, force_common_freq_bounds=args.force_common_freq_bounds,
        y_axis=args.y_axis, rows=args.rows, gap=gap_spec, seed=args.seed, time_zoom=args.time_zoom,
        target_aspect=target_aspect, target_sr=args.target_sr, sort_by=args.sort_by,
        sort_ascending=not args.sort_desc, title_mode=args.title_mode, title_color=args.title_color,
        title_bg=args.title_bg, show_titles=args.show_titles, plot_waveform=args.waveform,
        waveform_color=args.waveform_color, waveform_height_ratio=args.waveform_height_ratio,
        show_annotations=args.show_annotations, annotation_color=args.annotation_color,
        annotation_box_style=args.annotation_style, annotation_expand_mode=args.annotation_expand_mode,
        annotation_highlight=args.annotation_highlight,
        annotation_highlight_color=args.annotation_highlight_color,
        annotation_highlight_lw=args.annotation_highlight_lw,
        annotation_highlight_y=args.annotation_highlight_y,
        gap_color=args.gap_color, cmap=args.cmap, show_colorbar=args.show_colorbar,
        scalebar_sec=args.scalebar_sec, scalebar_pos=args.scalebar_pos, output_format=args.format,
        dpi=args.dpi, transparent=args.transparent, hatch_dates_file=args.hatch_dates_json
    )
    print(f"Plot successfully saved to: {output_path}")


if __name__ == "__main__":
    main()