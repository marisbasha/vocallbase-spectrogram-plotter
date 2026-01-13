#!/usr/bin/env python3
"""
Spectrogram plotter for annotated audio files.

Takes a single audio file + CSV annotations and creates a multi-row spectrogram grid,
grouping by any column(s) in the CSV and showing all annotations with frequency boxes.

Example usage:
    python sp_annotated.py ZF.wav --csv annotations.csv --group-by individual,age
    python sp_annotated.py audio.wav --csv vocs.csv --group-by category --sort-by onset
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import colorcet as cc
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "svg.fonttype": "none",
    "text.usetex": False,
    "figure.dpi": 150,
    "font.size": 12,
    "font.weight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
})


def resolve_cmap(cmap: str):
    """Resolve colormap by name from matplotlib or colorcet."""
    try:
        return matplotlib.colormaps.get_cmap(cmap)
    except Exception:
        pass
    try:
        return getattr(cc.cm, cmap)
    except Exception:
        pass
    return matplotlib.colormaps.get_cmap("viridis")


def detect_delimiter(csv_path: str) -> str:
    """Auto-detect CSV delimiter."""
    with open(csv_path, "r") as f:
        first_line = f.readline()
        return "\t" if "\t" in first_line else ","


def make_group_label(row: pd.Series, group_cols: List[str], label_format: Optional[str] = None) -> str:
    """
    Create a label for a group from the specified columns.

    If label_format is provided, use it as a format string.
    Otherwise, auto-format based on column names.
    """
    if label_format:
        return label_format.format(**row.to_dict())

    parts = []
    for col in group_cols:
        val = row[col]
        # Special formatting for known column types
        if col.lower() == "age":
            parts.append(f"({val} dph)")
        elif col.lower() in ("individual", "bird", "subject"):
            parts.append(str(val))
        elif col.lower() in ("category", "clustername", "class", "type"):
            parts.append(str(val))
        else:
            parts.append(str(val))

    return " ".join(parts)


def plot_annotated_spectrogram(
    audio_file: str,
    csv_file: str,
    output: str,
    *,
    # Grouping and sorting
    group_by: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    sort_ascending: bool = True,
    label_format: Optional[str] = None,
    # CSV column mapping
    onset_col: str = "onset",
    offset_col: str = "offset",
    min_freq_col: str = "minFrequency",
    max_freq_col: str = "maxFrequency",
    filename_col: str = "filename",
    # Time/display
    max_duration: float = 2.0,
    padding: float = 0.1,
    # Spectrogram
    n_fft: int = 512,
    hop_length: int = 64,
    target_sr: Optional[int] = None,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    db_floor: float = -80.0,
    db_ceil: float = 0.0,
    # Annotation boxes
    show_boxes: bool = True,
    box_color: str = "white",
    box_linewidth: float = 0.5,
    box_linestyle: str = "--",
    # Highlight bars
    show_highlight: bool = True,
    highlight_color: str = "green",
    highlight_linewidth: float = 3.0,
    highlight_y: float = 1.02,
    # Vertical lines (onset/offset markers)
    show_vlines: bool = False,
    vline_color: str = "white",
    vline_linewidth: float = 0.7,
    vline_linestyle: str = ":",
    # Title/labels
    title_color: str = "white",
    title_bg: str = "black",
    title_fontsize: float = 11,
    show_titles: bool = True,
    # Figure
    row_height: float = 2.5,
    fig_width: float = 12.0,
    # Scalebar
    scalebar_sec: float = 0.5,
    scalebar_pos: str = "right",
    # Colorbar
    show_colorbar: bool = True,
    # Style
    cmap: str = "CET_L20",
    gap_color: str = "white",
    # Output
    output_format: str = "png",
    dpi: int = 150,
    transparent: bool = False,
) -> str:
    """
    Create a multi-row spectrogram plot from an annotated audio file.

    Parameters
    ----------
    audio_file : str
        Path to the audio file (.wav)
    csv_file : str
        Path to the CSV annotation file
    output : str
        Output file path
    group_by : list of str, optional
        Column(s) to group annotations by. Each unique combination gets its own row.
        Example: ["individual", "age"] creates rows like "R3277 (99 dph)"
    sort_by : str, optional
        Column to sort groups by. Use "onset" to sort by first appearance,
        or any column name like "age", "individual", etc.
    sort_ascending : bool
        Sort direction (default True = ascending)
    label_format : str, optional
        Custom format string for row labels. Use {column_name} placeholders.
        Example: "{individual} - {age} days" -> "R3277 - 99 days"
    onset_col, offset_col : str
        Column names for onset/offset times (in seconds)
    min_freq_col, max_freq_col : str
        Column names for frequency bounds (in Hz)
    filename_col : str
        Column name for filename (used to filter if CSV has multiple files)
    max_duration : float
        Maximum duration per row in seconds
    padding : float
        Padding around annotations in seconds
    n_fft : int
        FFT window size
    hop_length : int
        Hop length for STFT
    target_sr : int, optional
        Resample audio to this sample rate
    fmin, fmax : float
        Frequency display range (Hz). fmax defaults to Nyquist/2.
    db_floor, db_ceil : float
        dB range for colormap
    show_boxes : bool
        Draw frequency boxes around annotations
    show_highlight : bool
        Draw colored bars above annotations
    show_vlines : bool
        Draw vertical lines at onset/offset times
    show_titles : bool
        Show group labels above each row
    scalebar_sec : float
        Length of scale bar in seconds (0 to hide)
    show_colorbar : bool
        Show colorbar
    cmap : str
        Colormap name (matplotlib, colorcet, or cmocean)
    output_format : str
        Output format (png, svg, pdf)
    dpi : int
        Output DPI
    transparent : bool
        Transparent background

    Returns
    -------
    str
        Path to the saved output file
    """
    # Load CSV
    delimiter = detect_delimiter(csv_file)
    df = pd.read_csv(csv_file, sep=delimiter)

    # Check required columns
    required = [onset_col, offset_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Available: {list(df.columns)}")

    # Filter by filename if the column exists and audio file matches
    audio_basename = os.path.basename(audio_file)
    if filename_col in df.columns:
        # Keep rows where filename matches (case-insensitive, partial match)
        mask = df[filename_col].astype(str).str.lower().str.contains(audio_basename.lower(), regex=False)
        if mask.any():
            df = df[mask].copy()

    # Load audio
    y_full, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    if target_sr is None:
        target_sr = sr

    # Default fmax to reasonable value
    if fmax is None:
        fmax = min(20000, target_sr // 2)

    # Determine grouping
    if group_by is None:
        # No grouping - show all annotations in one row
        df["_group"] = "all"
        group_cols = ["_group"]
    else:
        group_cols = group_by
        for col in group_cols:
            if col not in df.columns:
                raise ValueError(f"Group column '{col}' not found in CSV. Available: {list(df.columns)}")

    # Create group labels
    group_key_cols = group_cols.copy()

    # Get unique groups with their first occurrence info for sorting
    groups_info = []
    for _, group_df in df.groupby(group_cols, sort=False):
        first_row = group_df.iloc[0]
        label = make_group_label(first_row, group_cols, label_format)
        groups_info.append({
            "label": label,
            "df": group_df,
            "first_onset": group_df[onset_col].min(),
            **{col: first_row[col] for col in group_cols}
        })

    # Sort groups
    if sort_by:
        if sort_by == "onset":
            groups_info.sort(key=lambda x: x["first_onset"], reverse=not sort_ascending)
        elif sort_by in df.columns:
            # Try numeric sort first, fall back to string
            def sort_key(x):
                val = x.get(sort_by, x["label"])
                try:
                    return (0, float(val))
                except (ValueError, TypeError):
                    return (1, str(val))
            groups_info.sort(key=sort_key, reverse=not sort_ascending)
        else:
            groups_info.sort(key=lambda x: x["label"], reverse=not sort_ascending)

    n_rows = len(groups_info)
    if n_rows == 0:
        raise ValueError("No groups found in CSV")

    print(f"Plotting {n_rows} groups")

    # Figure setup
    fig_height = max(4.0, row_height * n_rows)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=1,
        figsize=(fig_width, fig_height),
        sharey=True, squeeze=False, dpi=dpi
    )
    axes = axes.flatten()
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.35)

    cmap_resolved = resolve_cmap(cmap)

    # Check if we have frequency columns
    has_freq = min_freq_col in df.columns and max_freq_col in df.columns

    for i, group_info in enumerate(groups_info):
        ax = axes[i]
        ax.set_facecolor(gap_color)
        for spine in ax.spines.values():
            spine.set_visible(False)

        group_df = group_info["df"]
        label = group_info["label"]

        # Find time range for this group
        chunk_start = group_df[onset_col].min() - padding
        chunk_end = group_df[offset_col].max() + padding

        # Limit duration
        if chunk_end - chunk_start > max_duration:
            chunk_end = chunk_start + max_duration
            # Filter annotations to this range
            group_df = group_df[
                (group_df[onset_col] >= chunk_start - padding) &
                (group_df[offset_col] <= chunk_end + padding)
            ]

        chunk_start = max(0, chunk_start)

        # Extract audio chunk
        start_sample = int(chunk_start * target_sr)
        end_sample = int(chunk_end * target_sr)
        y_chunk = y_full[start_sample:end_sample]

        if len(y_chunk) < n_fft:
            print(f"Warning: Chunk too short for group '{label}', skipping")
            continue

        chunk_duration = len(y_chunk) / target_sr

        # Compute spectrogram
        S = librosa.stft(y_chunk, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # Display spectrogram
        librosa.display.specshow(
            S_db, ax=ax, sr=target_sr, hop_length=hop_length,
            x_axis="time", y_axis="linear", fmin=fmin, fmax=fmax,
            cmap=cmap_resolved, vmin=db_floor, vmax=db_ceil
        )

        ax.set_xlim(0, chunk_duration)
        ax.set_ylim(fmin, fmax)

        # Draw annotations
        for _, ann in group_df.iterrows():
            rel_onset = ann[onset_col] - chunk_start
            rel_offset = ann[offset_col] - chunk_start

            # Skip if outside visible range
            if rel_offset < 0 or rel_onset > chunk_duration:
                continue

            # Clip to visible range
            rel_onset = max(0, rel_onset)
            rel_offset = min(chunk_duration, rel_offset)

            # Frequency boxes
            if show_boxes and has_freq:
                min_freq = ann[min_freq_col]
                max_freq = ann[max_freq_col]
                if pd.notna(min_freq) and pd.notna(max_freq):
                    rect_x = [rel_onset, rel_offset, rel_offset, rel_onset, rel_onset]
                    rect_y = [min_freq, min_freq, max_freq, max_freq, min_freq]
                    ax.plot(rect_x, rect_y, color=box_color,
                           linewidth=box_linewidth, linestyle=box_linestyle)

            # Vertical lines
            if show_vlines:
                ax.axvline(rel_onset, color=vline_color,
                          linewidth=vline_linewidth, linestyle=vline_linestyle)
                ax.axvline(rel_offset, color=vline_color,
                          linewidth=vline_linewidth, linestyle=vline_linestyle)

            # Highlight bar
            if show_highlight:
                ax.plot(
                    [rel_onset, rel_offset], [highlight_y, highlight_y],
                    color=highlight_color, linewidth=highlight_linewidth,
                    solid_capstyle="butt",
                    transform=ax.get_xaxis_transform(), clip_on=False
                )

        # Y-axis ticks (in kHz)
        if fmax <= 25000:
            tick_khz = [0, 5, 10, 20]
        else:
            tick_khz = [0, 10, 20, 40]
        tick_hz = [t * 1000 for t in tick_khz if fmin <= t * 1000 <= fmax]
        ax.set_yticks(tick_hz)
        ax.set_yticklabels([str(t // 1000) for t in tick_hz])
        ax.set_ylabel(None)
        ax.set_xticks([])
        ax.set_xlabel(None)

        # Title
        if show_titles and label != "all":
            ax.text(
                0, 1.06, label,
                fontsize=title_fontsize, fontweight="bold",
                ha="left", va="bottom", color=title_color,
                bbox=dict(facecolor=title_bg, pad=2),
                transform=ax.get_xaxis_transform(),
                clip_on=False
            )

    # Y-axis label
    fig.supylabel("Frequency (kHz)", fontsize=12, fontweight="bold", x=0.01)

    # Scalebar
    if scalebar_sec > 0:
        last_ax = axes[-1]
        transform = last_ax.get_xaxis_transform()
        y_pos = -0.18
        x_end = last_ax.get_xlim()[1]
        if scalebar_pos == "right":
            x_start = x_end - scalebar_sec
        else:
            x_start = 0
            x_end = scalebar_sec
        last_ax.plot([x_start, x_end], [y_pos, y_pos], color="black", lw=3,
                     transform=transform, clip_on=False, solid_capstyle="butt")
        last_ax.text((x_start + x_end) / 2, y_pos - 0.06, f"{scalebar_sec} s",
                     ha="center", va="top", fontsize=10, fontweight="bold",
                     transform=transform, clip_on=False)

    # Colorbar
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap_resolved,
                                    norm=plt.Normalize(vmin=db_floor, vmax=db_ceil))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                           fraction=0.015, pad=0.04)
        cbar.set_label("Power (dB)", fontsize=10, fontweight="bold")

    # Save
    plt.savefig(output, format=output_format, bbox_inches="tight",
                dpi=dpi, transparent=transparent)
    plt.close()

    print(f"Saved: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Plot spectrograms from annotated audio files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("audio", help="Audio file path (.wav)")
    parser.add_argument("--csv", required=True, help="CSV annotation file path")

    # Grouping/sorting
    group = parser.add_argument_group("Grouping & Sorting")
    group.add_argument("--group-by", type=str, default=None,
                       help="Column(s) to group by, comma-separated (e.g., 'individual,age')")
    group.add_argument("--sort-by", type=str, default=None,
                       help="Column to sort groups by (e.g., 'age', 'onset', 'individual')")
    group.add_argument("--sort-desc", action="store_true",
                       help="Sort descending instead of ascending")
    group.add_argument("--label-format", type=str, default=None,
                       help="Custom label format string (e.g., '{individual} - {age} days')")

    # CSV columns
    cols = parser.add_argument_group("CSV Column Names")
    cols.add_argument("--onset-col", default="onset", help="Onset time column name")
    cols.add_argument("--offset-col", default="offset", help="Offset time column name")
    cols.add_argument("--min-freq-col", default="minFrequency", help="Min frequency column name")
    cols.add_argument("--max-freq-col", default="maxFrequency", help="Max frequency column name")
    cols.add_argument("--filename-col", default="filename", help="Filename column name")

    # Time/display
    time = parser.add_argument_group("Time & Display")
    time.add_argument("--max-duration", type=float, default=2.0,
                      help="Maximum duration per row (seconds)")
    time.add_argument("--padding", type=float, default=0.1,
                      help="Padding around annotations (seconds)")

    # Spectrogram
    spec = parser.add_argument_group("Spectrogram")
    spec.add_argument("--n-fft", type=int, default=512, help="FFT window size")
    spec.add_argument("--hop-length", type=int, default=64, help="Hop length")
    spec.add_argument("--target-sr", type=int, default=None, help="Resample to this sample rate")
    spec.add_argument("--fmin", type=float, default=0.0, help="Minimum frequency (Hz)")
    spec.add_argument("--fmax", type=float, default=None, help="Maximum frequency (Hz)")
    spec.add_argument("--db-floor", type=float, default=-80.0, help="dB floor")
    spec.add_argument("--db-ceil", type=float, default=0.0, help="dB ceiling")

    # Annotations
    ann = parser.add_argument_group("Annotation Display")
    ann.add_argument("--no-boxes", dest="show_boxes", action="store_false",
                     help="Hide frequency boxes")
    ann.add_argument("--box-color", default="white", help="Frequency box color")
    ann.add_argument("--box-linewidth", type=float, default=0.5, help="Box line width")
    ann.add_argument("--box-linestyle", default="--", help="Box line style")
    ann.add_argument("--no-highlight", dest="show_highlight", action="store_false",
                     help="Hide highlight bars")
    ann.add_argument("--highlight-color", default="green", help="Highlight bar color")
    ann.add_argument("--highlight-linewidth", type=float, default=3.0, help="Highlight line width")
    ann.add_argument("--show-vlines", action="store_true",
                     help="Show vertical lines at onset/offset")
    ann.add_argument("--vline-color", default="white", help="Vertical line color")

    # Titles
    titles = parser.add_argument_group("Titles")
    titles.add_argument("--no-titles", dest="show_titles", action="store_false",
                        help="Hide row titles")
    titles.add_argument("--title-color", default="white", help="Title text color")
    titles.add_argument("--title-bg", default="black", help="Title background color")
    titles.add_argument("--title-fontsize", type=float, default=11, help="Title font size")

    # Figure
    fig = parser.add_argument_group("Figure")
    fig.add_argument("--row-height", type=float, default=2.5, help="Height per row (inches)")
    fig.add_argument("--fig-width", type=float, default=12.0, help="Figure width (inches)")
    fig.add_argument("--scalebar-sec", type=float, default=0.5,
                     help="Scale bar length in seconds (0 to hide)")
    fig.add_argument("--scalebar-pos", choices=["left", "right"], default="right",
                     help="Scale bar position")
    fig.add_argument("--no-colorbar", dest="show_colorbar", action="store_false",
                     help="Hide colorbar")

    # Style
    style = parser.add_argument_group("Style")
    style.add_argument("--cmap", default="CET_L20", help="Colormap name")
    style.add_argument("--gap-color", default="white", help="Background color")

    # Output
    out = parser.add_argument_group("Output")
    out.add_argument("--output", "-o", default=None, help="Output file path")
    out.add_argument("--format", default="png", help="Output format (png, svg, pdf)")
    out.add_argument("--dpi", type=int, default=150, help="Output DPI")
    out.add_argument("--transparent", action="store_true", help="Transparent background")

    args = parser.parse_args()

    # Parse group-by
    group_by = None
    if args.group_by:
        group_by = [col.strip() for col in args.group_by.split(",")]

    # Default output path
    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.audio))[0]
        output = f"{base}_annotated.{args.format}"

    plot_annotated_spectrogram(
        audio_file=args.audio,
        csv_file=args.csv,
        output=output,
        group_by=group_by,
        sort_by=args.sort_by,
        sort_ascending=not args.sort_desc,
        label_format=args.label_format,
        onset_col=args.onset_col,
        offset_col=args.offset_col,
        min_freq_col=args.min_freq_col,
        max_freq_col=args.max_freq_col,
        filename_col=args.filename_col,
        max_duration=args.max_duration,
        padding=args.padding,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        target_sr=args.target_sr,
        fmin=args.fmin,
        fmax=args.fmax,
        db_floor=args.db_floor,
        db_ceil=args.db_ceil,
        show_boxes=args.show_boxes,
        box_color=args.box_color,
        box_linewidth=args.box_linewidth,
        box_linestyle=args.box_linestyle,
        show_highlight=args.show_highlight,
        highlight_color=args.highlight_color,
        highlight_linewidth=args.highlight_linewidth,
        show_vlines=args.show_vlines,
        vline_color=args.vline_color,
        show_titles=args.show_titles,
        title_color=args.title_color,
        title_bg=args.title_bg,
        title_fontsize=args.title_fontsize,
        row_height=args.row_height,
        fig_width=args.fig_width,
        scalebar_sec=args.scalebar_sec,
        scalebar_pos=args.scalebar_pos,
        show_colorbar=args.show_colorbar,
        cmap=args.cmap,
        gap_color=args.gap_color,
        output_format=args.format,
        dpi=args.dpi,
        transparent=args.transparent,
    )


if __name__ == "__main__":
    main()
