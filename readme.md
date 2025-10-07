# VoCallBase Spectrogram Plotter

This script packs multiple spectrograms into a rectangular, time-true grid for comparative “gallery” figures with consistent time/frequency scales that can be used for the VoCallBase Guideline document, or any multi-spectrogram figure in general.

### Supports
- Linear or Mel spectrograms  
- Uniform STFT parameters across clips  
- Automatic or fixed row packing with gaps  
- Per-clip cropping via annotation CSVs  
- Optional waveform overlay per row  
- Annotations (brackets, frequency boxes, highlight bar)  
- Sorting/titles by filename or bird age (DPH)  
- Flexible aesthetics (colormap, background, colorbar, scale bar)  
- Reproducible layout via random seed  

---

## Key Features
- Waveform overlay per row (`--waveform`)
- Annotation highlight bar above each vocalization (`--annotation-highlight`)
- Extensive CLI toggles for spectrogram, layout, sorting, titles, and aesthetics
- “Callmark” timing mode for padding behavior used in CallMark
- Bird-age (DPH) sorting and titles from filenames
- Optional hatch-dates JSON for age computation

---

## Installation
Python ≥ 3.9 recommended.
```bash
pip install librosa numpy matplotlib pandas seaborn scikit-image colorcet cmocean
````

---

## Command-Line Usage

```bash
python sp.py /path/to/folder
```

Folder should contain WAV files and optionally a master annotation CSV.

### Output Options

| Option          | Description                         | Default |
| --------------- | ----------------------------------- | ------- |
| `--output`      | Output file path                    | None    |
| `--format`      | Output format (`png`, `svg`, `pdf`) | `svg`   |
| `--dpi`         | Figure resolution                   | `300`   |
| `--transparent` | Transparent background              | False   |

---

## Reproduce Zebrafinch Plot Style

```bash
python sp.py /path/to/folder \
  --format svg \
  --rows 9999 \
  --title-mode basename \
  --annotation-expand-mode callmark \
  --annotation-highlight \
  --annotation-highlight-color green \
  --annotation-highlight-lw 4
```

**Notes:**

* `--rows 9999`: one clip per row
* `--title-mode basename`: full filename titles
* `--annotation-expand-mode callmark`: callmark paper time padding
* `--annotation-highlight`: green bar above each annotation

---

## Waveform Overlay

Enable waveform above each row:

```bash
python sp.py /path/to/folder --waveform --waveform-color black --waveform-height-ratio 0.3
```

---

## Annotation Highlight Bar

```bash
--annotation-highlight
--annotation-highlight-color green
--annotation-highlight-lw 4.0
--annotation-highlight-y 1.02
```

Draws a colored bar at `y=1.02` (axes fraction).

---

## Spectrogram Parameters

| Option                         | Description            | Default       |
| ------------------------------ | ---------------------- | ------------- |
| `--spec-type`                  | `linear` or `mel`      | `linear`      |
| `--n-mels`                     | Mel bins (if mel)      | 128           |
| `--n-fft`                      | FFT window size        | 2048          |
| `--hop-length`                 | STFT hop               | `n_fft // 16` |
| `--win-length`                 | Window length          | None          |
| `--window`                     | Window function        | hann          |
| `--center/--no-center`         | Centered STFT          | center        |
| `--ref-db-max/--no-ref-db-max` | Reference to max power | on            |
| `--db-floor`                   | dB floor               | -80.0         |
| `--vmin`, `--vmax`             | dB bounds override     | None          |

---

## Frequency & Y-axis

| Option                       | Description                 |
| ---------------------------- | --------------------------- |
| `--fmin`, `--fmax`           | Frequency bounds            |
| `--force-common-freq-bounds` | Shared bounds across files  |
| `--y-axis`                   | `linear` or `mel` (display) |

---

## Layout & Gaps

| Option        | Description                   |
| ------------- | ----------------------------- |
| `--rows`      | Force number of rows          |
| `--gap`       | Fixed gap between clips       |
| `--gap-range` | Random gap range (s)          |
| `--seed`      | Random seed                   |
| `--time-zoom` | Horizontal scaling per second |
| `--aspect`    | Aspect ratio (e.g., `9:16`)   |

---

## Sorting & Titles

| Option         | Description                             |
| -------------- | --------------------------------------- |
| `--sort-by`    | `age`, `filename`, `duration`, `random` |
| `--sort-desc`  | Reverse sort                            |
| `--title-mode` | `bird`, `bird_dph`, `basename`, `none`  |
| `--no-titles`  | Hide all titles                         |

---

## Age / DPH Computation

Provide a JSON mapping bird IDs → hatch dates:

```json
{
  "R3280": 40793,
  "R3822": 41122,
  "R5018": 42142
}
```

Filename format: `R3280_40842_xxx.wav`
→ `DPH = 40842 - 40793`

Use:

```bash
--hatch-dates-json path/to/hatch_dates.json
```

---

## Annotation CSV

**Columns:**

* Required: `filename`, `onset`, `offset`
* Optional: `minFrequency`, `maxFrequency`, `label`

**Annotation Modes:**

| Option                     | Description                        |
| -------------------------- | ---------------------------------- |
| `--annotation-style`       | `auto`, `brackets`, `boxes`        |
| `--annotation-expand-mode` | `none`, `expand`, `callmark`       |
| `--annotation-highlight`   | Draw colored bar above annotations |

Matching behavior:

* Matches by `filename` substring (case-insensitive)
* Falls back to per-file CSV `<basename>_new.csv`

---

## Aesthetics

| Option                          | Description               | Default   |
| ------------------------------- | ------------------------- | --------- |
| `--cmap`                        | Colormap name             | `CET_L20` |
| `--gap-color`                   | Background color          | `white`   |
| `--show-colorbar/--no-colorbar` | Toggle colorbar           | on        |
| `--scalebar-sec`                | Scale bar length (s)      | 0.5       |
| `--scalebar-pos`                | Position (`left`/`right`) | right     |

---

## API Usage (Python)

```python
from sp import plot_folder

# Basic
plot_folder(folder="/path/to/folder", output_format="svg")

# Old style
plot_folder(
    folder="/path/to/folder",
    rows=9999,
    title_mode="basename",
    annotation_expand_mode="callmark",
    annotation_highlight=True,
    annotation_highlight_color="green",
    annotation_highlight_lw=4.0,
)

# With waveform & mel
plot_folder(
    folder="/path/to/folder",
    spec_type="mel",
    n_mels=128,
    plot_waveform=True,
    waveform_color="black",
)

# Age-based sorting
plot_folder(
    folder="/path/to/folder",
    sort_by="age",
    title_mode="bird_dph",
    hatch_dates_file="hatch_dates.json",
)
```

---

## Performance Tips

* Large `n_fft` = higher frequency resolution but slower
* `hop_length = n_fft // 16` is a good starting point
* Reduce `n_fft` or disable waveform for lower memory

---

## Troubleshooting

* If `librosa.load` fails: check WAV format (PCM)
* Missing annotations: verify CSV columns
* Layout issues: adjust `--time-zoom`, `--aspect`, or `--rows`
* Odd y-axis ticks: set `fmin`, `fmax` explicitly
