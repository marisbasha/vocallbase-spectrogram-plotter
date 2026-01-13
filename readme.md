# VoCallBase Spectrogram Plotter

Spectrogram plotter for annotated audio files. Takes a single audio file + CSV annotations and creates a multi-row spectrogram grid, grouping by any column(s) in the CSV and showing all annotations with frequency boxes.

## Quick Start

```bash
# Basic usage - group by individual and age, sorted by age
python3 sp_annotated.py ZF.wav --csv ZFVocalizations.csv --group-by individual,age --sort-by age

# Output to specific file
python3 sp_annotated.py ZF.wav --csv annotations.csv --group-by individual,age -o my_plot.png
```

## Example Files Used

The script was developed and tested with:
- **Audio**: `ZF.wav` - Zebra finch vocalizations (~820 seconds of audio at 44.1kHz)
- **Annotations**: `ZFVocalizations.csv` - CSV with columns:
  - `onset`, `offset` - time in seconds
  - `minFrequency`, `maxFrequency` - frequency bounds in Hz
  - `individual` - bird ID (e.g., R3277, R3406)
  - `age` - days post-hatch (e.g., 38, 99, 108)
  - `category` - Adults/Juveniles
  - `clustername` - vocalization type (e.g., "call")

## Parameters

### Grouping & Sorting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--group-by` | None | Column(s) to group by, comma-separated (e.g., `individual,age`) |
| `--sort-by` | None | Column to sort groups by (e.g., `age`, `onset`, `individual`) |
| `--sort-desc` | False | Sort descending instead of ascending |
| `--label-format` | Auto | Custom label format (e.g., `"{individual} - {age} days"`) |

### CSV Column Names

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--onset-col` | `onset` | Column name for onset time (seconds) |
| `--offset-col` | `offset` | Column name for offset time (seconds) |
| `--min-freq-col` | `minFrequency` | Column name for minimum frequency (Hz) |
| `--max-freq-col` | `maxFrequency` | Column name for maximum frequency (Hz) |
| `--filename-col` | `filename` | Column name for audio filename |

### Time & Display

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-duration` | 2.0 | Maximum duration per row (seconds) |
| `--padding` | 0.1 | Padding around annotations (seconds) |

### Spectrogram

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-fft` | 512 | FFT window size |
| `--hop-length` | 64 | Hop length for STFT |
| `--target-sr` | None | Resample to this sample rate |
| `--fmin` | 0.0 | Minimum frequency (Hz) |
| `--fmax` | None | Maximum frequency (Hz), defaults to 20kHz or Nyquist |
| `--db-floor` | -80.0 | dB floor for colormap |
| `--db-ceil` | 0.0 | dB ceiling for colormap |

### Annotation Display

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-boxes` | False | Hide frequency boxes |
| `--box-color` | white | Frequency box color |
| `--box-linewidth` | 0.5 | Box line width |
| `--box-linestyle` | `--` | Box line style |
| `--no-highlight` | False | Hide highlight bars above annotations |
| `--highlight-color` | green | Highlight bar color |
| `--highlight-linewidth` | 3.0 | Highlight line width |
| `--show-vlines` | False | Show vertical lines at onset/offset |
| `--vline-color` | white | Vertical line color |

### Titles

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-titles` | False | Hide row titles |
| `--title-color` | white | Title text color |
| `--title-bg` | black | Title background color |
| `--title-fontsize` | 11 | Title font size |

### Figure

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--row-height` | 2.5 | Height per row (inches) |
| `--fig-width` | 12.0 | Figure width (inches) |
| `--scalebar-sec` | 0.5 | Scale bar length (seconds), 0 to hide |
| `--scalebar-pos` | right | Scale bar position (`left` or `right`) |
| `--no-colorbar` | False | Hide colorbar |

### Style

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cmap` | CET_L20 | Colormap (matplotlib, colorcet, or cmocean) |
| `--gap-color` | white | Background color |

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output`, `-o` | Auto | Output file path |
| `--format` | png | Output format (`png`, `svg`, `pdf`) |
| `--dpi` | 150 | Output DPI |
| `--transparent` | False | Transparent background |

## Usage Examples

### Group by individual + age (zebra finch style)
```bash
python3 sp_annotated.py ZF.wav --csv ZFVocalizations.csv \
    --group-by individual,age --sort-by age
```
Output: Rows labeled "R3277 (99 dph)", "R3280 (38 dph)", etc.

### Group by category/cluster
```bash
python3 sp_annotated.py audio.wav --csv vocs.csv \
    --group-by clustername --sort-by clustername
```

### Group by species
```bash
python3 sp_annotated.py recording.wav --csv annotations.csv \
    --group-by species --sort-by onset
```

### Custom label format
```bash
python3 sp_annotated.py ZF.wav --csv ann.csv \
    --group-by individual,age \
    --label-format "{individual} at {age} days old"
```

### Different CSV column names
```bash
python3 sp_annotated.py audio.wav --csv data.csv \
    --onset-col start_time --offset-col end_time \
    --min-freq-col freq_low --max-freq-col freq_high
```

### High-resolution SVG output
```bash
python3 sp_annotated.py audio.wav --csv ann.csv \
    --group-by category --format svg --fig-width 16
```

### Show only highlight bars (no frequency boxes)
```bash
python3 sp_annotated.py audio.wav --csv ann.csv \
    --group-by individual --no-boxes
```

### Show vertical onset/offset lines
```bash
python3 sp_annotated.py audio.wav --csv ann.csv \
    --group-by category --show-vlines --no-boxes
```

### Adjust frequency range and duration
```bash
python3 sp_annotated.py audio.wav --csv ann.csv \
    --group-by individual --fmax 15000 --max-duration 3.0
```

### Different colormap
```bash
python3 sp_annotated.py audio.wav --csv ann.csv \
    --group-by category --cmap viridis
```

## CSV Format

The CSV should have at minimum `onset` and `offset` columns (in seconds). Optional columns:

```csv
onset,offset,minFrequency,maxFrequency,individual,age,category,filename
0.21,0.36,432,22050,R3277,99,Adults,ZF.wav
0.41,0.45,1384,20061,R3277,99,Adults,ZF.wav
```

- **onset/offset**: Required. Time in seconds.
- **minFrequency/maxFrequency**: Optional. Frequency bounds for boxes.
- **filename**: Optional. Used to filter annotations if CSV contains multiple files.
- **Any other columns**: Can be used for grouping/sorting.

## Dependencies

- numpy
- pandas
- matplotlib
- librosa
- colorcet
