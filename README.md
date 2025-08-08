# Image2String

A command-line toolkit to convert images into string art by matching image patches against a font-based character template.

## Overview

This project consists of two main scripts:

1. prepare_charset.py

    Extracts all fullwidth characters from a specified TrueType font (.ttf).

    Renders these characters into a uniform grid image and outputs a text file listing the characters.

2. image2string_refactor.py

    Takes an input image and a charset template (image + text file) to reconstruct the image as a block of text.

    Supports auto gamma correction, scaling, and multithreaded patch matching for performance.


## Dependencies
```basjh
pip install opencv-python numpy pillow fonttools tqdm
```

## Usage

### 1. Prepare Charset Template

Generate a charset `.txt` and corresponding grid `.png` from your font:

```
python prepare_charset.py path/to/font.ttf \
  --font-size 12 \
  --reference 你 \
  --basename full_charset \
  --bg-color white \
  --text-color black
```

* **ttf_file**: Path to the TrueType font file.

* **--reference**: Reference character for width matching (default: `你`).

* **--font-size**: Size for rendering each glyph (default: 12).

* **--basename**: Base name for outputs (default: `full_charset`).

* **--bg-color**, **--text-color**: Colors for the image grid.

An example:
```bash
python prepare_charset.py yahei.ttf
```

Outputs:

* full_charset.txt — sorted list of extracted fullwidth characters.

* full_charset-w{W}-h{H}.png — image grid; {W} and {H} are patch width/height.

### 2. Convert Image to Text Art

Use the generated template to convert any image to string art:

```bash
python image2string.py \
  assets/prts.jpg \
  full_charset-w12-h15.png \
  full_charset.txt \
  -o output.txt \
  -i output.png \
  --font path/to/font.ttf \
  --font-size 32 \
  --padding 0 \
  --scale 2 \
  --gamma \
  --workers 16
```

An example:
```bash
python image2string.py .\assets\prts.jpg full_charset-w12-h15.png full_charset.txt
```

Arguments:

* **input.jpg**: Source image to convert.

* **template.png** & **charset.txt**: Outputs from prepare_charset.py.

* **-o/--out_text**: Path to save the output text file (default: output.txt).

* **-i/--out_img**: Path to save the rendered text image (default: output.png).

* **--font**, **--font-size**, **--padding**: Options for rendering the final text image.

* **--scale**: Grayscale upsampling factor (default: 4).

* **--gamma**: Enable automatic gamma correction.

* **--workers**: Number of threads for parallel matching (default: 8).

## License

Distributed under the MIT License.

[Result](assets/result.png)