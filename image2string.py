#!/usr/bin/env python3
"""
Refactored image2string script with a configurable main function.
Handles template-based OCR-like conversion of an input image into text using a character set image.
"""
import re
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import bisect
from collections import defaultdict
import argparse
import sys

def find_nearest(sorted_list: list[float], query: float) -> float:
    pos = bisect.bisect_left(sorted_list, query)
    if pos == 0:
        return sorted_list[0]
    if pos == len(sorted_list):
        return sorted_list[-1]
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    return before if abs(before - query) <= abs(after - query) else after


def render_text_to_image(
    text: str,
    font_path: str,
    font_size: int,
    padding: int,
    bg_color: int = 255,
    text_color: int = 0
) -> Image.Image:
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise RuntimeError(f"Cannot load font: {font_path}")
    lines = text.split('\n')
    rendered_lines = []
    max_width = 0
    line_height = 0
    for line in lines:
        chars = []
        width_sum = 0
        for ch in line:
            bbox = font.getbbox(ch)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            img = Image.new('L', (w + 2*padding, h + 2*padding), color=bg_color)
            draw = ImageDraw.Draw(img)
            draw.text((padding - bbox[0], padding - bbox[1]), ch, font=font, fill=text_color)
            chars.append(img)
            width_sum += img.width
            line_height = max(line_height, img.height)
        rendered_lines.append(chars)
        max_width = max(max_width, width_sum)
    total_height = line_height * len(rendered_lines)
    out_img = Image.new('L', (max_width, total_height), color=bg_color)
    y_off = 0
    for row in rendered_lines:
        x_off = 0
        for img in row:
            out_img.paste(img, (x_off, y_off))
            x_off += img.width
        y_off += line_height
    return out_img


def reconstruct_2d_matrix(results: list[tuple[tuple[int,int], str]]) -> str:
    coords = [coord for coord, _ in results]
    max_x = max(x for x, _ in coords)
    max_y = max(y for _, y in coords)
    matrix = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    for (x, y), ch in results:
        matrix[y][x] = ch
    return "\n".join("".join(row) for row in matrix)


def match_patch(
    patch: np.ndarray,
    coord: tuple[int,int],
    charset_values: list[float],
    charset_dict: dict[float, tuple[str, np.ndarray]],
    method=cv2.TM_CCOEFF_NORMED
) -> tuple[tuple[int,int], str]:
    avg_val = np.mean(patch)
    nearest = find_nearest(charset_values, avg_val)
    chars, tmpl = charset_dict[nearest]
    res = cv2.matchTemplate(tmpl, patch, method)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    # Determine character index by column
    patch_h, patch_w = patch.shape[:2]
    idx = min(max_loc[0] // patch_w, len(chars) - 1)
    return coord, chars[idx]


def match_patches_parallel(
    image: np.ndarray,
    charset_dict: dict[float, tuple[str, np.ndarray]],
    patch_size: tuple[int,int],
    max_workers: int = 8
) -> str:
    h, w = image.shape
    pw, ph = patch_size
    coords = [(x//pw, y//ph, x, y) for y in range(0, h-ph+1, ph) for x in range(0, w-pw+1, pw)]
    charset_values = sorted(charset_dict.keys())
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(match_patch, image[y:y+ph, x:x+pw], (cx, cy), charset_values, charset_dict): (cx,cy)
                   for cx, cy, x, y in coords}
        for f in tqdm(as_completed(futures), total=len(futures), desc='Matching'):
            results.append(f.result())
    return reconstruct_2d_matrix(results)


def auto_gamma_correction(gray: np.ndarray) -> float:
    mean = np.mean(gray) / 255.0
    return 1.0 if mean == 0 else (np.log(0.5) / np.log(mean))


def apply_gamma_correction(gray: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([(i/255.0)**inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def prepare_charset(
    charset_str: str,
    template: np.ndarray,
    patch_size: tuple[int,int]
) -> dict[float, tuple[str, np.ndarray]]:
    w, h = patch_size
    cols = template.shape[1] // w
    buckets = defaultdict(list)
    for idx, ch in enumerate(charset_str):
        row, col = divmod(idx, cols)
        y, x = row*h, col*w
        patch = template[y:y+h, x:x+w]
        buckets[float(np.mean(patch))].append((ch, patch))
    return {k: (''.join(c for c,_ in v), np.hstack([p for _,p in v])) for k, v in buckets.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert image to text using template charset.")
    parser.add_argument('image', help='Input image file')
    parser.add_argument('template', help='Charset template image')
    parser.add_argument('charset', help='Text file listing characters in template')
    parser.add_argument('-o', '--out_text', default='output.txt', help='Output text file')
    parser.add_argument('-i', '--out_img', default='output.png', help='Rendered text image')
    parser.add_argument('--font', default='yahei.ttf', help='Font file for rendering')
    parser.add_argument('--font_size', type=int, default=24)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--scale', type=float, default=2, help='Scale factor for grayscale upsample')
    parser.add_argument('--patch_w', type=int, default=None, help='Patch width')
    parser.add_argument('--patch_h', type=int, default=None, help='Patch height')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gamma', action='store_true', help='Apply auto gamma correction')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load charset
    with open(args.charset, encoding='utf8') as f:
        charset_str = f.read().rstrip('\n')

    # Determine patch size from template or args
    match = re.search(r'-w(\d+)-h(\d+)', args.template)
    if args.patch_w and args.patch_h:
        pw, ph = args.patch_w, args.patch_h
    elif match:
        pw, ph = int(match.group(1)), int(match.group(2))
    else:
        print("Error: Cannot determine patch size. Provide --patch_w and --patch_h.", file=sys.stderr)
        sys.exit(1)

    # Load template
    tmpl = cv2.imread(args.template, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        print(f"Error: Cannot read template {args.template}", file=sys.stderr)
        sys.exit(1)

    charset_dict = prepare_charset(charset_str, tmpl, (pw, ph))

    # Load and preprocess image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Cannot read image {args.image}", file=sys.stderr)
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.gamma:
        g = auto_gamma_correction(gray)
        gray = apply_gamma_correction(gray, g)
    gray = cv2.resize(gray, None, fx=args.scale, fy=args.scale)

    # Perform matching
    text = match_patches_parallel(gray, charset_dict, (pw, ph), args.workers)

    # Save text
    with open(args.out_text, 'w', encoding='utf8') as f:
        f.write(text)
    print(f"Text output saved to {args.out_text}")

    # Render text to image
    img_out = render_text_to_image(text, args.font, pw, args.padding)
    img_out.save(args.out_img)
    print(f"Rendered image saved to {args.out_img}")

if __name__ == '__main__':
    main()
