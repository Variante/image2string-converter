#!/usr/bin/env python3

import argparse
import sys
from math import ceil, sqrt

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


def extract_fullwidth_chars(ttf_path: str, reference_char: str = '你') -> str:
    """
    Extracts all characters from the font whose advance width matches that of the reference character.

    :param ttf_path: Path to the .ttf font file
    :param reference_char: A character used as the width reference (default: '你')
    :return: A sorted string of matching fullwidth characters, including ideographic space
    :raises: ValueError if the reference character is not found in the font
    """
    font = TTFont(ttf_path)
    cmap = font['cmap'].getBestCmap()
    hmtx = font['hmtx']

    ref_code = ord(reference_char)
    ref_glyph = cmap.get(ref_code)
    if not ref_glyph:
        raise ValueError(f"Reference character '{reference_char}' not found in font.")
    ref_width = hmtx[ref_glyph][0]

    fullwidth = []
    for code, glyph in cmap.items():
        try:
            if hmtx[glyph][0] == ref_width:
                fullwidth.append(chr(code))
        except KeyError:
            continue
    # Add ideographic space
    fullwidth.append('\u3000')
    return ''.join(sorted(set(fullwidth)))


def render_text_to_square_image(text: str,
                                font_path: str,
                                font_size: int = 48,
                                bg_color: str = "white",
                                text_color: str = "black",
                                test_char: str = '你') -> tuple[Image.Image, str]:
    """
    Renders the text into a nearly square grid image using the given font.

    :param text: The string of characters to render
    :param font_path: Path to the .ttf font file
    :param font_size: Font size for rendering
    :param bg_color: Background color
    :param text_color: Text color
    :param test_char: Character used to measure cell size
    :return: Tuple of (PIL Image, patch descriptor string "w{width}-h{height}")
    """
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(test_char)
    char_width, char_height = bbox[2], bbox[3]

    total = len(text)
    cols = ceil(sqrt(total))
    rows = ceil(total / cols)
    print(f"Rendering {rows} rows x {cols} cols...")

    img_w = cols * char_width
    img_h = rows * char_height
    image = Image.new("L", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(image)

    for idx, ch in enumerate(text):
        x = (idx % cols) * char_width
        y = (idx // cols) * char_height
        draw.text((x, y), ch, font=font, fill=text_color)

    # Optional: crop empty margins
    cropped = image.crop(image.getbbox()) if image.getbbox() else image
    return cropped, f"w{char_width}-h{char_height}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract fullwidth characters from a TTF and render them into an image grid."
    )
    parser.add_argument("ttf_file", help="Path to the .ttf font file to process")
    parser.add_argument(
        "-r", "--reference",
        default="你",
        help="Reference character for width matching (default: '你')"
    )
    parser.add_argument(
        "-s", "--font-size",
        type=int,
        default=12,
        help="Font size for rendering (default: 12)"
    )
    parser.add_argument(
        "-b", "--basename",
        default="full_charset",
        help="Base name for output files (default: full_charset)"
    )
    parser.add_argument(
        "--bg-color",
        default="white",
        help="Background color (default: white)"
    )
    parser.add_argument(
        "--text-color",
        default="black",
        help="Text color (default: black)"
    )
    args = parser.parse_args()

    try:
        chars = extract_fullwidth_chars(args.ttf_file, args.reference)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    txt_out = f"{args.basename}.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(chars)
    print(f"Saved {len(chars)} characters to {txt_out}")

    img, patch = render_text_to_square_image(
        chars,
        args.ttf_file,
        font_size=args.font_size,
        bg_color=args.bg_color,
        text_color=args.text_color,
        test_char=args.reference
    )
    img_out = f"{args.basename}-{patch}.png"
    img.save(img_out)
    print(f"Image saved to {img_out}")


if __name__ == "__main__":
    main()
