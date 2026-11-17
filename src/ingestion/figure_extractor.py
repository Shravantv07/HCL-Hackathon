"""
Figure and table extraction from PDFs.

For each page:
  - Detect embedded raster images using PyMuPDF's get_images().
  - Save each image to data/figures/{paper_id}_p{page}_fig{n}.png
  - Find the caption by scanning text blocks below the image bounding box.
  - Collect the nearby paragraph text (within 120pt above/below) as context.
"""

import pymupdf as fitz
import re
from pathlib import Path
from typing import List, Dict, Any


# Caption patterns: "Figure 1", "Fig. 2", "Table 3", etc.
_CAPTION_RE = re.compile(
    r"^(fig(ure)?\.?\s*\d+|table\s*\d+)", re.IGNORECASE
)

# Minimum image dimensions to skip tiny icons / decorations
MIN_WIDTH_PX = 80
MIN_HEIGHT_PX = 80


def _find_caption(page: fitz.Page, img_rect: fitz.Rect, margin: float = 120.0) -> str:
    """
    Search text blocks within `margin` points below (or above) the image rect.
    Return the first block that looks like a caption, or the closest text block.
    """
    candidates = []
    for block in page.get_text("blocks", sort=True):
        # block = (x0, y0, x1, y1, text, block_no, block_type)
        bx0, by0, bx1, by1, text, *_ = block
        block_rect = fitz.Rect(bx0, by0, bx1, by1)
        text = text.strip()
        if not text:
            continue

        # Check if block is below the image (within margin)
        below = by0 >= img_rect.y1 and by0 <= img_rect.y1 + margin
        # Also accept text just above in case the caption is above the figure
        above = by1 <= img_rect.y0 and by1 >= img_rect.y0 - margin / 2

        if below or above:
            candidates.append((abs(by0 - img_rect.y1), text))

    if not candidates:
        return ""

    # Prefer blocks that start with a caption keyword
    for _, text in sorted(candidates):
        if _CAPTION_RE.match(text):
            return text

    # Fall back to the closest block
    candidates.sort()
    return candidates[0][1]


def _nearby_text(page: fitz.Page, img_rect: fitz.Rect, margin: float = 150.0) -> str:
    """Collect all text blocks within `margin` points of the image."""
    parts = []
    for block in page.get_text("blocks", sort=True):
        bx0, by0, bx1, by1, text, *_ = block
        text = text.strip()
        if not text:
            continue
        block_rect = fitz.Rect(bx0, by0, bx1, by1)
        # Vertical proximity check
        if abs(by0 - img_rect.y1) <= margin or abs(img_rect.y0 - by1) <= margin:
            parts.append(text)
    return " ".join(parts)


def extract_figures(
    pdf_path: str,
    paper_id: str,
    figures_dir: str,
) -> List[Dict[str, Any]]:
    """
    Extract all embedded images from a PDF and pair each with its caption.

    Returns a list of figure dicts:
        {
            paper_id: str,
            page: int,
            fig_id: str,        # e.g. "p01_p3_fig1"
            image_path: str,    # absolute path to saved PNG
            caption: str,
            nearby_text: str,
            width: int,
            height: int,
        }
    """
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures: List[Dict[str, Any]] = []

    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            img_list = page.get_images(full=True)
            fig_counter = 0

            for img_info in img_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                width = base_image["width"]
                height = base_image["height"]

                if width < MIN_WIDTH_PX or height < MIN_HEIGHT_PX:
                    continue  # skip icons / decorations

                fig_counter += 1
                fig_id = f"{paper_id}_p{page_num}_fig{fig_counter}"
                img_ext = base_image["ext"]  # png, jpeg, etc.
                save_path = out_dir / f"{fig_id}.{img_ext}"

                with open(save_path, "wb") as f:
                    f.write(base_image["image"])

                # Find the image bounding box on the page
                # get_image_rects returns Rect objects directly in newer PyMuPDF
                try:
                    raw_rects = page.get_image_rects(xref)
                    img_rects = []
                    for item in raw_rects:
                        # item may be a Rect directly, or a tuple/list of 4 numbers
                        if isinstance(item, fitz.Rect):
                            img_rects.append(item)
                        else:
                            coords = item[1] if (isinstance(item, (list, tuple)) and len(item) > 1) else item
                            if hasattr(coords, '__len__') and len(coords) == 4:
                                img_rects.append(fitz.Rect(coords[0], coords[1], coords[2], coords[3]))
                    img_rect = img_rects[0] if img_rects else fitz.Rect(0, 0, width, height)
                except Exception:
                    img_rect = fitz.Rect(0, 0, width, height)

                caption = _find_caption(page, img_rect)
                nearby = _nearby_text(page, img_rect)

                figures.append(
                    {
                        "paper_id": paper_id,
                        "page": page_num,
                        "fig_id": fig_id,
                        "image_path": str(save_path.resolve()),
                        "caption": caption,
                        "nearby_text": nearby,
                        "width": width,
                        "height": height,
                    }
                )

    return figures


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 4:
        print("Usage: python figure_extractor.py <pdf_path> <paper_id> <figures_dir>")
        sys.exit(1)

    results = extract_figures(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"Extracted {len(results)} figures")
    for fig in results:
        print(f"  {fig['fig_id']} | p.{fig['page']} | {fig['width']}x{fig['height']} | caption: {fig['caption'][:60]!r}")
