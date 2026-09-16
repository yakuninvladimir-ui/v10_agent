"""Visual Channel Media generation: Pure Pillow rendering with pure-Python PNG fallback."""

from __future__ import annotations

import io
import struct
import zlib
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False
    Image = None
    ImageDraw = None
    ImageFont = None

from v10_agent.planning_set import PlanningSet
from v10_agent.types import Grid2D

# ARC-AGI-3 16-Color Palette
ARC_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # 0: Black
    1: (30, 136, 229),   # 1: Blue
    2: (211, 47, 47),    # 2: Red
    3: (56, 142, 60),    # 3: Green
    4: (251, 192, 45),   # 4: Yellow
    5: (117, 117, 117),  # 5: Grey
    6: (194, 24, 91),    # 6: Magenta
    7: (245, 124, 0),    # 7: Orange
    8: (0, 172, 193),    # 8: Light Blue
    9: (136, 14, 79),    # 9: Maroon
    10: (255, 255, 255), # 10: White (Mirror axis/neutral)
    11: (128, 216, 255), # 11: Cyan / Light Blue (Target sockets & UI)
    12: (255, 220, 0),   # 12: Bright Yellow
    13: (255, 133, 27),  # 13: Bright Orange
    14: (79, 204, 48),   # 14: Bright Green
    15: (163, 86, 214),  # 15: Purple
}

GRID_LINE_COLOR = (45, 45, 45)


def _encode_png(width: int, height: int, rgb_bytes: bytes) -> bytes:
    """Encode raw 24-bit RGB scanlines to standard PNG format without third-party dependencies."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        content = tag + data
        crc = zlib.crc32(content) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + content + struct.pack(">I", crc)

    stride = width * 3
    raw_scanlines = b"".join(
        b"\x00" + rgb_bytes[i * stride : (i + 1) * stride]
        for i in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw_scanlines, level=6))
        + _chunk(b"IEND", b"")
    )


def _render_grid_pure(grid: Grid2D, scale: int = 20) -> bytes:
    """Pure-Python standard-library grid-to-PNG renderer."""
    if not grid or not grid[0]:
        return _encode_png(32, 32, bytes(32 * 32 * 3))
    height = len(grid)
    width = len(grid[0])
    img_w = width * scale
    img_h = height * scale
    buf = bytearray(img_w * img_h * 3)

    for r in range(height):
        for c in range(width):
            color = ARC_COLOR_MAP.get(grid[r][c], (0, 0, 0))
            for py in range(r * scale, (r + 1) * scale):
                for px in range(c * scale, (c + 1) * scale):
                    idx = (py * img_w + px) * 3
                    if py == r * scale or py == (r + 1) * scale - 1 or px == c * scale or px == (c + 1) * scale - 1:
                        buf[idx : idx + 3] = GRID_LINE_COLOR
                    else:
                        buf[idx : idx + 3] = color

    return _encode_png(img_w, img_h, bytes(buf))


def _render_annotated_pure(grid: Grid2D, planning_set: PlanningSet, scale: int = 24) -> bytes:
    """Pure-Python standard-library annotated frame-to-PNG renderer."""
    if not grid or not grid[0]:
        return _encode_png(64, 64, bytes(64 * 64 * 3))
    height = len(grid)
    width = len(grid[0])
    img_w = width * scale
    img_h = height * scale
    buf = bytearray(img_w * img_h * 3)

    def set_px(x: int, y: int, color: tuple[int, int, int]):
        if 0 <= x < img_w and 0 <= y < img_h:
            idx = (y * img_w + x) * 3
            buf[idx : idx + 3] = color

    def draw_box(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 2):
        for w in range(width):
            for x in range(x0 + w, x1 - w + 1):
                set_px(x, y0 + w, color)
                set_px(x, y1 - w, color)
            for y in range(y0 + w, y1 - w + 1):
                set_px(x0 + w, y, color)
                set_px(x1 - w, y, color)

    # 1. Base grid drawing
    for r in range(height):
        for c in range(width):
            color = ARC_COLOR_MAP.get(grid[r][c], (0, 0, 0))
            for py in range(r * scale, (r + 1) * scale):
                for px in range(c * scale, (c + 1) * scale):
                    idx = (py * img_w + px) * 3
                    if py == r * scale or py == (r + 1) * scale - 1 or px == c * scale or px == (c + 1) * scale - 1:
                        buf[idx : idx + 3] = GRID_LINE_COLOR
                    else:
                        buf[idx : idx + 3] = color

    # 2. Annotate Planning Objects (bounding boxes and crosshairs)
    bbox_colors = [
        (255, 255, 255),  # White
        (0, 255, 255),    # Cyan
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 0),      # Lime
    ]

    for idx, obj in enumerate(planning_set.objects):
        outline_color = bbox_colors[idx % len(bbox_colors)]
        bx0 = obj.bbox.min_col * scale
        by0 = obj.bbox.min_row * scale
        bx1 = (obj.bbox.max_col + 1) * scale - 1
        by1 = (obj.bbox.max_row + 1) * scale - 1

        draw_box(bx0, by0, bx1, by1, outline_color, width=2)

        # Centroid crosshair
        cx = int(round(obj.centroid.col * scale + scale / 2))
        cy = int(round(obj.centroid.row * scale + scale / 2))
        arm = max(2, scale // 4)
        for d in range(-arm, arm + 1):
            set_px(cx + d, cy, outline_color)
            set_px(cx, cy + d, outline_color)

    return _encode_png(img_w, img_h, bytes(buf))


def render_grid_png(grid: Grid2D, scale: int = 20) -> bytes:
    """Render a clean 2D ARC grid to PNG bytes."""
    if _HAS_PIL and Image is not None and ImageDraw is not None:
        try:
            if not grid or not grid[0]:
                img = Image.new("RGB", (32, 32), color=(0, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            height = len(grid)
            width = len(grid[0])
            img_w = width * scale
            img_h = height * scale

            img = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            for r in range(height):
                for c in range(width):
                    color_idx = grid[r][c]
                    rgb = ARC_COLOR_MAP.get(color_idx, (0, 0, 0))
                    x0 = c * scale
                    y0 = r * scale
                    x1 = x0 + scale - 1
                    y1 = y0 + scale - 1
                    draw.rectangle([x0, y0, x1, y1], fill=rgb, outline=GRID_LINE_COLOR)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            pass
    return _render_grid_pure(grid, scale)


def render_annotated_frame_png(
    grid: Grid2D,
    planning_set: PlanningSet,
    scale: int = 24,
) -> bytes:
    """Render an annotated ARC frame highlighting planning objects with aliases and bounding boxes."""
    if _HAS_PIL and Image is not None and ImageDraw is not None and ImageFont is not None:
        try:
            if not grid or not grid[0]:
                img = Image.new("RGB", (64, 64), color=(0, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            height = len(grid)
            width = len(grid[0])
            img_w = width * scale
            img_h = height * scale

            img = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            # 1. Base grid drawing
            for r in range(height):
                for c in range(width):
                    color_idx = grid[r][c]
                    rgb = ARC_COLOR_MAP.get(color_idx, (0, 0, 0))
                    x0 = c * scale
                    y0 = r * scale
                    x1 = x0 + scale - 1
                    y1 = y0 + scale - 1
                    draw.rectangle([x0, y0, x1, y1], fill=rgb, outline=GRID_LINE_COLOR)

            # Use default bitmap font
            font = ImageFont.load_default()

            # 2. Annotate Planning Objects (bounding boxes and alias badges)
            bbox_colors = [
                (255, 255, 255),  # White
                (0, 255, 255),    # Cyan
                (255, 255, 0),    # Yellow
                (255, 0, 255),    # Magenta
                (0, 255, 0),      # Lime
            ]

            for idx, obj in enumerate(planning_set.objects):
                alias = planning_set.object_real_to_alias.get(obj.id, obj.id)
                outline_color = bbox_colors[idx % len(bbox_colors)]

                # Bounding box
                bx0 = obj.bbox.min_col * scale
                by0 = obj.bbox.min_row * scale
                bx1 = (obj.bbox.max_col + 1) * scale - 1
                by1 = (obj.bbox.max_row + 1) * scale - 1

                draw.rectangle([bx0, by0, bx1, by1], outline=outline_color, width=2)

                # Centroid crosshair
                cx = int(round(obj.centroid.col * scale + scale / 2))
                cy = int(round(obj.centroid.row * scale + scale / 2))
                arm = max(2, scale // 4)
                draw.line([cx - arm, cy, cx + arm, cy], fill=outline_color, width=2)
                draw.line([cx, cy - arm, cx, cy + arm], fill=outline_color, width=2)

                # Alias label badge
                badge_w = max(14, len(alias) * 8 + 4)
                badge_h = 12
                badge_x0 = bx0
                badge_y0 = max(0, by0 - badge_h)
                badge_x1 = badge_x0 + badge_w
                badge_y1 = badge_y0 + badge_h

                draw.rectangle([badge_x0, badge_y0, badge_x1, badge_y1], fill=(0, 0, 0), outline=outline_color)
                draw.text((badge_x0 + 2, badge_y0 + 1), alias, fill=outline_color, font=font)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            pass
    return _render_annotated_pure(grid, planning_set, scale)


def _render_dual_pure(grid: Grid2D, planning_set: PlanningSet, scale: int = 20) -> bytes:
    """Pure-Python standard-library dual frame renderer."""
    if not grid or not grid[0]:
        return _encode_png(64, 32, bytes(64 * 32 * 3))
    height = len(grid)
    width = len(grid[0])
    if max(height, width) >= 48:
        scale = min(scale, 10)
    elif max(height, width) >= 28:
        scale = min(scale, 14)

    panel_w = width * scale
    sep_w = 4
    img_w = panel_w * 2 + sep_w
    img_h = height * scale
    buf = bytearray(img_w * img_h * 3)

    def set_px(x: int, y: int, color: tuple[int, int, int]):
        if 0 <= x < img_w and 0 <= y < img_h:
            idx = (y * img_w + x) * 3
            buf[idx : idx + 3] = color

    def draw_box(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 2):
        for w in range(width):
            for x in range(x0 + w, x1 - w + 1):
                set_px(x, y0 + w, color)
                set_px(x, y1 - w, color)
            for y in range(y0 + w, y1 - w + 1):
                set_px(x0 + w, y, color)
                set_px(x1 - w, y, color)

    # 1. Base grid drawing (left panel = raw, right panel = annotated base)
    offset_x = panel_w + sep_w
    for r in range(height):
        for c in range(width):
            color = ARC_COLOR_MAP.get(grid[r][c], (0, 0, 0))
            for py in range(r * scale, (r + 1) * scale):
                for px in range(c * scale, (c + 1) * scale):
                    is_border = (
                        py == r * scale
                        or py == (r + 1) * scale - 1
                        or px == c * scale
                        or px == (c + 1) * scale - 1
                    )
                    cell_color = GRID_LINE_COLOR if is_border else color
                    set_px(px, py, cell_color)
                    set_px(offset_x + px, py, cell_color)

    # 2. Separator
    for py in range(img_h):
        for px in range(panel_w, panel_w + sep_w):
            set_px(px, py, (80, 80, 80))

    # 3. Annotations on right panel
    bbox_colors = [
        (255, 255, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 0),
        (255, 128, 0),
    ]
    for idx, obj in enumerate(planning_set.objects):
        outline_color = bbox_colors[idx % len(bbox_colors)]
        bx0 = offset_x + obj.bbox.min_col * scale
        by0 = obj.bbox.min_row * scale
        bx1 = offset_x + (obj.bbox.max_col + 1) * scale - 1
        by1 = (obj.bbox.max_row + 1) * scale - 1
        draw_box(bx0, by0, bx1, by1, outline_color, width=2)

        # Centroid crosshair
        cx = offset_x + int(round(obj.centroid.col * scale + scale / 2))
        cy = int(round(obj.centroid.row * scale + scale / 2))
        arm = max(2, scale // 4)
        for d in range(-arm, arm + 1):
            set_px(cx + d, cy, outline_color)
            set_px(cx, cy + d, outline_color)

    return _encode_png(img_w, img_h, bytes(buf))


def render_dual_frame_png(
    grid: Grid2D,
    planning_set: PlanningSet,
    scale: int = 24,
) -> bytes:
    """Render a side-by-side comparison: Raw Observed Grid (left) and ARGA Annotated Objects (right)."""
    if _HAS_PIL and Image is not None and ImageDraw is not None and ImageFont is not None:
        try:
            if not grid or not grid[0]:
                img = Image.new("RGB", (64, 32), color=(0, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            height = len(grid)
            width = len(grid[0])
            if max(height, width) >= 48:
                scale = min(scale, 10)
            elif max(height, width) >= 28:
                scale = min(scale, 14)

            panel_w = width * scale
            panel_h = height * scale
            header_h = 24
            sep_w = max(4, scale // 4)
            img_w = panel_w * 2 + sep_w
            img_h = panel_h + header_h

            img = Image.new("RGB", (img_w, img_h), color=(15, 15, 15))
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            # Headers
            draw.text((4, 6), "RAW OBSERVED GRID", fill=(220, 220, 220), font=font)
            offset_x = panel_w + sep_w
            draw.text((offset_x + 4, 6), "ARGA ANNOTATED OBJECTS & ROLES", fill=(220, 220, 220), font=font)

            # Separator
            draw.rectangle([panel_w, 0, panel_w + sep_w - 1, img_h - 1], fill=(50, 50, 50))
            draw.line([panel_w + sep_w // 2, 0, panel_w + sep_w // 2, img_h - 1], fill=(120, 120, 120), width=1)

            # 1. Draw grids for both panels
            for r in range(height):
                for c in range(width):
                    color_idx = grid[r][c]
                    rgb = ARC_COLOR_MAP.get(color_idx, (0, 0, 0))

                    # Left panel (Raw)
                    lx0 = c * scale
                    ly0 = header_h + r * scale
                    lx1 = lx0 + scale - 1
                    ly1 = ly0 + scale - 1
                    draw.rectangle([lx0, ly0, lx1, ly1], fill=rgb, outline=GRID_LINE_COLOR)

                    # Right panel (Annotated Base)
                    rx0 = offset_x + c * scale
                    ry0 = header_h + r * scale
                    rx1 = rx0 + scale - 1
                    ry1 = ry0 + scale - 1
                    draw.rectangle([rx0, ry0, rx1, ry1], fill=rgb, outline=GRID_LINE_COLOR)

            bbox_colors = [
                (255, 255, 255),  # White
                (0, 255, 255),    # Cyan
                (255, 255, 0),    # Yellow
                (255, 0, 255),    # Magenta
                (0, 255, 0),      # Lime
                (255, 128, 0),    # Orange
            ]

            for idx, obj in enumerate(planning_set.objects):
                alias = planning_set.object_real_to_alias.get(obj.id, obj.id)
                role = getattr(obj, "role", "")
                label_text = f"{alias}:{role[:4]}" if role and role != "generic_entity" else alias
                outline_color = bbox_colors[idx % len(bbox_colors)]

                bx0 = offset_x + obj.bbox.min_col * scale
                by0 = header_h + obj.bbox.min_row * scale
                bx1 = offset_x + (obj.bbox.max_col + 1) * scale - 1
                by1 = header_h + (obj.bbox.max_row + 1) * scale - 1

                draw.rectangle([bx0, by0, bx1, by1], outline=outline_color, width=2)

                # Centroid crosshair
                cx = offset_x + int(round(obj.centroid.col * scale + scale / 2))
                cy = header_h + int(round(obj.centroid.row * scale + scale / 2))
                arm = max(2, scale // 4)
                draw.line([cx - arm, cy, cx + arm, cy], fill=outline_color, width=2)
                draw.line([cx, cy - arm, cx, cy + arm], fill=outline_color, width=2)

                # Badge
                badge_w = max(14, len(label_text) * 7 + 4)
                badge_h = 12
                badge_x0 = bx0
                badge_y0 = max(header_h, by0 - badge_h)
                badge_x1 = badge_x0 + badge_w
                badge_y1 = badge_y0 + badge_h

                draw.rectangle([badge_x0, badge_y0, badge_x1, badge_y1], fill=(0, 0, 0), outline=outline_color)
                draw.text((badge_x0 + 2, badge_y0 + 1), label_text, fill=outline_color, font=font)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            pass
    return _render_dual_pure(grid, planning_set, scale)

