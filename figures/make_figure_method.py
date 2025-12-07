import os
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pympi
import matplotlib.patches as patches
import sys
import textwrap
import colorsys
import hashlib

def extract_exact_frames(video_file, out_dir, start=60, end=90, interval=2, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)

    # Check video dimensions
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0', video_file
    ]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        width, height = map(int, result.stdout.strip().split('x'))
    except Exception:
        width, height = None, None

    crop_filter = None
    if width and height and width > height:
        # Center crop to square (width=height)
        crop_filter = f"crop=ih:ih:(iw-oh)/2:0"

    num_frames = int((end - start) / interval)
    midpoints = start + (np.arange(num_frames) + 0.5) * interval  # exact middle frames

    for idx, t in enumerate(midpoints, 1):
        out_path = os.path.join(out_dir, f'frame_{idx:02d}.png')
        if not overwrite and os.path.exists(out_path):
            print(f"Skipping {out_path} (exists, overwrite=False)")
            continue

        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = t % 60
        t_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        cmd = ['ffmpeg']
        if overwrite:
            cmd.append('-y')
        cmd += [
            '-hide_banner', '-loglevel', 'error',
            '-i', video_file,
            '-ss', t_str
        ]
        if crop_filter:
            cmd += ['-vf', crop_filter]
        cmd += [
            '-frames:v', '1',
            out_path
        ]

        print(f"Extracting frame {idx}: {t_str} -> {out_path}")
        subprocess.run(cmd, check=True)

def load_elan_segments(eaf_path, start, end, tiers):
    eaf = pympi.Elan.Eaf(eaf_path)
    tier_data = {tier: [] for tier in tiers}
    for tier in tiers:
        if tier not in eaf.get_tier_names():
            continue
        for ann in eaf.get_annotation_data_for_tier(tier):
            seg_start, seg_end, text = ann
            seg_start /= 1000.0
            seg_end /= 1000.0
            if seg_end <= start or seg_start >= end:
                continue
            shown_text = text
            if tier == "CSLR" and text and "/" in text:
                parts = text.split("/")
                if len(parts) > 1 and parts[1].strip():
                    shown_text = parts[1].strip()
                else:
                    shown_text = parts[0].strip()
            clipped_start = max(seg_start, start)
            clipped_end = min(seg_end, end)
            tier_data[tier].append((clipped_start, clipped_end, shown_text))
    return tier_data

def wrap_text(text, max_width_px, font_size=10, px_per_char=7, max_lines=2):
    if not text:
        return ''
    max_chars = max(1, int(max_width_px / px_per_char))
    wrapped = textwrap.wrap(text, width=max_chars, break_long_words=True, replace_whitespace=False)
    if len(wrapped) > max_lines:
        visible = wrapped[:max_lines]
        max_last = max(1, max_chars - 1)
        visible[-1] = visible[-1][:max_last] + '…'
        return '\n'.join(visible)
    else:
        return '\n'.join(wrapped)

# New: per-subtitle color assignment and interval helpers

def assign_subtitle_color(text: str):
    """Return a hex color for a subtitle sentence, or None if not matched.
    - Warm color for sentence 1
    - Cold color for sentence 2
    - Distinct color for the short phrase (Chris Davis')
    """
    if not text:
        return None
    s1 = "Reptiles and amphibians are linked with dark, dank places, warts, witchcraft and sliminess."
    s2 = "Many people loathe or even hate them, but a few stalwart individuals love them."
    s3 = "Chris Davis'"
    lt = text.strip().lower()
    if s1.lower() in lt:
        return '#f39c6b'  # warm
    if s2.lower() in lt:
        return '#6bb8ff'  # cold
    if s3.lower() in lt:
        return '#c77dff'  # distinct
    return None


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def pick_subtitle_color_for_interval(sub_intervals, start: float, end: float):
    """Pick the color of the subtitle interval with the largest overlap with [start, end].
    sub_intervals: List of (s0, s1, color)
    Returns color or None.
    """
    best_color = None
    best_ov = 0.0
    for s0, s1, col in sub_intervals:
        ov = interval_overlap(start, end, s0, s1)
        if ov > best_ov:
            best_ov = ov
            best_color = col
    return best_color

# Slightly vary a base hex color in a stable way per segment, keeping it similar

def _hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    r, g, b = (max(0, min(1, c)) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))


def color_variant(base_hex: str, key: str) -> str:
    """Return a small variant of base_hex, stable for a given key.
    Keeps hue/sat similar, but with a bit more distinction.
    """
    h = hashlib.md5(key.encode('utf-8')).digest()
    t0 = h[0] / 255.0
    t1 = h[1] / 255.0
    t2 = h[2] / 255.0

    r, g, b = _hex_to_rgb(base_hex)
    h_, l_, s_ = colorsys.rgb_to_hls(r, g, b)

    # Lightness within ±20%
    l_factor = 1.0 + 0.4 * (t0 - 0.5)  # 0.8..1.2
    l_new = max(0.0, min(1.0, l_ * l_factor))

    # Hue nudge within ±8°
    h_nudge = (t1 - 0.5) * (16.0 / 360.0)
    h_new = (h_ + h_nudge) % 1.0

    # Saturation within ±10%
    s_factor = 1.0 + 0.2 * (t2 - 0.5)  # 0.9..1.1
    s_new = max(0.0, min(1.0, s_ * s_factor))

    r2, g2, b2 = colorsys.hls_to_rgb(h_new, l_new, s_new)
    return _rgb_to_hex((r2, g2, b2))

def open_pdf(path):
    if sys.platform.startswith('darwin'):
        subprocess.run(['open', path])
    elif sys.platform.startswith('linux'):
        subprocess.run(['xdg-open', path])
    elif sys.platform.startswith('win'):
        opener = getattr(os, 'startfile', None)
        if opener:
            opener(path)
        else:
            subprocess.run(['cmd', '/c', 'start', '', path])
    else:
        print(f"Please open {path} manually.")

def compute_y_positions_top_down(
    num_tiers, tier_height, frame_height,
    big_space_mode, has_sign_and_subt_d,
    small_spacing=6, large_spacing=None,
    override_top_pad=None, override_spacings=None
):
    """
    Build y-positions from top (just below frames) downward,
    ensuring row 0 (Sign) is the topmost tier.
    Returns:
      y_positions: list of TOP y for each tier (negative values)
      y_tiers_top_of_last: TOP y of the final (bottom-most) tier
      top_pad: padding from bottom of frames to top of the first tier
      spacings: list of spacings between rows (len=num_tiers-1)
    """
    if large_spacing is None:
        large_spacing = int(1.5 * tier_height)

    # Determine base top pad and spacings
    if override_top_pad is not None:
        top_pad = override_top_pad
    else:
        if big_space_mode:
            top_pad = large_spacing
        else:
            top_pad = small_spacing

    if override_spacings is not None and len(override_spacings) == max(0, num_tiers - 1):
        spacings = list(override_spacings)
    else:
        if big_space_mode:
            spacings = [large_spacing] * (num_tiers - 1)
        else:
            spacings = [small_spacing] * (num_tiers - 1)

    # Compute top-down y positions (negative = below the frames)
    y_positions = []
    y_top = -(top_pad + tier_height)  # top of first row (Sign)
    y_positions.append(y_top)
    y_cursor = y_top
    for i in range(1, num_tiers):
        y_cursor = y_cursor - spacings[i - 1] - tier_height
        y_positions.append(y_cursor)

    y_tiers_top_of_last = y_positions[-1]
    return y_positions, y_tiers_top_of_last, top_pad, spacings


def _draw_process_box(ax, x_center, y_center, text, width, height,
                      facecolor='#ffffff', edgecolor='#555', textcolor='#333', z=25):
    rect = patches.FancyBboxPatch(
        (x_center - width/2, y_center - height/2), width, height,
        boxstyle='round,pad=0.02,rounding_size=6', linewidth=1.2,
        edgecolor=edgecolor, facecolor=facecolor, zorder=z
    )
    ax.add_patch(rect)
    ax.text(x_center, y_center, text, ha='center', va='center', fontsize=12,
            color=textcolor, zorder=z+1, fontweight='medium')


def _draw_vertical_arrow(ax, x, y_from, y_to, color='#555', lw=1.8, z=26, shorten=True):
    # Optionally shorten the arrow length (keep midpoint fixed)
    if shorten:
        L = y_to - y_from
        new_L = 0.7 * L  # reduce length by 30%
        mid = (y_from + y_to) / 2.0
        y_from = mid - new_L / 2.0
        y_to = mid + new_L / 2.0
    ax.annotate(
        '', xy=(x, y_to), xytext=(x, y_from), zorder=z,
        arrowprops=dict(
            arrowstyle='-|>',
            lw=lw,
            color=color,
            mutation_scale=14,
            shrinkA=0,
            shrinkB=0,
            capstyle='round',
        )
    )

def _draw_curved_arrow(ax, x_from, y_from, x_to, y_to, color='#777', lw=2.0, rad=0.25, z=26, shorten=True):
    # Shorten by pulling endpoints toward midpoint by 15% each (overall 70% length)
    if shorten:
        mx = (x_from + x_to) / 2.0
        my = (y_from + y_to) / 2.0
        x_from = mx + (x_from - mx) * 0.7
        y_from = my + (y_from - my) * 0.7
        x_to = mx + (x_to - mx) * 0.7
        y_to = my + (y_to - my) * 0.7
    ax.annotate(
        '', xy=(x_to, y_to), xytext=(x_from, y_from), zorder=z,
        arrowprops=dict(
            arrowstyle='-|>',
            lw=lw,
            color=color,
            mutation_scale=14,
            connectionstyle=f"arc3,rad={rad}",
            capstyle='round',
        )
    )

def plot_timeline_with_tiers(frame_dir, num_frames, start_time, interval, output_pdf, frame_height, tier_data,
                             include_sign_gt=False, include_subtitle_gt=True, layout: str = 'default'):
    frames = []
    widths = []
    for i in range(1, num_frames + 1):
        frame_path = os.path.join(frame_dir, f'frame_{i:02d}.png')
        if not os.path.exists(frame_path):
            print(f"Warning: {frame_path} not found, stopping at frame {i-1}")
            break
        img = Image.open(frame_path)
        aspect = img.width / img.height
        img = img.resize((int(frame_height * aspect), frame_height))
        frames.append(img)
        widths.append(img.width)
    if not frames:
        print("No frames loaded.")
        return

    total_width = sum(widths)

    # Neutral colors for non-embedded rows
    neutral_face = '#f2f2f2'
    neutral_edge = '#bfbfbf'

    # Prepare subtitle color intervals (used for embedded coloring and sign belonging)
    subtitle_color_intervals = []
    if layout == 'embedded':
        # Use aligned subtitle timings for sign-subtitle belonging
        for seg_start, seg_end, text in tier_data.get('SUBTITLE_SHIFTED', []):
            c = assign_subtitle_color(text)
            if c:
                subtitle_color_intervals.append((seg_start, seg_end, c))

    # Layouts
    if layout == 'embedded':
        # Two sign rows (top-down): SIGN, SIGN embedded
        # Three subtitle rows (bottom-up): SUBTITLE (bottom), SUBTITLE embedded, SUBTITLE aligned (top)
        base_tiers = [
            ('SIGN',               'SIGN',                '#f6d88b'),  # color overridden below
            ('SIGN_EMBEDDED',      'SIGN\nembedded',     '#f6d88b'),  # color per segment
            ('SUBTITLE_ALIGNED',   'SUBTITLE\naligned',  '#b8d8ff'),  # neutral below
            ('SUBTITLE_EMBEDDED',  'SUBTITLE\nembedded', '#ccebd0'),  # color per segment
            ('SUBTITLE',           'SUBTITLE',           '#ccebd0'),  # neutral below
        ]
        # Map visual tier to data source tier
        data_key_map = {
            'SIGN': 'SIGN',
            'SIGN_EMBEDDED': 'SIGN',
            'SUBTITLE': 'SUBTITLE',
            'SUBTITLE_EMBEDDED': 'SUBTITLE',
            'SUBTITLE_ALIGNED': 'SUBTITLE_SHIFTED',
        }
    else:
        # Fixed order (top -> bottom) for default layout
        base_tiers = [
            ('SIGN',                r'SIGN$_{\mathrm{pred}}$',                      '#f6d88b'),
            ('CSLR',                r'SIGN$_{\mathrm{gt}}$',     '#ffae7f'),
            ('SUBTITLE',            r'SUBTITLE$_{\mathrm{ori.}}$',           '#ccebd0'),
            ('SUBTITLE_SHIFTED',    r'SUBTITLE$_{\mathrm{pred}}$',                 '#b8d8ff'),
            ('SUBTITLE_CORRECTED',  r'SUBTITLE$_{\mathrm{gt}}$', '#d1b3ff'),
        ]
        data_key_map = {}

    # Apply include flags only in default layout
    tier_names, tier_labels, tier_colors = [], [], []
    for name, label, color in base_tiers:
        if layout == 'default':
            if name == 'CSLR' and not include_sign_gt:
                continue
            if name == 'SUBTITLE_CORRECTED' and not include_subtitle_gt:
                continue
        tier_names.append(name)
        tier_labels.append(label)
        tier_colors.append(color)

    num_tiers = len(tier_names)
    tier_height = 38
    small_spacing = 6
    large_spacing = int(1.5 * tier_height)

    # Big-space mode (applies to default layout only)
    if layout == 'embedded':
        big_space_mode = False
    else:
        big_space_mode = (not include_sign_gt) and (not include_subtitle_gt)

    # Are Sign and Subtitle_d adjacent in this filtered list?
    has_sign_and_subt_d = False
    if 'SIGN' in tier_names and 'SUBTITLE' in tier_names:
        has_sign_and_subt_d = (tier_names.index('SUBTITLE') - tier_names.index('SIGN') == 1)

    # In embedded layout, widen gaps to fit process boxes nicely
    override_spacings = None
    override_top_pad = None
    if layout == 'embedded' and num_tiers >= 2:
        box_h = 28
        margin = 12
        required_gap = box_h + 2 * margin  # more room for arrows + box
        aligned_gap = int(required_gap * 0.6)  # 40% reduction for aligned triplet area
        override_spacings = [small_spacing] * (num_tiers - 1)
        # Gap between SIGN and SIGN_EMBEDDED (keep full gap for process box)
        if 'SIGN' in tier_names and 'SIGN_EMBEDDED' in tier_names:
            i0 = tier_names.index('SIGN')
            i1 = tier_names.index('SIGN_EMBEDDED')
            gap_idx = max(i0, i1) - 1  # index into spacings
            if 0 <= gap_idx < len(override_spacings):
                override_spacings[gap_idx] = max(override_spacings[gap_idx], required_gap)
        # Gap between SIGN_EMBEDDED and SUBTITLE_ALIGNED (reduced)
        if 'SIGN_EMBEDDED' in tier_names and 'SUBTITLE_ALIGNED' in tier_names:
            i0 = tier_names.index('SIGN_EMBEDDED')
            i1 = tier_names.index('SUBTITLE_ALIGNED')
            gap_idx = max(i0, i1) - 1
            if 0 <= gap_idx < len(override_spacings):
                override_spacings[gap_idx] = max(override_spacings[gap_idx], aligned_gap)
        # Gap between SUBTITLE_ALIGNED and SUBTITLE_EMBEDDED (reduced)
        if 'SUBTITLE_ALIGNED' in tier_names and 'SUBTITLE_EMBEDDED' in tier_names:
            i0 = tier_names.index('SUBTITLE_ALIGNED')
            i1 = tier_names.index('SUBTITLE_EMBEDDED')
            gap_idx = max(i0, i1) - 1
            if 0 <= gap_idx < len(override_spacings):
                override_spacings[gap_idx] = max(override_spacings[gap_idx], aligned_gap)
        # Gap between SUBTITLE_EMBEDDED and SUBTITLE (keep full gap for process box)
        if 'SUBTITLE' in tier_names and 'SUBTITLE_EMBEDDED' in tier_names:
            i0 = tier_names.index('SUBTITLE')
            i1 = tier_names.index('SUBTITLE_EMBEDDED')
            gap_idx = max(i0, i1) - 1
            if 0 <= gap_idx < len(override_spacings):
                override_spacings[gap_idx] = max(override_spacings[gap_idx], required_gap)
        # Add a bit more breathing room under frames
        override_top_pad = max(22, (override_top_pad or 0))

    # Compute y positions top-down (Sign at top of tiers)
    y_positions, y_last_top, top_pad, spacings = compute_y_positions_top_down(
        num_tiers, tier_height, frame_height, big_space_mode, has_sign_and_subt_d,
        small_spacing=small_spacing, large_spacing=large_spacing,
        override_top_pad=override_top_pad, override_spacings=override_spacings
    )

    y_last_bottom = y_last_top - tier_height  # bottom of the lowest row

    # Figure height to fit frames + tiers + timestamps comfortably
    tiers_block_height = (top_pad + num_tiers * tier_height + sum(spacings))
    # Timestamps go below the bottom row:
    timestamp_y = y_last_bottom - 14
    fig_height = frame_height + (tiers_block_height + 55 + 20)  # extra margin

    fig, ax = plt.subplots(figsize=(total_width / 80, fig_height / 80))
    x_offsets = [0]
    for w in widths:
        x_offsets.append(x_offsets[-1] + w)

    frame_pixel_width = widths[0] if widths else 0
    first_frame_left = 0
    last_frame_right = total_width
    frame_seconds = num_frames * interval
    plot_end = start_time + num_frames * interval

    # Draw process boxes and arrows (embedded layout only)
    if layout == 'embedded':
        # Compute common x position and box size
        x_mid = total_width * 0.5
        box_w = max(240, min(720, total_width * 0.8))  # wider boxes
        box_h = 28
        arrow_pad = 6
        row_pad = 8

        # 1) Frames -> SIGN via f_segmentation_network
        if 'SIGN' in tier_names:
            j_sign = tier_names.index('SIGN')
            y_sign_top = y_positions[j_sign]
            y_sign_center = (y_sign_top - tier_height) + tier_height / 2.0
            y_box = (0 + y_sign_top) / 2.0
            _draw_process_box(ax, x_mid, y_box, 'Sign segmentation model', box_w, box_h,
                              facecolor='#ffffff', edgecolor='#666', textcolor='#333', z=25)
            # frames bottom up to just below box bottom
            _draw_vertical_arrow(ax, x_mid, 0 + arrow_pad, y_box + box_h/2 - arrow_pad, color='#666', lw=1.8, z=26)
            # box top down to just above SIGN center
            _draw_vertical_arrow(ax, x_mid, y_box - box_h/2 + arrow_pad, y_sign_center - arrow_pad, color='#666', lw=1.8, z=26)

        # 2) SIGN -> SIGN_EMBEDDED via g_sign_embedding (top-down)
        if 'SIGN' in tier_names and 'SIGN_EMBEDDED' in tier_names:
            j_sign = tier_names.index('SIGN')
            j_sign_emb = tier_names.index('SIGN_EMBEDDED')
            y_sign_center = (y_positions[j_sign] - tier_height) + tier_height / 2.0
            y_sign_emb_center = (y_positions[j_sign_emb] - tier_height) + tier_height / 2.0
            y_box = ( (y_positions[j_sign] - tier_height) + y_positions[j_sign_emb] ) / 2.0
            _draw_process_box(ax, x_mid, y_box, 'Sign embedding model', box_w, box_h,
                              facecolor='#ffffff', edgecolor='#666', textcolor='#333', z=25)
            # from near SIGN bottom to just above box bottom
            _draw_vertical_arrow(ax, x_mid, y_sign_center - arrow_pad, y_box + box_h/2 - arrow_pad, color='#666', lw=1.8, z=26)
            # from just below box top to near SIGN_EMBEDDED top
            _draw_vertical_arrow(ax, x_mid, y_box - box_h/2 + arrow_pad, y_sign_emb_center + arrow_pad, color='#666', lw=1.8, z=26)

        # 3) SUBTITLE -> SUBTITLE_EMBEDDED via g_subtitle_embedding (bottom-up)
        if 'SUBTITLE' in tier_names and 'SUBTITLE_EMBEDDED' in tier_names:
            j_sub = tier_names.index('SUBTITLE')
            j_sub_emb = tier_names.index('SUBTITLE_EMBEDDED')
            y_sub_center = (y_positions[j_sub] - tier_height) + tier_height / 2.0
            y_sub_emb_center = (y_positions[j_sub_emb] - tier_height) + tier_height / 2.0
            y_box = ( y_positions[j_sub] + (y_positions[j_sub_emb] - tier_height) ) / 2.0
            _draw_process_box(ax, x_mid, y_box, 'Text embedding model', box_w, box_h,
                              facecolor='#ffffff', edgecolor='#666', textcolor='#333', z=25)
            # from just above SUBTITLE center up to just below box top
            _draw_vertical_arrow(ax, x_mid, y_sub_center + arrow_pad, y_box - box_h/2 - arrow_pad, color='#666', lw=1.8, z=26, shorten=False)
            # from just above box bottom up to just below SUBTITLE_EMBEDDED center

        # New: Curved arrows from SIGN_EMBEDDED and SUBTITLE_EMBEDDED to SUBTITLE_ALIGNED (two targets -> four arrows)
        if 'SUBTITLE_ALIGNED' in tier_names:
            aligned_segments = tier_data.get('SUBTITLE_SHIFTED', [])
            targets = []
            for seg_start, seg_end, text in aligned_segments:
                c = assign_subtitle_color(text)
                if not c or c == '#c77dff':
                    continue
                left_x = (seg_start - start_time) / frame_seconds * total_width
                right_x = (seg_end - start_time) / frame_seconds * total_width
                if seg_start < start_time:
                    left_x = max(first_frame_left - frame_pixel_width/2, left_x)
                else:
                    left_x = max(first_frame_left, left_x)
                if seg_end > plot_end:
                    right_x = min(last_frame_right + frame_pixel_width/2, right_x)
                else:
                    right_x = min(last_frame_right, right_x)
                cx = (left_x + right_x) / 2.0
                targets.append((cx, c))

            j_al = tier_names.index('SUBTITLE_ALIGNED')
            y_al_center = (y_positions[j_al] - tier_height) + tier_height / 2.0
            j_sign_emb = tier_names.index('SIGN_EMBEDDED') if 'SIGN_EMBEDDED' in tier_names else None
            j_sub_emb = tier_names.index('SUBTITLE_EMBEDDED') if 'SUBTITLE_EMBEDDED' in tier_names else None
            y_sign_emb_center = (y_positions[j_sign_emb] - tier_height) + tier_height / 2.0 if j_sign_emb is not None else None
            y_sub_emb_center = (y_positions[j_sub_emb] - tier_height) + tier_height / 2.0 if j_sub_emb is not None else None

            dx = 22
            for cx, c in targets:
                # SIGN_EMBEDDED -> SUBTITLE_ALIGNED: use subtitle color
                if y_sign_emb_center is not None:
                    _draw_curved_arrow(
                        ax,
                        cx - dx, y_sign_emb_center - (tier_height * 0.25),
                        cx,      y_al_center + (tier_height * 0.20),
                        color=c, lw=2.0, rad=-0.25, z=24, shorten=True
                    )
                # SUBTITLE_EMBEDDED -> SUBTITLE_ALIGNED: use same subtitle color
                if y_sub_emb_center is not None:
                    _draw_curved_arrow(
                        ax,
                        cx + dx, y_sub_emb_center + (tier_height * 0.25),
                        cx,      y_al_center - (tier_height * 0.20),
                        color=c, lw=2.0, rad=0.25, z=24, shorten=True
                    )

    # Vertical dashed lines (under everything): from frame bottom to bottom of last row
    y1 = 0
    y2 = y_last_bottom
    for x in x_offsets:
        ax.plot([x, x], [y1, y2], color='k', linewidth=0.7, linestyle='--', alpha=0.5, zorder=1)

    # Draw frames
    for i, img in enumerate(frames):
        extent = [x_offsets[i], x_offsets[i+1], 0, img.height]
        ax.imshow(img, extent=extent, zorder=10)

        # Draw vertical border line on the right edge (except after last frame)
        if i < len(frames) - 1:
            ax.plot([x_offsets[i+1], x_offsets[i+1]], [0, img.height],
                    color='grey', linewidth=2.0, zorder=20)

    # Draw tiers (top -> bottom)
    for j, (tier, label, color) in enumerate(zip(tier_names, tier_labels, tier_colors)):
        y_top = y_positions[j]           # top of the rounded box
        y_lower = y_top - tier_height    # convert to lower-left for patch

        # Tier label on the left
        ax.text(-150, y_lower + tier_height / 2, label,
            ha='left', va='center',
            fontsize=17, fontweight='bold', zorder=11)

        # Resolve data source for this visual tier
        data_key = data_key_map.get(tier, tier)
        segments = tier_data.get(data_key, [])
        for seg_start, seg_end, text in segments:
            left_x = (seg_start - start_time) / frame_seconds * total_width
            right_x = (seg_end - start_time) / frame_seconds * total_width
            if seg_start < start_time:
                left_x = max(first_frame_left - frame_pixel_width/2, left_x)
            else:
                left_x = max(first_frame_left, left_x)
            if seg_end > plot_end:
                right_x = min(last_frame_right + frame_pixel_width/2, right_x)
            else:
                right_x = min(last_frame_right, right_x)

            width = right_x - left_x

            # Choose per-segment colors based on layout and tier
            if layout == 'embedded':
                if tier in ('SIGN', 'SUBTITLE'):
                    seg_face = neutral_face
                    seg_edge = neutral_edge
                elif tier in ('SUBTITLE_EMBEDDED', 'SUBTITLE_ALIGNED'):
                    c = assign_subtitle_color(text)
                    seg_face = c if c else neutral_face
                    seg_edge = seg_face
                elif tier == 'SIGN_EMBEDDED':
                    c_base = pick_subtitle_color_for_interval(subtitle_color_intervals, seg_start, seg_end)
                    if c_base:
                        seg_face = color_variant(c_base, f"{seg_start:.3f}-{seg_end:.3f}")
                        seg_edge = seg_face
                    else:
                        seg_face = neutral_face
                        seg_edge = neutral_edge
                else:
                    seg_face = color
                    seg_edge = color

            rect = patches.FancyBboxPatch(
                (left_x, y_lower), width, tier_height,
                boxstyle="round,pad=0.01", linewidth=1.1,
                edgecolor=seg_edge, facecolor=seg_face, alpha=0.90, zorder=12
            )
            ax.add_patch(rect)

            # segment boundary hints
            boundary_color = '#888'
            boundary_alpha = 0.7
            if abs(seg_start - start_time) > 1e-3:
                ax.plot([left_x, left_x], [y_lower, y_lower + tier_height],
                        color=boundary_color, linewidth=1.1, zorder=13, alpha=boundary_alpha)
            if abs(seg_end - plot_end) > 1e-3:
                ax.plot([right_x, right_x], [y_lower, y_lower + tier_height],
                        color=boundary_color, linewidth=1.1, zorder=13, alpha=boundary_alpha)

            # text
            if text:
                wrapped_text = wrap_text(text, width, font_size=10, px_per_char=7, max_lines=2)
            else:
                wrapped_text = ''
            if wrapped_text:
                rotation = 45 if tier == 'CSLR' else 0
                ax.text(
                    left_x + width / 2, y_lower + tier_height / 2, wrapped_text,
                    ha='center', va='center', fontsize=12, fontweight='normal',
                    wrap=True, clip_on=True, rotation=rotation, zorder=15, color='#000',
                )

    # Timestamps at the vertical borders, placed below the last row
    for i in range(len(x_offsets)):
        time_sec = start_time + i * interval
        minutes = int((time_sec % 3600) // 60)
        seconds = int(time_sec % 60)
        ts = f"{minutes:02d}:{seconds:02d}"
        ax.text(x_offsets[i], timestamp_y, ts, ha='center', va='top',
                fontsize=14, fontfamily='monospace', zorder=20)

    # Ellipsis if content continues outside window (still near frames)
    show_left_ellipsis = any(seg[0] < start_time for segs in tier_data.values() for seg in segs)
    show_right_ellipsis = any(seg[1] > plot_end for segs in tier_data.values() for seg in segs)
    ellipsis_y = frame_height + 8
    if show_left_ellipsis:
        ax.text(-12, ellipsis_y, '…', ha='left', va='bottom', fontsize=18, alpha=0.7)
    if show_right_ellipsis:
        ax.text(total_width + 8, ellipsis_y, '…', ha='left', va='bottom', fontsize=18, alpha=0.7)

    ax.axis('off')
    ax.set_xlim(0, total_width)
    # Ensure we include the timestamps in the view
    lower_ylim = timestamp_y - 20
    ax.set_ylim(lower_ylim, frame_height)
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_pdf}")
    open_pdf(output_pdf)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default='./BOBSL', help='Workspace root')
    parser.add_argument('--video_id', type=str, default='5224144816887051284')
    parser.add_argument('--fps', type=float, default=0.5)
    parser.add_argument('--start', type=float, default=60)
    parser.add_argument('--end', type=float, default=90)
    parser.add_argument('--frame_height', type=int, default=180)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--include_sign_gt', action='store_true', default=False)
    parser.add_argument('--include_subtitle_gt', action='store_true', default=False)
    parser.add_argument('--embedded_layout', action='store_true', help='Use embedded sign/subtitle layout with duplicated rows')
    args = parser.parse_args()

    os.makedirs(args.working_dir, exist_ok=True)
    frame_dir = os.path.join(args.working_dir, 'frames_out')
    input_video = os.path.join(args.working_dir, f"{args.video_id}.mp4")
    input_eaf = os.path.join(args.working_dir, f"{args.video_id}_updated.eaf")
    interval = 1 / args.fps

    # Output filename
    signgt_str = 'signGT' if args.include_sign_gt else 'noSignGT'
    subgt_str = 'subGT' if args.include_subtitle_gt else 'noSubGT'
    layout_str = 'embedded' if args.embedded_layout else 'default'
    output_pdf = os.path.join(
        args.working_dir,
        f"{args.video_id}_{int(args.start)}_{int(args.end)}_{signgt_str}_{subgt_str}_{layout_str}.pdf"
    )

    print(f"Extracting frames from {input_video} at {args.fps} fps, from {args.start}s to {args.end}s")
    extract_exact_frames(input_video, frame_dir, args.start, args.end, interval, args.overwrite)

    num_frames = int((args.end - args.start) / interval)

    tier_names = [
        'SIGN',
        'CSLR',
        'SUBTITLE',
        'SUBTITLE_SHIFTED',
        'SUBTITLE_CORRECTED'
    ]
    print(f"Loading ELAN annotations from {input_eaf}...")
    tier_data = load_elan_segments(input_eaf, args.start, args.end, tiers=tier_names)

    print("Rendering figure...")
    plot_timeline_with_tiers(
        frame_dir, num_frames, args.start, interval, output_pdf, args.frame_height, tier_data,
        include_sign_gt=args.include_sign_gt,
        include_subtitle_gt=args.include_subtitle_gt,
        layout=('embedded' if args.embedded_layout else 'default')
    )

if __name__ == '__main__':
    main()
