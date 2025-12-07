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

def open_pdf(path):
    if sys.platform.startswith('darwin'):
        subprocess.run(['open', path])
    elif sys.platform.startswith('linux'):
        subprocess.run(['xdg-open', path])
    elif sys.platform.startswith('win'):
        os.startfile(path)
    else:
        print(f"Please open {path} manually.")

def compute_y_positions_top_down(
    num_tiers, tier_height, frame_height,
    big_space_mode, has_sign_and_subt_d,
    small_spacing=6, large_spacing=None
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

    if big_space_mode:
        # In big-space mode, make ALL gaps equal and large:
        #   top (frames→Sign) = large
        #   Sign→Subtitle_d   = large
        #   Subtitle_d→Subtitle = large
        top_pad = large_spacing
        spacings = [large_spacing] * (num_tiers - 1)
    else:
        # Compact mode (original behavior): small everywhere
        top_pad = small_spacing
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


def plot_timeline_with_tiers(frame_dir, num_frames, start_time, interval, output_pdf, frame_height, tier_data,
                             include_sign_gt=False, include_subtitle_gt=True):
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

    # Fixed order (top -> bottom)
    base_tiers = [
        ('SIGN',                r'SIGN$_{\mathrm{pred}}$',                      '#f6d88b'),   # 0
        ('CSLR',                r'SIGN$_{\mathrm{gt}}$',     '#ffae7f'),   # 1 (optional)
        ('SUBTITLE',            r'SUBTITLE$_{\mathrm{ori.}}$',           '#ccebd0'),   # 2
        ('SUBTITLE_SHIFTED',    r'SUBTITLE$_{\mathrm{pred}}$',                 '#b8d8ff'),   # 3
        ('SUBTITLE_CORRECTED',  r'SUBTITLE$_{\mathrm{gt}}$', '#d1b3ff'),   # 4 (optional)
    ]

    # Apply include flags
    tier_names, tier_labels, tier_colors = [], [], []
    for name, label, color in base_tiers:
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

    # Big-space mode: both GT tiers excluded
    big_space_mode = (not include_sign_gt) and (not include_subtitle_gt)

    # Are Sign and Subtitle_d adjacent in this filtered list?
    has_sign_and_subt_d = False
    if 'SIGN' in tier_names and 'SUBTITLE' in tier_names:
        has_sign_and_subt_d = (tier_names.index('SUBTITLE') - tier_names.index('SIGN') == 1)

    # Compute y positions top-down (Sign at top of tiers)
    y_positions, y_last_top, top_pad, spacings = compute_y_positions_top_down(
        num_tiers, tier_height, frame_height, big_space_mode, has_sign_and_subt_d,
        small_spacing=small_spacing, large_spacing=large_spacing
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

        segments = tier_data.get(tier, [])
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
            rect = patches.FancyBboxPatch(
                (left_x, y_lower), width, tier_height,
                boxstyle="round,pad=0.01", linewidth=1.1,
                edgecolor=color, facecolor=color, alpha=0.90, zorder=12
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
    args = parser.parse_args()

    os.makedirs(args.working_dir, exist_ok=True)
    frame_dir = os.path.join(args.working_dir, 'frames_out')
    input_video = os.path.join(args.working_dir, f"{args.video_id}.mp4")
    input_eaf = os.path.join(args.working_dir, f"{args.video_id}_updated.eaf")
    interval = 1 / args.fps

    signgt_str = 'signGT' if args.include_sign_gt else 'noSignGT'
    subgt_str = 'subGT' if args.include_subtitle_gt else 'noSubGT'
    output_pdf = os.path.join(
        args.working_dir,
        f"{args.video_id}_{int(args.start)}_{int(args.end)}_{signgt_str}_{subgt_str}.pdf"
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
        include_subtitle_gt=args.include_subtitle_gt
    )

if __name__ == '__main__':
    main()
