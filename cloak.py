#!/usr/bin/env python3
"""
invisible_cloak_full.py

Features:
- CLI: --camera, --color (red|green|blue|auto)
- Auto color sampling with a guide box (part of cloak inside box is enough)
- SPACE to sample (auto) or capture background (after stepping out)
- 'r' to recapture background, 'c' to recalibrate color (auto mode)
- Status messages on-screen
"""

import cv2
import numpy as np
import argparse

KERNEL = np.ones((3, 3), np.uint8)


def get_color_range(hsv_frame, rect):
    """Extract HSV range from region of interest inside rect (returns lower, upper)."""
    x, y, w, h = rect
    roi = hsv_frame[y:y+h, x:x+w]

    h_vals = roi[:, :, 0].flatten()
    s_vals = roi[:, :, 1].flatten()
    v_vals = roi[:, :, 2].flatten()

    # Use percentiles to avoid outliers, then add a margin
    h_min, h_max = np.percentile(h_vals, [5, 95])
    s_min, s_max = np.percentile(s_vals, [5, 95])
    v_min, v_max = np.percentile(v_vals, [5, 95])

    lower = np.array([max(0, h_min - 10), max(50, s_min - 40), max(50, v_min - 40)], dtype=int)
    upper = np.array([min(180, h_max + 10), min(255, s_max + 40), min(255, v_max + 40)], dtype=int)

    return lower, upper


def make_range_list_from_color_name(color_name):
    """Return a list of (lower, upper) HSV pairs for a named color."""
    if color_name == "red":
        # red needs two ranges (wrap around hue)
        return [
            (np.array([0, 120, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
            (np.array([170, 120, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
        ]
    elif color_name == "blue":
        return [
            (np.array([94, 80, 2], dtype=np.uint8), np.array([126, 255, 255], dtype=np.uint8)),
        ]
    elif color_name == "green":
        return [
            (np.array([36, 40, 40], dtype=np.uint8), np.array([70, 255, 255], dtype=np.uint8)),
            (np.array([70, 40, 40], dtype=np.uint8), np.array([86, 255, 255], dtype=np.uint8)),
        ]

    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Invisible Cloak with OpenCV (full)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--color", type=str, default="red",
                        choices=["red", "green", "blue", "auto"],
                        help="Cloak color: red|green|blue|auto (auto = sample from box)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}")
        return

    # Try to get frame size (fallback by reading a frame if needed)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w == 0 or frame_h == 0:
        ret_tmp, tmp = cap.read()
        if not ret_tmp:
            print("Error: cannot read from camera.")
            cap.release()
            return
        frame_h, frame_w = tmp.shape[:2]

    # ROI for auto-detection — centered box, scaled to frame
    box_w = min(220, frame_w // 3)
    box_h = min(220, frame_h // 3)
    rect_x = frame_w // 2 - box_w // 2
    rect_y = frame_h // 2 - box_h // 2
    rect = (rect_x, rect_y, box_w, box_h)

    auto_mode = (args.color.lower() == "auto")

    # If manual color (not auto), initialize cloak ranges right away
    cloak_ranges = None
    sampling_done = False
    if not auto_mode:
        cloak_ranges = make_range_list_from_color_name(args.color.lower())
        sampling_done = True  # already have color set for manual mode

    background = None

    print("Instructions:")
    print(" - SPACE : sample (auto mode) or capture background (step out of view).")
    print(" - r     : recapture background")
    print(" - c     : recalibrate color (auto mode only)")
    print(" - q/ESC : quit")
    if auto_mode:
        print("Auto mode enabled: place part of the cloak inside the box when sampling.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Active effect if we have both background and cloak ranges
        if background is not None and cloak_ranges is not None:
            # Combine multiple inRange masks from a list of (lower, upper) pairs
            mask = None
            for lower, upper in cloak_ranges:
                m = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
                mask = m if mask is None else cv2.bitwise_or(mask, m)


            # refine mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
            mask = cv2.dilate(mask, KERNEL, iterations=1)

            mask_inv = cv2.bitwise_not(mask)

            background_part = cv2.bitwise_and(background, background, mask=mask)
            foreground_part = cv2.bitwise_and(frame, frame, mask=mask_inv)
            final = cv2.addWeighted(background_part, 1, foreground_part, 1, 0)

            # Status text (include 'c' only if auto_mode)
            status_text = "INVISIBLE CLOAK ACTIVE! Press 'r' to recapture bg"
            if auto_mode:
                status_text += " | 'c' to recalibrate color"
            status_text += " | 'q' to quit"

            cv2.putText(final, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imshow("Invisible Cloak Effect", final)

        else:
            # When we don't have both background and cloak ranges, show helpful instructions
            if auto_mode and not sampling_done:
                # Draw sampling rectangle
                x, y, w, h = rect
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display,
                            "Place the cloak inside box & press SPACE to sample color",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(display,
                            "(only a patch of the cloak is needed)",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 50), 1)
            elif auto_mode and sampling_done and background is None:
                cv2.putText(display,
                            "Step Out of view and press SPACE to capture background",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            else:
                # background not captured yet (manual or after reset)
                cv2.putText(display,
                            "Step out completely and press SPACE to capture background",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Invisible Cloak Effect", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            break

        elif key == ord('r'):  # recapture background
            background = None
            print("[INFO] Background reset. Step out and press SPACE to capture it again.")

        elif key == ord('c') and auto_mode:
            # recalibrate color
            cloak_ranges = None
            sampling_done = False
            print("[INFO] Recalibration requested. Place part of the cloak in the box and press SPACE to sample.")

        elif key == 32:  # SPACE pressed
            # Priority in auto mode: if sampling not done -> sample color; else if background missing -> capture background
            if auto_mode and not sampling_done:
                x, y, w, h = rect
                hsv_frame = hsv  # already computed
                lower, upper = get_color_range(hsv_frame, rect)
                cloak_ranges = [(np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))]
                sampling_done = True
                print(f"[INFO] Auto-detected color range: lower={lower}, upper={upper}")
            else:
                # Capture background (user should step out)
                background = frame.copy()
                print("[INFO] Background captured. Invisible cloak effect is now active (if color set).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
