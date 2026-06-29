"""
Smart Adaptive Traffic Light System
====================================
Dua mode:
  📸 Image Mode  : Upload 1 foto intersection → langsung hitung durasi
                   lampu hijau adaptif (one-shot, no simulation).
  📹 Video Mode  : Simulasi real-time dengan state machine.

Run: streamlit run app.py
"""

import tempfile
import time

import cv2
import numpy as np
import streamlit as st

import config
from detector import VehicleDetector
from traffic_controller import TrafficLightController


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Smart Adaptive Traffic Light",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Smart Adaptive Traffic Light System")
# st.caption(
#     "Classical Computer Vision — OpenCV + NumPy + Streamlit. "
#     "No deep learning, no CNN, no YOLO."
# )


# ============================================================
# SESSION STATE
# ============================================================
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    mode = st.radio(
        "Mode",
        ["📸 Image Mode", "📹 Video Mode"],
        index=0,
        help="Image Mode: 1 foto → durasi langsung. Video Mode: simulasi real-time.",
    )
    st.divider()

    # ----- IMAGE MODE SIDEBAR -----
    if mode == "📸 Image Mode":
        uploaded_image = st.file_uploader(
            "Upload intersection image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Foto intersection dari atas (top-down) paling ideal.",
        )

        st.subheader("Image Detection Parameters")
        st.caption("HSV color-based detection (lebih cocok drone view).")
        saturation_thresh = st.slider(
            "Saturation threshold", 10, 120, 120,
            help="Lebih kecil = lebih banyak warna terdeteksi sebagai kendaraan.",
        )
        bright_thresh = st.slider(
            "Brightness threshold (terang)", 120, 230, 175,
            help="Threshold mobil putih/terang. Lebih besar = strict.",
        )
        dark_thresh = st.slider(
            "Brightness threshold (gelap)", 10, 100, 55,
            help="Threshold mobil hitam/gelap. Lebih kecil = strict.",
        )
        img_min_area = st.slider(
            "Min vehicle area (px)", 100, 10000, 150, step=50,
            help="Lebih kecil = catch kendaraan kecil di drone view.",
        )
        img_max_area = st.slider(
            "Max vehicle area (px)", 5000, 200000, 80000, step=1000)
        max_aspect = st.slider(
            "Max aspect ratio", 2.0, 15.0, 6.0, step=0.5,
            help="Filter lane marking yang panjang/tipis. Lebih kecil = strict.",
        )
        img_morph_size = st.slider("Morph kernel size", 3, 11, 5, step=2)

    # ----- VIDEO MODE SIDEBAR -----
    else:
        uploaded_video = st.file_uploader(
            "Upload traffic video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Video CCTV/dash-cam intersection.",
        )
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            st.session_state.video_path = tfile.name
            st.success("✅ Video loaded")

        st.subheader("Video Detection Parameters")
        bg_threshold = st.slider("Background threshold (MOG2)", 4, 60, config.BG_THRESHOLD)
        min_area = st.slider("Min vehicle area (px)", 100, 5000, config.MIN_CONTOUR_AREA, step=100)
        max_area = st.slider("Max vehicle area (px)", 5000, 100000, config.MAX_CONTOUR_AREA, step=1000)
        morph_size = st.slider("Morph kernel size", 3, 11, config.MORPH_KERNEL_SIZE, step=2)

    st.divider()

    # ----- ROI Configuration (preset + optional custom) -----
    st.subheader("📐 ROI Configuration")
    roi_preset = st.selectbox(
        "ROI Preset",
        list(config.ROI_PRESETS.keys()) + ["Custom (set manually)"],
        index=0,
        help=(
            "MTID Drone View: tuned untuk dataset MTID dari Kaggle.\n"
            "Generic 4-way: untuk foto intersection 4-jalur simetris (mis. Google Maps).\n"
            "Custom: set sendiri rectangle bounds tiap lane."
        ),
    )

    if roi_preset == "Custom (set manually)":
        # Inisialisasi dari MTID preset supaya user punya starting point yang masuk akal
        default = config.ROIS_MTID_DRONE
        active_rois = {}
        for lane in ["North", "South", "East", "West"]:
            poly = default[lane]
            # Hitung bounding rect dari polygon awal
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            init_x1, init_y1 = min(xs), min(ys)
            init_x2, init_y2 = max(xs), max(ys)

            with st.expander(f"Lane: {lane}", expanded=False):
                cols = st.columns(2)
                x1 = cols[0].number_input(
                    "x1", 0, 1280, init_x1, step=10, key=f"{lane}_x1",
                )
                y1 = cols[1].number_input(
                    "y1", 0, 720, init_y1, step=10, key=f"{lane}_y1",
                )
                x2 = cols[0].number_input(
                    "x2", 0, 1280, init_x2, step=10, key=f"{lane}_x2",
                )
                y2 = cols[1].number_input(
                    "y2", 0, 720, init_y2, step=10, key=f"{lane}_y2",
                )
            # Convert rectangle ke polygon (clockwise dari top-left)
            active_rois[lane] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        active_rois = config.ROI_PRESETS[roi_preset]

    st.divider()

    # ----- SHARED: Traffic Light Timing -----
    st.subheader("Traffic Light Timing")
    base_green = st.slider("Base green (s)", 5, 60, config.BASE_GREEN_TIME)
    min_green = st.slider("Min green (s)", 3, 30, config.MIN_GREEN_TIME)
    max_green = st.slider("Max green (s)", 20, 120, config.MAX_GREEN_TIME)

    # ----- VIDEO MODE EXTRA: yellow + playback -----
    if mode == "📹 Video Mode":
        yellow_time = st.slider("Yellow (s)", 1, 10, config.YELLOW_TIME)
        all_red_time = st.slider("All-red (s)", 1, 10, config.ALL_RED_TIME)
        st.divider()
        st.subheader("Playback")
        speed_factor = st.slider("Speed multiplier", 0.5, 5.0, 1.0, step=0.5)
        loop_video = st.checkbox("Loop video saat habis", value=True)
        st.divider()
        col_btn1, col_btn2 = st.columns(2)
        preview_btn = col_btn1.button("👁️ Preview ROI", use_container_width=True)
        start_btn = col_btn2.button("▶️ Start", type="primary", use_container_width=True)
        stop_btn = st.button("⏹️ Stop", use_container_width=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    tw, th = config.TARGET_FRAME_WIDTH, config.TARGET_FRAME_HEIGHT
    if (w, h) != (tw, th):
        frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    return frame


def compute_one_shot_green_times(
    counts: dict, base_green: float, min_green: float, max_green: float,
) -> dict:
    """One-shot adaptive green time: green_i = base × N × (count_i / total)."""
    n_lanes = len(counts)
    total = sum(counts.values())
    if total <= 0:
        return {lane: float(base_green) for lane in counts}
    green_times = {}
    for lane, count in counts.items():
        proportion = count / total
        green = base_green * n_lanes * proportion
        green = max(min_green, min(max_green, green))
        green_times[lane] = float(green)
    return green_times


def draw_annotations(frame, vehicles_per_lane, states, rois):
    """Overlay ROI + bbox + count labels."""
    annotated = frame.copy()
    overlay = frame.copy()

    for lane_name, polygon in rois.items():
        polygon_np = np.array(polygon, dtype=np.int32)
        state = states.get(lane_name, "RED")
        if state == "GREEN":
            color = (0, 220, 0)
        elif state == "YELLOW":
            color = (0, 220, 220)
        else:
            color = (0, 0, 220)
        cv2.fillPoly(overlay, [polygon_np], color)
        cv2.polylines(annotated, [polygon_np], True, color, 2)

    annotated = cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0)

    for lane_name, vehicles in vehicles_per_lane.items():
        for v in vehicles:
            x, y, w, h = v["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cx, cy = v["centroid"]
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)

    # Posisi label DI LUAR polygon supaya gak nutupin mobil
    label_positions = {
        "West":  (160, 220),
        "North": (700, 135),
        "East":  (685, 370),
        "South": (430, 440),
    }
    
    for lane_name, polygon in rois.items():
        count = len(vehicles_per_lane.get(lane_name, []))
        polygon_np = np.array(polygon)
        
        pos = label_positions.get(lane_name)
        if pos is None:
            cx = int(np.mean(polygon_np[:, 0]))
            cy = int(np.mean(polygon_np[:, 1]))
        else:
            cx, cy = pos
        
        label = f"{lane_name}: {count}"
        (tw_, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (cx - 5, cy - th_ - 5),
                      (cx + tw_ + 5, cy + 5), (0, 0, 0), -1)
        cv2.putText(annotated, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


def render_traffic_lights_html(states, current_lane, phase,
                                time_remaining, phase_duration):
    """HTML panel untuk Video Mode."""
    color_map = {
        "GREEN": "#22c55e", "YELLOW": "#facc15",
        "RED": "#ef4444", "ALL_RED": "#ef4444",
    }
    rows = []
    for lane, state in states.items():
        is_active = (lane == current_lane)
        display_label = "RED" if state == "ALL_RED" else state
        countdown_html = ""
        if is_active:
            pct = (1 - time_remaining / phase_duration) * 100 if phase_duration > 0 else 0
            countdown_html = f"""
              <div style="font-size:13px; color:#cbd5e1; margin-top:4px;">
                ⏱ {time_remaining:.1f}s / {phase_duration:.1f}s
              </div>
              <div style="background:#334155; height:6px; border-radius:3px; margin-top:6px;">
                <div style="width:{pct:.0f}%; height:100%; background:{color_map[state]};
                            border-radius:3px;"></div>
              </div>
            """
        border_style = "2px solid #fff" if is_active else "1px solid #334155"
        rows.append(f"""
          <div style="margin-bottom:12px; padding:14px; border-radius:10px;
                      background:#1e293b; border:{border_style};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div style="font-weight:600; color:#f1f5f9; font-size:15px;">{lane}</div>
              <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:18px; height:18px; border-radius:50%;
                            background:{color_map[state]};
                            box-shadow:0 0 10px {color_map[state]};"></div>
                <span style="color:#cbd5e1; font-size:13px;">{display_label}</span>
              </div>
            </div>
            {countdown_html}
          </div>
        """)
    return "".join(rows)


def render_image_lights_html(green_times, counts):
    """HTML panel untuk Image Mode (visualisasi alokasi green time)."""
    max_green = max(green_times.values()) if green_times else 0
    rows = []
    for lane in green_times:
        green = green_times[lane]
        count = counts[lane]
        is_longest = green == max_green and max_green > 0
        bar_width = (green / max_green) * 100 if max_green > 0 else 0
        border = "2px solid #22c55e" if is_longest else "1px solid #334155"
        badge = ("<span style='background:#22c55e; color:#0f172a; padding:2px 8px;"
                 " border-radius:6px; font-size:11px; font-weight:700; margin-left:6px;'>LONGEST</span>"
                 if is_longest else "")
        rows.append(f"""
          <div style="margin-bottom:14px; padding:14px; border-radius:10px;
                      background:#1e293b; border:{border};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
              <div style="font-weight:600; color:#f1f5f9; font-size:15px;">{lane}{badge}</div>
              <div style="color:#cbd5e1; font-size:13px;">{count} kendaraan</div>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
              <div style="flex:1; background:#0f172a; height:14px; border-radius:7px; overflow:hidden;">
                <div style="width:{bar_width:.0f}%; height:100%; background:#22c55e;"></div>
              </div>
              <div style="color:#22c55e; font-weight:700; font-size:18px; min-width:60px; text-align:right;">
                {green:.1f}s
              </div>
            </div>
          </div>
        """)
    return "".join(rows)


# ============================================================
# MAIN AREA
# ============================================================
rois = active_rois
lane_names = list(rois.keys())


# ------------------------------------------------------------
# IMAGE MODE
# ------------------------------------------------------------
if mode == "📸 Image Mode":

    if uploaded_image is None:
        st.info("👈 Upload foto intersection di sidebar untuk mulai.")
        with st.expander("📖 Cara kerja Image Mode", expanded=True):
            st.markdown("""
            **Image Mode** memproses 1 foto secara one-shot (tidak ada video, tidak ada simulasi waktu):

            1. **Convert BGR → HSV color space** — Pisahkan warna (Hue), kepekatan (Saturation),
               dan kecerahan (Value).
            2. **Identifikasi pixel "bukan-jalan"**:
               - Saturation tinggi → kendaraan berwarna (bus kuning, truk merah, mobil biru)
               - Brightness sangat tinggi → mobil putih, marka jalan
               - Brightness sangat rendah → mobil hitam/gelap
            3. **Morphological Opening + Closing** — hapus noise titik, tutup lubang dalam blob.
            4. **Contour Detection** — Cari kontur tiap blob, filter by:
               - **Area** (buang noise terlalu kecil & artifact terlalu besar)
               - **Aspect ratio** (buang lane marking yang panjang & tipis)
            5. **Point-in-Polygon** — Cek centroid tiap kendaraan ada di ROI mana
               (North/South/East/West).
            6. **Adaptive Allocation**:
               ```
               green_i = base × N × (count_i / total_count)
               ```
               Dibatasi `[min_green, max_green]`.

            **Kenapa HSV color, bukan edge detection?**
            Drone view dari atas: roof mobil itu uniform color, edge antar mobil lemah,
            dilation bisa menggabungkan beberapa mobil jadi 1 blob besar. HSV color-based
            lebih robust — tiap mobil tetap punya warna distinct meski berdekatan.

            **Tips foto:** top-down view ideal, keempat jalur kelihatan jelas, resolusi ≥ 960×540.
            """)
    else:
        # Decode image
        bytes_data = uploaded_image.read()
        img_array = np.frombuffer(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            st.error("❌ Gagal baca gambar. Coba format lain.")
        else:
            image = resize_frame(image)

            # Process
            result = VehicleDetector.process_single_image(
                image, rois,
                saturation_thresh=saturation_thresh,
                bright_thresh=bright_thresh,
                dark_thresh=dark_thresh,
                kernel_size=img_morph_size,
                min_area=img_min_area,
                max_area=img_max_area,
                max_aspect_ratio=max_aspect,
            )
            counts = result["counts"]
            green_times = compute_one_shot_green_times(
                counts, base_green, min_green, max_green,
            )

            # Highlight lane dengan green time terlama sebagai GREEN
            max_lane = max(green_times, key=green_times.get) if green_times else None
            fake_states = {lane: ("GREEN" if lane == max_lane else "RED")
                           for lane in lane_names}
            annotated = draw_annotations(
                image, result["vehicles_per_lane"], fake_states, rois,
            )

            # ----- DISPLAY -----
            st.subheader("🎯 Hasil Analisis")
            total_count = sum(counts.values())
            if total_count > 0:
                st.caption(
                    f"Total kendaraan terdeteksi: **{total_count}**. "
                    f"Lane kepadatan tertinggi: **{max_lane}** → green time terlama."
                )
            else:
                st.warning(
                    "⚠ Belum ada kendaraan terdeteksi. Coba turunkan **Min vehicle area** "
                    "atau **Canny low threshold** di sidebar, atau cek apakah ROI sudah "
                    "menutupi area jalur yang benar."
                )

            col_img, col_lights = st.columns([3, 1])
            with col_img:
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Deteksi — ROI hijau = green light terpanjang, "
                            "bbox putih = kendaraan terdeteksi.",
                    use_container_width=True,
                )
                with st.expander("🔍 Lihat detection mask (binary)"):
                    st.image(
                        result["mask"],
                        caption="Binary mask hasil edge detection + morphology.",
                        use_container_width=True,
                    )

            with col_lights:
                st.markdown("**Alokasi Durasi Lampu Hijau**")
                st.markdown(
                    render_image_lights_html(green_times, counts),
                    unsafe_allow_html=True,
                )

            # ----- TABEL -----
            st.subheader("📊 Detail Per Jalur")
            table_rows = []
            for lane in lane_names:
                count = counts[lane]
                proportion = (count / total_count * 100) if total_count > 0 else 0
                green = green_times[lane]
                fixed_green = base_green
                if green > fixed_green * 1.1:
                    category = "🔴 Padat"
                elif green < fixed_green * 0.9:
                    category = "🟢 Sepi"
                else:
                    category = "🟡 Normal"
                table_rows.append({
                    "Lane": lane,
                    "Kendaraan": count,
                    "Proporsi": f"{proportion:.1f}%",
                    "Green Adaptif": f"{green:.1f}s",
                    "Green Fixed": f"{fixed_green:.0f}s",
                    "Selisih": f"{green - fixed_green:+.1f}s",
                    "Kategori": category,
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

            # ----- BAR CHART -----
            st.subheader("📈 Adaptif vs Fixed Timing")
            chart_data = {
                "Lane": lane_names,
                "Adaptive (smart)": [green_times[l] for l in lane_names],
                "Fixed (konvensional)": [float(base_green)] * len(lane_names),
            }
            st.bar_chart(chart_data, x="Lane",
                         color=["#22c55e", "#64748b"])

            # ----- METRICS -----
            total_adaptive = sum(green_times.values())
            total_fixed = base_green * len(lane_names)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Cycle Adaptif", f"{total_adaptive:.1f}s")
            col_m2.metric("Total Cycle Fixed", f"{total_fixed:.0f}s")
            col_m3.metric("Total Kendaraan", f"{total_count}")

            if total_count > 0 and max_lane:
                pct_change = (green_times[max_lane] / base_green - 1) * 100
                st.success(
                    f"✅ Lane **{max_lane}** dengan **{counts[max_lane]} kendaraan** "
                    f"(proporsi {counts[max_lane]/total_count*100:.1f}%) "
                    f"dapat green time **{green_times[max_lane]:.1f}s** "
                    f"— **{pct_change:+.0f}%** vs fixed timing."
                )


# ------------------------------------------------------------
# VIDEO MODE
# ------------------------------------------------------------
elif mode == "📹 Video Mode":

    if stop_btn:
        st.session_state.running = False
    if start_btn:
        if st.session_state.video_path is None:
            st.sidebar.error("⚠️ Upload video dulu.")
        else:
            st.session_state.running = True
            st.session_state.history = []

    # Preview ROI
    if preview_btn:
        if st.session_state.video_path is None:
            st.error("⚠️ Upload video dulu untuk preview ROI.")
        else:
            cap = cv2.VideoCapture(st.session_state.video_path)
            ret, first_frame = cap.read()
            cap.release()
            if ret:
                first_frame = resize_frame(first_frame)
                preview = first_frame.copy()
                overlay = first_frame.copy()
                for lane_name, polygon in rois.items():
                    poly = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(overlay, [poly], (0, 200, 0))
                    cv2.polylines(preview, [poly], True, (0, 255, 0), 2)
                    cx = int(np.mean(poly[:, 0]))
                    cy = int(np.mean(poly[:, 1]))
                    cv2.putText(preview, lane_name, (cx - 20, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                preview = cv2.addWeighted(overlay, 0.2, preview, 0.8, 0)
                st.subheader("ROI Preview (first frame)")
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.info("Edit `DEFAULT_ROIS` di `config.py` kalau ROI belum pas.")
            else:
                st.error("Gagal baca video.")

    # Simulation loop
    if st.session_state.running and st.session_state.video_path:
        detector = VehicleDetector(
            history=config.BG_HISTORY,
            threshold=bg_threshold,
            detect_shadows=config.BG_DETECT_SHADOWS,
            kernel_size=morph_size,
            open_iter=config.MORPH_OPEN_ITER,
            close_iter=config.MORPH_CLOSE_ITER,
        )
        controller = TrafficLightController(
            lane_names=lane_names,
            base_green=base_green,
            min_green=min_green,
            max_green=max_green,
            yellow_time=yellow_time,
            all_red_time=all_red_time,
        )

        cap = cv2.VideoCapture(st.session_state.video_path)
        if not cap.isOpened():
            st.error("Gagal buka video.")
            st.session_state.running = False
        else:
            col_video, col_lights = st.columns([3, 1])
            video_placeholder = col_video.empty()
            lights_placeholder = col_lights.empty()
            stats_placeholder = st.empty()
            history_placeholder = st.empty()

            frame_count = 0
            delay = config.PLAYBACK_DELAY / max(0.1, speed_factor)

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    if loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        st.session_state.running = False
                        break

                frame = resize_frame(frame)
                result = detector.process(frame, rois,
                                          min_area=min_area, max_area=max_area)
                counts = result["counts"]

                if frame_count % config.COUNT_UPDATE_INTERVAL == 0:
                    controller.update_counts(counts)

                state = controller.step()
                annotated = draw_annotations(
                    frame, result["vehicles_per_lane"], state["states"], rois,
                )
                video_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

                lights_placeholder.markdown(
                    render_traffic_lights_html(
                        state["states"], state["current_lane"], state["phase"],
                        state["time_remaining"], state["phase_duration"],
                    ),
                    unsafe_allow_html=True,
                )

                smoothed = state["smoothed_counts"]
                stats_md = ["**Live counts** (instant → smoothed):"]
                for lane in lane_names:
                    stats_md.append(f"- **{lane}**: {counts[lane]} → {smoothed[lane]:.1f}")
                stats_md.append(
                    f"\n**Cycle**: {state['cycle_count']}  |  "
                    f"**Active**: {state['current_lane']} ({state['phase']})"
                )
                stats_placeholder.markdown("\n".join(stats_md))

                if controller.history:
                    recent = controller.history[-10:]
                    rows = "\n".join(
                        f"| {h['cycle']} | {h['lane']} | "
                        f"{h['count_at_start']:.1f} | {h['green_duration']:.1f}s |"
                        for h in recent
                    )
                    history_placeholder.markdown(
                        "**Recent green-light allocations (last 10):**\n\n"
                        "| Cycle | Lane | Smoothed Count | Green Duration |\n"
                        "|---|---|---|---|\n" + rows
                    )

                frame_count += 1
                time.sleep(delay)
            cap.release()

    # Idle
    if not st.session_state.running and not preview_btn:
        st.info("👈 Upload video, klik **Preview ROI**, lalu **Start**.")
        with st.expander("📖 Cara kerja Video Mode", expanded=True):
            st.markdown("""
            1. **MOG2 Background Subtraction** — Pisahkan kendaraan bergerak.
            2. **Morphological** — Opening + Closing untuk cleanup mask.
            3. **Contour Detection** — Filter blob by area.
            4. **ROI Counting** — Point-in-polygon test per ROI.
            5. **Adaptive Timing** — Green proporsional dengan rolling-average smoothing.
            6. **State Machine** — GREEN → YELLOW → ALL_RED → next lane.
            """)