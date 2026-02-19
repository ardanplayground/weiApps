import streamlit as st
from PIL import Image
import numpy as np
import io
import os

st.set_page_config(page_title="BMP Converter", layout="wide", page_icon="🖼️")

st.markdown("""
<style>
.info-box {
    background: #1e1e2e;
    border-left: 4px solid #89b4fa;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.88em;
}
.success-box {
    background: #1e1e2e;
    border-left: 4px solid #a6e3a1;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.88em;
}
.warn-box {
    background: #1e1e2e;
    border-left: 4px solid #f9e2af;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.88em;
}
</style>
""", unsafe_allow_html=True)

# ── Default Presets ────────────────────────────────────────────────────────────
DEFAULT_PRESETS = {
    "ST7735 128x128": {"width": 128, "height": 128, "bit_depth": 16, "max_kb": 64},
    "ST7735 160x128": {"width": 160, "height": 128, "bit_depth": 16, "max_kb": 80},
    "ILI9341 320x240": {"width": 320, "height": 240, "bit_depth": 16, "max_kb": 300},
    "ILI9341 240x240": {"width": 240, "height": 240, "bit_depth": 16, "max_kb": 200},
    "SSD1306 128x64": {"width": 128, "height": 64, "bit_depth": 16, "max_kb": 32},
    "Custom": {"width": 320, "height": 320, "bit_depth": 16, "max_kb": 500},
}

if "presets" not in st.session_state:
    st.session_state.presets = DEFAULT_PRESETS.copy()
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "ILI9341 320x240"

# ── Core Functions ─────────────────────────────────────────────────────────────

def apply_rgb565(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    arr[:, :, 0] = (arr[:, :, 0] >> 3) << 3
    arr[:, :, 1] = (arr[:, :, 1] >> 2) << 2
    arr[:, :, 2] = (arr[:, :, 2] >> 3) << 3
    return Image.fromarray(arr, "RGB")

def to_bmp_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="BMP")
    return buf.getvalue()

def auto_compress(img: Image.Image, max_kb: int) -> tuple:
    max_bytes = max_kb * 1024
    current = img.copy()
    compressed = False
    for _ in range(40):
        bmp = to_bmp_bytes(current)
        if len(bmp) <= max_bytes:
            break
        w, h = current.size
        if w <= 8 or h <= 8:
            break
        scale = (max_bytes / len(bmp)) ** 0.5 * 0.95
        current = current.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.LANCZOS)
        compressed = True
    return current, compressed

def process_image(img: Image.Image, cfg: dict) -> dict:
    steps = []
    out = img.convert("RGB")
    steps.append(f"Original: {out.width}×{out.height} px")

    # Step 1: Resize to target resolution
    tw, th = cfg["width"], cfg["height"]
    out = out.resize((tw, th), Image.LANCZOS)
    steps.append(f"Resize → {tw}×{th} px")

    # Step 2: Bit depth
    if cfg["bit_depth"] == 16:
        out = apply_rgb565(out)
        steps.append("16-bit RGB565 applied")
    else:
        steps.append("24-bit (no depth reduction)")

    # Step 3: Auto-compress
    compressed = False
    if cfg["max_kb"] > 0:
        out, compressed = auto_compress(out, cfg["max_kb"])
        if compressed:
            steps.append(f"Auto-compressed → {out.width}×{out.height} px")

    bmp = to_bmp_bytes(out)
    final_kb = len(bmp) / 1024
    steps.append(f"Final: {out.width}×{out.height} px  |  {final_kb:.1f} KB")

    return {
        "image": out,
        "bmp_bytes": bmp,
        "steps": steps,
        "size_kb": final_kb,
        "compressed": compressed,
        "within_limit": final_kb <= cfg["max_kb"] if cfg["max_kb"] > 0 else True,
    }

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🖼️ JPG / PNG → BMP Converter")
st.caption("Otomatis resize, 16-bit RGB565, dan auto-compress sesuai preset device atau konfigurasi custom.")

# ── Device Config ──────────────────────────────────────────────────────────────
with st.expander("⚙️ Device & Conversion Settings", expanded=True):
    left, right = st.columns([1, 2])

    with left:
        st.markdown("##### Pilih Preset Device")
        preset_names = list(st.session_state.presets.keys())
        sel = st.radio(
            "preset", preset_names,
            index=preset_names.index(st.session_state.selected_preset)
                  if st.session_state.selected_preset in preset_names else 0,
            label_visibility="collapsed"
        )
        st.session_state.selected_preset = sel

        st.markdown("---")
        st.markdown("##### ➕ Tambah Preset Baru")
        new_name = st.text_input("Nama preset", placeholder="e.g. My LCD 480x320")
        ca, cb = st.columns(2)
        with ca:
            np_w = st.number_input("Width", 8, 4096, 480, key="np_w")
            np_bd = st.selectbox("Bit depth", [16, 24], key="np_bd")
        with cb:
            np_h = st.number_input("Height", 8, 4096, 320, key="np_h")
            np_kb = st.number_input("Max KB", 0, 102400, 500, key="np_kb")
        if st.button("💾 Simpan Preset", use_container_width=True):
            if new_name.strip():
                st.session_state.presets[new_name.strip()] = {
                    "width": int(np_w), "height": int(np_h),
                    "bit_depth": int(np_bd), "max_kb": int(np_kb),
                }
                st.session_state.selected_preset = new_name.strip()
                st.success(f"Preset '{new_name.strip()}' disimpan!")
                st.rerun()
            else:
                st.warning("Nama preset tidak boleh kosong.")

    with right:
        st.markdown("##### Edit Setting Preset Terpilih")
        cfg = st.session_state.presets[sel].copy()
        c1, c2 = st.columns(2)
        with c1:
            cfg["width"]     = st.number_input("Target Width (px)", 8, 4096, cfg["width"], key="cw")
            cfg["bit_depth"] = st.selectbox("Bit Depth", [16, 24],
                                index=0 if cfg["bit_depth"] == 16 else 1, key="cbd")
        with c2:
            cfg["height"]  = st.number_input("Target Height (px)", 8, 4096, cfg["height"], key="ch")
            cfg["max_kb"]  = st.number_input("Max File Size (KB, 0 = no limit)", 0, 102400, cfg["max_kb"], key="ckb")

        force_sq = st.checkbox("Force Square Resolution (W = H)", value=cfg["width"] == cfg["height"])
        if force_sq:
            cfg["height"] = cfg["width"]

        st.session_state.presets[sel] = cfg

        limit_str = f"{cfg['max_kb']} KB" if cfg["max_kb"] > 0 else "No limit"
        st.markdown(f"""
        <div class="info-box">
        📐 <b>Resolution:</b> {cfg['width']} × {cfg['height']} px &nbsp;|&nbsp;
        🎨 <b>Bit Depth:</b> {cfg['bit_depth']}-bit &nbsp;|&nbsp;
        📦 <b>Max Size:</b> {limit_str}
        </div>
        """, unsafe_allow_html=True)

        if sel not in DEFAULT_PRESETS:
            if st.button("🗑️ Hapus Preset Ini", type="secondary"):
                del st.session_state.presets[sel]
                st.session_state.selected_preset = list(st.session_state.presets.keys())[0]
                st.rerun()

# ── Upload & Convert ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📤 Upload Gambar")
files = st.file_uploader(
    "Upload JPG / PNG — bisa multiple file",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if files:
    active_cfg = st.session_state.presets[st.session_state.selected_preset]
    st.markdown(f"**{len(files)} file** akan dikonversi dengan preset **{sel}**")

    for uf in files:
        st.markdown("---")
        img_orig = Image.open(uf)

        col_a, col_b, col_c = st.columns([1, 1, 1])

        with col_a:
            st.markdown(f"**🔴 Original** — `{uf.name}`")
            st.image(img_orig, use_container_width=True)
            st.caption(f"{img_orig.width}×{img_orig.height} px  |  {uf.size/1024:.1f} KB")

        with st.spinner(f"Converting {uf.name}..."):
            res = process_image(img_orig, active_cfg)

        with col_b:
            st.markdown("**🟢 Hasil BMP**")
            st.image(res["image"], use_container_width=True)
            icon = "✅" if res["within_limit"] else "⚠️"
            st.caption(f"{res['image'].width}×{res['image'].height} px  |  {icon} {res['size_kb']:.1f} KB")

        with col_c:
            st.markdown("**📋 Conversion Log**")
            for step in res["steps"]:
                st.markdown(f"<div class='info-box'>• {step}</div>", unsafe_allow_html=True)

            if res["compressed"]:
                st.markdown("<div class='warn-box'>⚠️ Gambar di-downscale agar muat limit KB</div>", unsafe_allow_html=True)

            if res["within_limit"]:
                st.markdown("<div class='success-box'>✅ Konversi berhasil!</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warn-box'>⚠️ Ukuran melebihi limit (gambar terlalu kecil)</div>", unsafe_allow_html=True)

            base = os.path.splitext(uf.name)[0]
            st.download_button(
                label=f"⬇️ Download {base}.bmp",
                data=res["bmp_bytes"],
                file_name=f"{base}.bmp",
                mime="image/bmp",
                use_container_width=True,
                key=f"dl_{uf.name}_{uf.size}"
            )
else:
    st.markdown("""
    <div style='text-align:center;padding:50px;color:#6c7086;'>
    <h3>👆 Upload gambar JPG atau PNG untuk mulai konversi</h3>
    <p>Atur preset device di atas, lalu upload — semua proses berjalan otomatis.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit & Pillow — RGB565 16-bit BMP Converter")
