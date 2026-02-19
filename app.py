import streamlit as st
from PIL import Image
import numpy as np
import io
import os

st.set_page_config(page_title="Image to BMP Converter", layout="centered")
st.title("🖼️ Image to BMP Converter")
st.markdown("Convert JPG/PNG images to BMP with advanced options: bit depth 16, square resolution, and auto-compress by size.")

# ── Sidebar Settings ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ BMP Settings")

enable_16bit = st.sidebar.checkbox("Set Bit Depth to 16-bit", value=True)

enable_square = st.sidebar.checkbox("Force Square Resolution (W = H)", value=False)
if enable_square:
    square_size = st.sidebar.number_input("Square Size (px)", min_value=16, max_value=4096, value=512, step=16)

enable_compress = st.sidebar.checkbox("Auto-Compress (Limit by File Size)", value=False)
if enable_compress:
    max_size_kb = st.sidebar.number_input("Max File Size (KB)", min_value=10, max_value=10240, value=500, step=10)

# ── Helper Functions ───────────────────────────────────────────────────────────

def convert_to_16bit_bmp(img: Image.Image) -> Image.Image:
    """Convert image to RGB565-like 16-bit by reducing color precision."""
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    # Reduce to 5-6-5 bit precision (RGB565)
    arr[:, :, 0] = (arr[:, :, 0] >> 3) << 3   # R: 5 bits
    arr[:, :, 1] = (arr[:, :, 1] >> 2) << 2   # G: 6 bits
    arr[:, :, 2] = (arr[:, :, 2] >> 3) << 3   # B: 5 bits
    return Image.fromarray(arr, "RGB")

def make_square(img: Image.Image, size: int) -> Image.Image:
    """Resize image to exact square dimensions."""
    return img.resize((size, size), Image.LANCZOS)

def auto_compress(img: Image.Image, max_kb: int) -> Image.Image:
    """Downscale image iteratively until BMP output is within max_kb."""
    max_bytes = max_kb * 1024
    current_img = img.copy()
    
    while True:
        buf = io.BytesIO()
        current_img.save(buf, format="BMP")
        size = buf.tell()
        if size <= max_bytes:
            break
        w, h = current_img.size
        if w <= 16 or h <= 16:
            st.warning("⚠️ Image too small to compress further. Result may exceed target size.")
            break
        scale = (max_bytes / size) ** 0.5
        new_w = max(16, int(w * scale))
        new_h = max(16, int(h * scale))
        current_img = current_img.resize((new_w, new_h), Image.LANCZOS)
    
    return current_img

def process_image(uploaded_file) -> tuple[bytes, dict]:
    """Process a single uploaded image and return BMP bytes + info dict."""
    img = Image.open(uploaded_file).convert("RGB")
    info = {"Original Size": f"{img.width} x {img.height} px"}

    # Step 1: Square resolution
    if enable_square:
        img = make_square(img, square_size)
        info["After Square Resize"] = f"{img.width} x {img.height} px"

    # Step 2: 16-bit depth
    if enable_16bit:
        img = convert_to_16bit_bmp(img)
        info["Bit Depth"] = "16-bit (RGB565)"
    else:
        info["Bit Depth"] = "24-bit (default BMP)"

    # Step 3: Auto-compress
    if enable_compress:
        img = auto_compress(img, max_size_kb)
        info["After Compress Size"] = f"{img.width} x {img.height} px"

    # Save to BMP
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    bmp_bytes = buf.getvalue()
    info["Final File Size"] = f"{len(bmp_bytes) / 1024:.1f} KB"
    info["Final Resolution"] = f"{img.width} x {img.height} px"

    return bmp_bytes, info

# ── File Upload ────────────────────────────────────────────────────────────────
st.markdown("---")
uploaded_files = st.file_uploader(
    "Upload JPG or PNG image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown("---")
    for uploaded_file in uploaded_files:
        st.subheader(f"📄 {uploaded_file.name}")
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Original Image", use_container_width=True)

        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                bmp_bytes, info = process_image(uploaded_file)

                with col2:
                    result_img = Image.open(io.BytesIO(bmp_bytes))
                    st.image(result_img, caption="Preview (BMP)", use_container_width=True)

                # Info table
                st.markdown("**Conversion Info:**")
                for k, v in info.items():
                    st.write(f"- **{k}:** {v}")

                # Download button
                base_name = os.path.splitext(uploaded_file.name)[0]
                st.download_button(
                    label=f"⬇️ Download {base_name}.bmp",
                    data=bmp_bytes,
                    file_name=f"{base_name}.bmp",
                    mime="image/bmp",
                    key=f"dl_{uploaded_file.name}"
                )
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        st.markdown("---")

else:
    st.info("👆 Upload one or more JPG/PNG images to get started.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.caption("Built with Streamlit & Pillow")
