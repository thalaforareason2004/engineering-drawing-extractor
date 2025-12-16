import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from PIL import Image
import io
import base64
import os
import shutil
import tempfile

from openai import OpenAI  # For OpenRouter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HF Space clients (YOLO + LLM) ----------------

yolo_client = Client("jeyanthangj2004/engg-draw-extractor")
llm_client = Client("jeyanthangj2004/eng-draw-llm-flan-t5")

# ---------------- OpenRouter VLM client (fallback when no Gradio URL) ----------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
else:
    openrouter_client = None

# Vision model on OpenRouter (must support images)
VLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"


# ---------------- Utility functions ----------------

def encode_image_to_base64(image_path_or_obj):
    """
    Encode an image (path or PIL.Image) as data:image/png;base64,...
    Used for returning images to frontend.
    """
    try:
        if isinstance(image_path_or_obj, str):
            with Image.open(image_path_or_obj) as img:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_str = base64.b64encode(buf.getvalue()).decode()
        else:
            buf = io.BytesIO()
            image_path_or_obj.save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Encoding error: {e}")
        return None


def extract_image_path_from_gallery_item(item: dict):
    if not isinstance(item, dict):
        return None
    val = item.get("image")
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("path")
    return None


def pil_image_to_base64_str(pil_image: Image.Image) -> str:
    """Encode PIL image to base64 string (no data: prefix)."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------- VLM prompt templates (simplified) ----------------

def full_page_prompt():
    return (
        "You see a complete engineering drawing page. "
        "Write EXACTLY THREE short sentences, each on its own line, and do NOT number them. "
        "Sentence 1: say what general type of component or assembly this page appears to show. "
        "Sentence 2: say what views are present (for example front, side, section, isometric) and what aspects of the part they highlight. "
        "Sentence 3: mention any visible title-block information and whether a table or BOM is present."
    )


def drawing_crop_prompt(full_page_text: str):
    return (
        f"The full engineering drawing page was previously described as: {full_page_text} "
        "Now you see a CROPPED REGION of that same drawing. "
        "Write EXACTLY THREE short sentences, each on its own line, and do NOT number them. "
        "Sentence 1: say what part or sub-component this cropped region represents. "
        "Sentence 2: say what kind of view(s) are shown here and what geometry they focus on. "
        "Sentence 3: describe key qualitative geometric features visible in this region without using numeric values."
    )


def title_block_crop_prompt(full_page_text: str):
    return (
        f"The full engineering drawing page was previously described as: {full_page_text} "
        "Now you see a CROPPED TITLE BLOCK REGION from that same drawing. "
        "Write EXACTLY THREE short sentences, each on its own line, and do NOT number them. "
        "Sentence 1: give drawing number, title and revision if readable, otherwise say they are unreadable. "
        "Sentence 2: give scale and sheet number if visible, otherwise say they are unreadable. "
        "Sentence 3: mention material, date and drafter if readable, otherwise say they are unreadable."
    )


def table_crop_prompt(full_page_text: str):
    return (
        f"The full engineering drawing page was previously described as: {full_page_text} "
        "Now you see a CROPPED TABLE REGION from that same drawing. "
        "Write EXACTLY THREE short sentences, each on its own line, and do NOT number them. "
        "Sentence 1: say what kind of table this is and list the visible column headings. "
        "Sentence 2: describe one to three example rows in words. "
        "Sentence 3: summarize what this table represents for the overall drawing."
    )


def detail_crop_prompt(full_page_text: str):
    return (
        f"The full engineering drawing page was previously described as: {full_page_text} "
        "Now you see a CROPPED DETAIL VIEW from that same drawing. "
        "Write EXACTLY THREE short sentences, each on its own line, and do NOT number them. "
        "Sentence 1: say what local feature or area this detail focuses on. "
        "Sentence 2: mention any labels or callouts and what they point to. "
        "Sentence 3: summarize any visible notes or tolerances in qualitative terms without numeric values."
    )


def get_crop_prompt(region_type: str, full_page_text: str):
    r = (region_type or "").lower()
    if r == "drawing":
        return drawing_crop_prompt(full_page_text)
    if r == "title_block":
        return title_block_crop_prompt(full_page_text)
    if r == "table":
        return table_crop_prompt(full_page_text)
    if r in ("detail", "details"):
        return detail_crop_prompt(full_page_text)
    # default to drawing
    return drawing_crop_prompt(full_page_text)


# ---------------- Simple OpenRouter VLM call ----------------

def call_vlm_openrouter(
    pil_image: Image.Image,
    region_type: str,
    full_page_text: str | None,
    max_new_tokens: int,
) -> str:
    """
    Simple one-shot call to OpenRouter VLM:
    - Builds the appropriate prompt (page vs crop)
    - Uses max_new_tokens as given
    - Returns the assistant text content (or a basic fallback string)
    """
    if openrouter_client is None:
        return "[OpenRouter API key not configured on backend]"

    # Choose prompt
    if region_type == "page":
        prompt = full_page_prompt()
    else:
        if not full_page_text or not full_page_text.strip():
            return "[full_page_text is required for crop regions when using OpenRouter]"
        prompt = get_crop_prompt(region_type, full_page_text)

    # Use the provided max_new_tokens, defaulting if needed
    try:
        tokens_to_use = int(max_new_tokens) if max_new_tokens else 256
    except Exception:
        tokens_to_use = 256

    # Encode image
    image_b64 = pil_image_to_base64_str(pil_image)

    try:
        response = openrouter_client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    },
                ],
            }],
            max_tokens=tokens_to_use,
        )
    except Exception as e:
        print(f"OpenRouter VLM error: {e}")
        return f"[OpenRouter VLM error: {e}]"

    choice = response.choices[0]
    msg = choice.message
    content = getattr(msg, "content", None)

    # Simple content extraction
    if isinstance(content, str):
        output_text = content.strip()
    elif isinstance(content, list):
        pieces = []
        for part in content:
            if hasattr(part, "type") and part.type in ("text", "output_text"):
                pieces.append(getattr(part, "text", "") or "")
            elif isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                pieces.append(part.get("text", "") or "")
        output_text = " ".join(p.strip() for p in pieces if p).strip()
    else:
        output_text = str(content) if content is not None else ""

    if not output_text:
        output_text = "[No content returned by OpenRouter VLM]"
    return output_text


# ---------------- API endpoints ----------------

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/analyze-page")
async def analyze_page(
    file: UploadFile = File(...),
    vlm_url: str = Form(None),  # Optional: Colab Gradio URL for VLM
):
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. YOLO via HF Space
        yolo_result = yolo_client.predict(
            image=handle_file(temp_file_path),
            conf=0.3,
            api_name="/analyze_yolo",
        )

        annotated_info = yolo_result[0]
        raw_crops_info = yolo_result[1]
        enh_crops_info = yolo_result[2]
        detections = yolo_result[3]

        annotated_image_b64 = encode_image_to_base64(annotated_info)

        # 2. VLM – FULL PAGE (RAW ONLY)
        vlm_client = None
        full_page_vlm_text = None

        # Case A: Use external Gradio VLM if URL is provided (Colab live URL)
        if vlm_url:
            try:
                vlm_client = Client(vlm_url)
                full_page_vlm_text = vlm_client.predict(
                    handle_file(temp_file_path),  # raw_image
                    "page",                       # region_type
                    None,                         # full_page_text (NOT USED)
                    None,                         # enhanced_image (NOT USED)
                    512,                          # max_new_tokens
                    api_name="/predict",
                )
            except Exception as e:
                print(f"Full-page VLM (Gradio) error: {e}")
                vlm_client = None
                full_page_vlm_text = None

        # Case B: If no Gradio VLM URL, use OpenRouter directly
        elif openrouter_client is not None:
            try:
                with Image.open(temp_file_path) as img:
                    pil_full = img.convert("RGB")
                    full_page_vlm_text = call_vlm_openrouter(
                        pil_image=pil_full,
                        region_type="page",
                        full_page_text=None,
                        max_new_tokens=512,
                    )
            except Exception as e:
                print(f"Full-page VLM (OpenRouter) error: {e}")
                full_page_vlm_text = None

        page_context_for_crops = full_page_vlm_text

        # 3. Crops (RAW + ENHANCED + CONTEXT)
        crops_data = []
        num_crops = min(len(raw_crops_info), len(enh_crops_info))

        for i in range(num_crops):
            raw_item = raw_crops_info[i]
            enh_item = enh_crops_info[i]

            raw_crop_path = extract_image_path_from_gallery_item(raw_item)
            enh_crop_path = extract_image_path_from_gallery_item(enh_item)

            det = detections[i] if i < len(detections) else {}

            cls_name = det.get("cls_name") or det.get("label") or det.get("class") or "unknown"
            conf = det.get("conf") or det.get("score") or 0.0
            box = det.get("box") or []

            vlm_text = None

            # Crop description via VLM
            # Case A: Use Gradio VLM (if vlm_client is set and paths are valid)
            if vlm_client and raw_crop_path and enh_crop_path:
                try:
                    vlm_text = vlm_client.predict(
                        handle_file(raw_crop_path),   # raw_image
                        cls_name,                     # region_type
                        page_context_for_crops,       # full_page_text (REQUIRED)
                        handle_file(enh_crop_path),   # enhanced_image
                        256,                          # max_new_tokens (backed by Colab logic)
                        api_name="/predict",
                    )
                except Exception as e:
                    print(f"Crop VLM (Gradio) error {i}: {e}")
                    vlm_text = None

            # Case B: No Gradio VLM; use OpenRouter VLM directly
            elif openrouter_client is not None and raw_crop_path and page_context_for_crops:
                try:
                    with Image.open(raw_crop_path) as img_raw:
                        pil_raw = img_raw.convert("RGB")
                        vlm_text = call_vlm_openrouter(
                            pil_image=pil_raw,
                            region_type=cls_name,
                            full_page_text=page_context_for_crops,
                            max_new_tokens=256,
                        )
                except Exception as e:
                    print(f"Crop VLM (OpenRouter) error {i}: {e}")
                    vlm_text = None

            crops_data.append({
                "cls_name": cls_name,
                "conf": float(conf),
                "box": box,
                "raw_crop": encode_image_to_base64(raw_crop_path),
                "enhanced_crop": encode_image_to_base64(enh_crop_path),
                "vlm_text": vlm_text,
            })

        # 4. LLM input assembly
        analysis_parts = []

        if full_page_vlm_text:
            analysis_parts.append(
                "Full-page VLM description:\n" + str(full_page_vlm_text)
            )

        for i, crop in enumerate(crops_data):
            if crop["vlm_text"]:
                analysis_parts.append(
                    f"\nCrop {i} (class={crop['cls_name']}, conf={crop['conf']:.2f}):\n"
                    f"{crop['vlm_text']}"
                )

        analysis_text = "\n".join(analysis_parts) if analysis_parts else ""

        # 5. LLM summary via HF Space
        summary = llm_client.predict(
            analysis_text,
            256,
            api_name="/gradio_summarize",
        )

        return {
            "annotated_image": annotated_image_b64,
            "llm_summary": summary,
            "crops": crops_data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
