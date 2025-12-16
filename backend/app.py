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
llm_client  = Client("jeyanthangj2004/eng-draw-llm-flan-t5")

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


# ---------------- VLM prompt templates (same logic as Colab) ----------------

def full_page_prompt():
    return (
        "You are an assistant that qualitatively describes full engineering drawing sheets. "
        "You are given an image of a COMPLETE engineering drawing page. "
        "Write EXACTLY THREE concise sentences. Do NOT number the sentences. Each sentence must be on a new line. "
        "Sentence 1 should state what general type of component or assembly this page appears to describe. "
        "Sentence 2 should describe the types of views present on the page and what aspects of the part they show. "
        "Sentence 3 should mention any clearly visible title block information and whether a table or BOM is present."
    )


def drawing_crop_prompt(full_page_text):
    return (
        "You are given a combined cropped image where the ORIGINAL region is on the left and the ENHANCED region is on the right. "
        f"The full engineering drawing was previously described as follows: {full_page_text} "
        "This cropped region shows only part of that same drawing. Without repeating the full-page description, "
        "describe this cropped region in EXACTLY THREE concise sentences, each on its own line. "
        "Sentence 1: what part or sub-component this region represents. "
        "Sentence 2: what view type(s) are visible and what geometry they highlight. "
        "Sentence 3: key qualitative geometric features without numeric values."
    )


def title_block_crop_prompt(full_page_text):
    return (
        "You are given a combined cropped image where the ORIGINAL region is on the left and the ENHANCED region is on the right. "
        f"The full engineering drawing was previously described as follows: {full_page_text} "
        "This cropped region shows only the title block area. Without repeating the full-page description, "
        "describe this title block region in EXACTLY THREE concise sentences, each on its own line. "
        "Sentence 1: drawing number, title, and revision if readable, otherwise say unreadable. "
        "Sentence 2: scale and sheet number if visible, otherwise say unreadable. "
        "Sentence 3: material, date, and drafter if readable, otherwise say unreadable."
    )


def table_crop_prompt(full_page_text):
    return (
        "You are given a combined cropped image where the ORIGINAL region is on the left and the ENHANCED region is on the right. "
        f"The full engineering drawing was previously described as follows: {full_page_text} "
        "This cropped region shows a table from that drawing. Without repeating the full-page description, "
        "describe this table region in EXACTLY THREE concise sentences, each on its own line. "
        "Sentence 1: what kind of table it is and which column headings are visible. "
        "Sentence 2: describe one to three example rows in words. "
        "Sentence 3: summarize what this table represents for the overall drawing."
    )


def detail_crop_prompt(full_page_text):
    return (
        "You are given a combined cropped image where the ORIGINAL region is on the left and the ENHANCED region is on the right. "
        f"The full engineering drawing was previously described as follows: {full_page_text} "
        "This cropped region shows a local detail view. Without repeating the full-page description, "
        "describe this detail region in EXACTLY THREE concise sentences, each on its own line. "
        "Sentence 1: what local feature this detail focuses on. "
        "Sentence 2: any labels or callouts and what they refer to. "
        "Sentence 3: visible notes or tolerances, described qualitatively without numeric values."
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


# ---------------- OpenRouter VLM call (fallback logic) ----------------

def call_vlm_openrouter(
    pil_image: Image.Image,
    region_type: str,
    full_page_text: str | None,
    max_new_tokens: int,
) -> str:
    """
    Call OpenRouter VLM (Nemotron VL) directly with the same behavior as in Colab:
    - For 'page', use full_page_prompt and given max_new_tokens.
    - For crops, use crop prompts, force max_new_tokens >= 768.
    - Fallback with a simpler prompt if finish_reason == 'length' and content is empty.
    """
    if openrouter_client is None:
        return "[OpenRouter API key not configured on backend]"

    # Choose prompt and adjust tokens
    if region_type == "page":
        prompt = full_page_prompt()
        tokens_to_use = int(max_new_tokens)
    else:
        if not full_page_text or not full_page_text.strip():
            return "[full_page_text is required for crop regions when using OpenRouter]"
        prompt = get_crop_prompt(region_type, full_page_text)
        tokens_to_use = max(int(max_new_tokens), 768)

    # Encode image
    image_b64 = pil_image_to_base64_str(pil_image)

    def extract_text(content):
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            pieces = []
            for part in content:
                if hasattr(part, "type"):
                    if part.type in ("text", "output_text"):
                        pieces.append(getattr(part, "text", "") or "")
                elif isinstance(part, dict):
                    if part.get("type") in ("text", "output_text"):
                        pieces.append(part.get("text", "") or "")
            return " ".join(p.strip() for p in pieces if p).strip()
        else:
            return str(content) if content is not None else ""

    def call_model(current_prompt: str):
        return openrouter_client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": current_prompt},
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

    # First attempt
    try:
        response = call_model(prompt)
    except Exception as e:
        print(f"OpenRouter VLM error: {e}")
        return f"[OpenRouter VLM error: {e}]"

    choice = response.choices[0]
    msg = choice.message
    content = getattr(msg, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)

    output_text = extract_text(content)

    # Fallback on 'length' with empty content
    if finish_reason == "length" and not output_text:
        if region_type == "page":
            simple_prompt = (
                "Describe this engineering drawing in exactly three short sentences, each on its own line, "
                "covering overall component type, view types, and visible title block or tables."
            )
        else:
            simple_prompt = (
                "Describe this cropped region of an engineering drawing in exactly three concise sentences, "
                "each on its own line, stating what it shows, what views are visible, and its key geometric features."
            )

        try:
            response2 = call_model(simple_prompt)
            choice2 = response2.choices[0]
            content2 = getattr(choice2.message, "content", None)
            output_text2 = extract_text(content2)
            if output_text2:
                output_text = output_text2
            else:
                output_text = "[Model hit the token limit twice without producing content.]"
        except Exception as e2:
            print(f"OpenRouter VLM fallback error: {e2}")
            output_text = f"[OpenRouter VLM fallback error: {e2}]"

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
        detections     = yolo_result[3]

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
            conf     = det.get("conf") or det.get("score") or 0.0
            box      = det.get("box") or []

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
                        256,                          # max_new_tokens (not critical; Colab forces for crops)
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
                        # For OpenRouter we only need the raw crop; enhanced is only for visualization in Gradio UI
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
