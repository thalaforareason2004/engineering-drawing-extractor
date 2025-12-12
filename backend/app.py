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

app = FastAPI()

# CORS for local testing (relax later if you want to restrict origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HF Space clients (YOLO + LLM) using full URLs and ssl_verify=False
# NOTE: ssl_verify=False is for local dev where corporate proxies/self-signed certs
#       cause verification errors. Do not use this on untrusted networks.
yolo_client = Client(
    "https://jeyanthangj2004-engg-draw-extractor.hf.space",
    ssl_verify=False,
)
llm_client = Client(
    "https://jeyanthangj2004-eng-draw-llm-flan-t5.hf.space",
    ssl_verify=False,
)


def encode_image_to_base64(image_path_or_obj):
    """Convert a file path or PIL.Image to a data:image/png;base64,... string."""
    try:
        if isinstance(image_path_or_obj, str):
            with Image.open(image_path_or_obj) as img:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            # Assume it's a PIL Image
            buffered = io.BytesIO()
            image_path_or_obj.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def extract_image_path_from_gallery_item(item: dict):
    """
    YOLO Space returns gallery items like:
      {"image": "/tmp/gradio/.../image.webp", "caption": None}
    or:
      {"image": {"path": "...", "url": ...}, "caption": None}
    This helper normalizes to a simple file path string.
    """
    if not isinstance(item, dict):
        return None

    val = item.get("image")
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("path")
    return None


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/analyze-page")
async def analyze_page(
    file: UploadFile = File(...),
    vlm_url: str = Form(None),
):
    """
    Main orchestrator endpoint:
    - Receives image + optional VLM URL.
    - Calls YOLO Space for detections + crops.
    - If VLM URL is provided, calls VLM:
        - full page → full_page_vlm_text
        - each enhanced crop → crop["vlm_text"]
    - Builds analysis text (VLM-first, YOLO fallback) and calls LLM Space for a summary.
    """
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Received VLM URL: {vlm_url or 'None'}")

        # 1. Call YOLO backend Space
        print("Calling YOLO Space...")
        yolo_result = yolo_client.predict(
            image=handle_file(temp_file_path),
            conf=0.3,
            api_name="/analyze_yolo",
        )

        # result: (annotated_image_path, raw_crops_gallery, enh_crops_gallery, detections_list)
        annotated_info = yolo_result[0]
        raw_crops_info = yolo_result[1]
        enh_crops_info = yolo_result[2]
        detections     = yolo_result[3]

        # Encode annotated image as data URL
        annotated_image_b64 = encode_image_to_base64(annotated_info)

        # Prepare optional VLM client
        vlm_client = None
        full_page_vlm_text = None

        if vlm_url:
            try:
                print("Initializing VLM client...")
                vlm_client = Client(vlm_url, ssl_verify=False)

                print("Calling VLM on full page...")
                full_page_vlm_text = vlm_client.predict(
                    image=handle_file(temp_file_path),
                    max_new_tokens=512,
                    api_name="/predict",
                )
                print("Full-page VLM text obtained.")
            except Exception as e:
                print(f"Error calling VLM on full page: {e}")
                vlm_client = None
                full_page_vlm_text = None

        # 2. Assemble crops data, optionally calling VLM per crop
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
            if vlm_client:
                try:
                    # Prefer enhanced crop for VLM if available
                    path_for_vlm = enh_crop_path or raw_crop_path
                    if path_for_vlm:
                        print(f"Calling VLM on crop {i}...")
                        vlm_text = vlm_client.predict(
                            image=handle_file(path_for_vlm),
                            max_new_tokens=256,
                            api_name="/predict",
                        )
                except Exception as e:
                    print(f"Error calling VLM on crop {i}: {e}")
                    vlm_text = None

            crop_item = {
                "cls_name": cls_name,
                "conf": float(conf),
                "box": box,
                "raw_crop": encode_image_to_base64(raw_crop_path),
                "enhanced_crop": encode_image_to_base64(enh_crop_path),
                "vlm_text": vlm_text,
            }
            crops_data.append(crop_item)

        # 3. Build analysis text for LLM (VLM-first, YOLO fallback)
        analysis_parts = []

        if full_page_vlm_text:
            analysis_parts.append("Full-page VLM description:\n" + full_page_vlm_text)

        for i, crop in enumerate(crops_data):
            if crop["vlm_text"]:
                analysis_parts.append(
                    f"\nCrop {i} (class={crop['cls_name']}, conf={crop['conf']:.2f}, box={crop['box']}):\n"
                    f"{crop['vlm_text']}"
                )

        if not analysis_parts:
            # Fallback to YOLO-only info if no VLM text
            analysis_lines = []
            for det in detections:
                label = det.get("cls_name") or det.get("label") or det.get("class") or "unknown"
                conf  = det.get("conf") or det.get("score") or 0.0
                box   = det.get("box") or []
                analysis_lines.append(f"class={label}, conf={conf:.2f}, box={box}")
            analysis_text = "\n".join(analysis_lines)
        else:
            analysis_text = "\n".join(analysis_parts)

        print("Final analysis text for LLM:\n", analysis_text)

        # 4. Call LLM backend Space with correct api_name
        print("Calling LLM Space...")
        summary = llm_client.predict(
            analysis_text,
            256,                      # max_new_tokens
            api_name="/gradio_summarize",
        )

        response_data = {
            "annotated_image": annotated_image_b64,
            "llm_summary": summary,
            "crops": crops_data,
        }

        return response_data

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # For local development
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)