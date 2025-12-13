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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HF Space clients
yolo_client = Client("jeyanthangj2004/engg-draw-extractor")
llm_client  = Client("jeyanthangj2004/eng-draw-llm-flan-t5")


def encode_image_to_base64(image_path_or_obj):
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


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/analyze-page")
async def analyze_page(
    file: UploadFile = File(...),
    vlm_url: str = Form(None),
):
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. YOLO
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

        # 2. VLM full page FIRST
        vlm_client = None
        full_page_vlm_text = None

        if vlm_url:
            try:
                vlm_client = Client(vlm_url)

                full_page_vlm_text = vlm_client.predict(
                    raw_image=handle_file(temp_file_path),
                    enhanced_image=handle_file(temp_file_path),
                    region_type="page",
                    max_new_tokens=512,
                    api_name="/predict",
                )
            except Exception as e:
                print(f"Full-page VLM error: {e}")
                vlm_client = None
                full_page_vlm_text = None

        # This is the CONTEXT passed to every crop
        page_context_for_crops = full_page_vlm_text or ""

        # 3. Crop loop (VLM uses full-page context)
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
            if vlm_client and raw_crop_path:
                try:
                    enhanced_for_vlm = enh_crop_path or raw_crop_path

                    vlm_text = vlm_client.predict(
                        raw_image=handle_file(raw_crop_path),
                        enhanced_image=handle_file(enhanced_for_vlm),
                        region_type=cls_name,
                        full_page_context=page_context_for_crops,  # ✅ CONTEXT ADDED
                        max_new_tokens=256,
                        api_name="/predict",
                    )
                except Exception as e:
                    print(f"Crop VLM error {i}: {e}")
                    vlm_text = None

            crops_data.append({
                "cls_name": cls_name,
                "conf": float(conf),
                "box": box,
                "raw_crop": encode_image_to_base64(raw_crop_path),
                "enhanced_crop": encode_image_to_base64(enh_crop_path),
                "vlm_text": vlm_text,
            })

        # 4. Build analysis for LLM
        analysis_parts = []

        if full_page_vlm_text:
            analysis_parts.append(
                "Full-page VLM description:\n" + full_page_vlm_text
            )

        for i, crop in enumerate(crops_data):
            if crop["vlm_text"]:
                analysis_parts.append(
                    f"\nCrop {i} (class={crop['cls_name']}, conf={crop['conf']:.2f}):\n"
                    f"{crop['vlm_text']}"
                )

        if not analysis_parts:
            analysis_text = "\n".join(
                f"class={d.get('cls_name','unknown')}, conf={d.get('conf',0):.2f}, box={d.get('box',[])}"
                for d in detections
            )
        else:
            analysis_text = "\n".join(analysis_parts)

        # 5. LLM summary
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
