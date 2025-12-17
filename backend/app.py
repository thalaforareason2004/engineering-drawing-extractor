import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gradio_client import Client, handle_file
from PIL import Image
import io
import base64
import os
import shutil
import tempfile
import csv
import json
import asyncio

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

def parse_page_extract(text: str):
    fields = {}
    if not text:
        return fields
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return fields

def json_event(type_str, data_dict):
    """Helper to format JSON event string for streaming (NDJSON)"""
    return json.dumps({"type": type_str, "data": data_dict}) + "\n"

def status_event(message):
    return json.dumps({"type": "status", "message": message}) + "\n"

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/analyze-page")
async def analyze_page(
    file: UploadFile = File(...),
    vlm_url: str = Form(None),
):
    async def process_stream():
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        try:
            # Save uploaded file
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            yield status_event("Running YOLO detection...")

            # 1. YOLO
            # Run in thread to avoid blocking the event loop
            yolo_result = await asyncio.to_thread(
                yolo_client.predict,
                image=handle_file(temp_file_path),
                conf=0.3,
                api_name="/analyze_yolo"
            )

            annotated_info = yolo_result[0]
            raw_crops_info = yolo_result[1]
            enh_crops_info = yolo_result[2]
            detections     = yolo_result[3]

            annotated_image_b64 = encode_image_to_base64(annotated_info)
            
            # Send YOLO Result immediately (First visual update)
            yield json_event("yolo_result", {
                "annotated_image": annotated_image_b64,
                "detection_count": len(detections)
            })

            vlm_client = None
            full_page_vlm_text = None
            page_extract_text = None
            page_extract_dict = {}
            csv_path = None

            page_context_for_crops = ""

            if vlm_url:
                try:
                    vlm_client = Client(vlm_url)
                    yield status_event("Analyzing full page context...")

                    # 2a. FULL PAGE DESCRIPTION
                    full_page_vlm_text = await asyncio.to_thread(
                        vlm_client.predict,
                        handle_file(temp_file_path),
                        "page",
                        None,
                        None,
                        512,
                        api_name="/predict"
                    )
                    page_context_for_crops = full_page_vlm_text

                    # 2b. FULL PAGE STRUCTURED EXTRACTION
                    yield status_event("Extracting structured data...")
                    page_extract_text = await asyncio.to_thread(
                        vlm_client.predict,
                        handle_file(temp_file_path),
                        "page_extract",
                        None,
                        None,
                        512,
                        api_name="/predict"
                    )

                    page_extract_dict = parse_page_extract(page_extract_text)
                    csv_path = os.path.join(temp_dir, "page_extract.csv")

                    # Send Page Extraction Result (Second visual update)
                    yield json_event("page_extract", {
                        "full_page_vlm_text": full_page_vlm_text,
                        "page_extract_text": page_extract_text,
                        "page_extract_table": page_extract_dict,
                        "page_extract_csv_path": csv_path
                    })

                except Exception as e:
                    yield status_event(f"VLM Warning: {str(e)}")
                    print(f"VLM Init Error: {e}")

            # 3. CROPS PROCESSING
            crops_data = []
            num_crops = min(len(raw_crops_info), len(enh_crops_info))

            yield status_event(f"Processing {num_crops} detected regions...")

            for i in range(num_crops):
                # Notify UI about current crop processing
                yield status_event(f"Analyzing region {i+1}/{num_crops}...")

                raw_item = raw_crops_info[i]
                enh_item = enh_crops_info[i]

                raw_crop_path = extract_image_path_from_gallery_item(raw_item)
                enh_crop_path = extract_image_path_from_gallery_item(enh_item)

                det = detections[i] if i < len(detections) else {}
                cls_name = det.get("cls_name") or det.get("label") or det.get("class") or "unknown"
                conf     = det.get("conf") or det.get("score") or 0.0
                box      = det.get("box") or []

                vlm_text = None
                if vlm_client and raw_crop_path and enh_crop_path:
                    try:
                        vlm_text = await asyncio.to_thread(
                            vlm_client.predict,
                            handle_file(raw_crop_path),
                            cls_name,
                            page_context_for_crops,
                            handle_file(enh_crop_path),
                            256,
                            api_name="/predict"
                        )
                    except Exception as e:
                        print(f"Crop VLM error: {e}")
                        vlm_text = "Analysis failed for this region."

                crop_data = {
                    "cls_name": cls_name,
                    "conf": float(conf),
                    "box": box,
                    "raw_crop": encode_image_to_base64(raw_crop_path),
                    "enhanced_crop": encode_image_to_base64(enh_crop_path),
                    "vlm_text": vlm_text,
                }
                
                crops_data.append(crop_data)

                # YIELD INDIVIDUAL CROP RESULT IMMEDIATELY
                yield json_event("crop_result", crop_data)

            # 4. LLM SUMMARY
            yield status_event("Generating technical summary...")
            
            analysis_parts = []
            if full_page_vlm_text:
                analysis_parts.append("Full-page VLM description:\n" + full_page_vlm_text)

            for i, crop in enumerate(crops_data):
                if crop["vlm_text"]:
                    analysis_parts.append(
                        f"\nCrop {i} (class={crop['cls_name']}):\n{crop['vlm_text']}"
                    )

            analysis_text = "\n".join(analysis_parts)
            
            summary = ""
            try:
                 summary = await asyncio.to_thread(
                    llm_client.predict,
                    analysis_text,
                    256,
                    api_name="/gradio_summarize"
                )
            except Exception as e:
                summary = f"Summary generation failed: {e}"

            # Final Summary Event
            yield json_event("summary", {
                "llm_summary": summary
            })
            
            yield status_event("Analysis complete!")

        except Exception as e:
            # Send error event
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        
        finally:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    # Return the StreamingResponse
    return StreamingResponse(process_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
