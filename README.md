# Engineering Drawing Extractor

A web application to analyze engineering drawing images using Hugging Face Spaces.

## Features
- **Upload**: Upload an engineering drawing image.
- **Analyze**:
    - **YOLO Detection**: Detects components using `jeyanthangj2004/engg-draw-extractor`.
    - **LLM Summary**: Generates a summary using `jeyanthangj2004/eng-draw-llm-flan-t5`.
- **Visualize**: Displays annotated images, crop details (raw & enhanced), and summaries.

## Project Structure
```
engineering-drawing-extractor/
├── backend/
│   ├── app.py              # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Frontend UI (HTML/CSS/JS)
└── README.md
```

## Setup & Installation

### Prerequisites
- Python 3.8+ installed.

### 1. Backend Setup
Navigate to the backend directory:
```bash
cd backend
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Backend
Start the FastAPI server:
```bash
uvicorn app:app --reload
```
The backend will run at `http://127.0.0.1:8000`.

### 3. Using the Frontend
Simply open `frontend/index.html` in your web browser.
- You can double-click the file in your file explorer.
- Or serve it using a simple HTTP server (e.g., `python -m http.server 5500` inside `frontend/`).

## Usage
1. Open the frontend in your browser.
2. Click "Choose File" and select an engineering drawing image.
3. Click "Analyze".
4. Wait for the processing to complete (YOLO and LLM steps).
5. View the annotated image, summary, and extracted crops.
