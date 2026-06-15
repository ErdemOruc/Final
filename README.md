# Plant Health Analysis Pipeline 🌿🍅

This project is a complete deep learning pipeline for agricultural analysis. It combines object detection, image classification, and Large Language Models (LLMs) to automatically detect plant leaves and fruits, diagnose diseases or damage, assess ripeness, and provide actionable agricultural advice.

## 🚀 Features

- **Object Detection (YOLO)**: Detects leaves and fruits in an image.
- **Disease & Damage Classification (Keras)**: 
  - Classifies leaf diseases (e.g., healthy, blight, spot).
  - Assesses fruit damage levels and health (predictions with <= 50% confidence default to Healthy to avoid false alarms).
  - Determines fruit ripeness.
- **AI-Powered Advice (Ollama / LLM)**: Automatically generates practical treatment or handling advice for any detected issues using a local LLM (Can be toggled via `LLM_IS_OPEN` flag).
- **FastAPI Backend**: Provides a fast, easy-to-use REST API for image upload and analysis.
- **Advanced Bounding Box Filtering**: Custom NMS (Non-Maximum Suppression) guarantees no overlapping bounding boxes in detection outputs.
- **Image Annotation**: Returns a detailed JSON report along with an annotated image showing bounding boxes and detected conditions.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Ollama (for LLM Advice):**
   - Install [Ollama](https://ollama.ai/)
   - Pull the required model (default is `phi3`):
     ```bash
     ollama run phi3
     ```
     *(Make sure Ollama is running on `http://localhost:11434`)*
   - To enable the LLM advice in the API, open `main.py` and set `LLM_IS_OPEN = True` (it is `False` by default for faster inference).

## 🚦 Usage

1. **Start the FastAPI server:**
   ```bash
   python main.py
   ```
   *The server will start on `http://localhost:8000`.*

2. **API Endpoints:**
   - `POST /analyze`: Upload an image (`multipart/form-data`) to get a full analysis report.
   - `GET /latest`: Get the JSON result of the last processed image.
   - `GET /latest/image`: Get the annotated image of the last processed request.
   - `GET /health`: Check the status of loaded YOLO models, Keras models, and the LLM service.

## 📂 Project Structure

- `main.py`: The FastAPI application and main pipeline logic.
- `LoadModels.py`: Script to initialize and load YOLO and Keras models into memory.
- `services/`: Contains specific service classes for handling YOLO, Keras predictions, LLM queries, and image annotation.
- `models/`: Pydantic schemas for API responses.
- `YOLO/`: Contains YOLO model scripts and weights.
- `KERAS/`: Contains Keras classification models and their individual training scripts.
  - **Note on Training**: For the `OldDamaged` classification, training is split into two scripts:
    1. `OldDamagedModel.py`: Trains the top layers (Transfer Learning).
    2. `OldDamagedFineTune.py`: Unfreezes the base model and fine-tunes with a low learning rate for higher accuracy.

## 📝 Requirements

See `requirements.txt` for a full list of dependencies. Main libraries include `fastapi`, `tensorflow`, `ultralytics`, `opencv-python`, and `requests`.
