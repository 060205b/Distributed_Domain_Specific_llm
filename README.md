# Team---C LLM Project

## Project Structure
```
llm_project/
├── data/             # Dummy data (and later, your preprocessed data)
│   └── dummy_data.txt
├── models/           # Trained models and checkpoints
├── onnx_model/       # ONNX exported models
├── .venv/            # Virtual environment (DO NOT COMMIT THIS)
├── api.py            # FastAPI inference server
├── convert_to_onnx.py  # Model conversion script
├── preprocess.py     # Data preprocessing script
├── train.py          # Distributed training script
├── requirements.txt  # Python dependencies
└── static/           # Web Interface
    ├── index.html
    ├── style.css
    └── script.js
```

## Requirements
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

**Dependencies:**
```
transformers
torch
torchvision
torchaudio
accelerate
bitsandbytes
peft
fastapi
uvicorn[standard]
python-multipart
datasets
onnxruntime-gpu  # Or onnxruntime, depending on GPU availability
optimum
deepspeed
scipy
scikit-learn
pypdf2
python-docx
Jinja2
```

## Setup Instructions

### 1. Environment Setup
1. Ensure all laptops are connected to the switch and can communicate.
2. Set up NFS as described previously.
3. Create the project directory structure on each laptop.
4. Create and activate virtual environments on each laptop.
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Data Preprocessing
Run the preprocessing script on **Laptop 1**:
```bash
python preprocess.py
```
This will generate processed data (e.g., `preprocessed.json`) inside `data/preprocessed/` (accessible via NFS).

### 3. Distributed Training
1. Ensure `train.py` is available on Laptop 1 (optional: copy to other laptops).
2. Create a `hostfile` in the project directory with the following content:
   ```
   192.168.1.2 slots=1  # Laptop 2's IP
   192.168.1.3 slots=1  # Laptop 3's IP
   ```
3. Run training on **Laptop 1** using DeepSpeed:
   ```bash
   deepspeed --hostfile hostfile train.py --deepspeed deepspeed_config
   ```

**Monitoring Training:**
Run TensorBoard to monitor logs:
```bash
tensorboard --logdir models/logs
```
Access via browser at: `http://<Laptop 1's IP>:6006`

### 4. ONNX Conversion (Laptop 2 or 3)
1. Copy `convert_to_onnx.py` to a laptop with a GPU.
2. Modify `convert_to_onnx.py` to set `model_id` to the trained model checkpoint path (e.g., `/mnt/llm_data/models/final_checkpoint`).
3. Convert the model:
   ```bash
   python convert_to_onnx.py
   ```
This will generate an `onnx_model/` directory.

### 5. Running the Inference Server (Laptop 2 or 3)
1. Copy `api.py` to the laptop with the ONNX model.
2. Ensure `model_path` in `api.py` points to the `onnx_model` directory.
3. Start the FastAPI server:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

### 6. Setting Up the Web Interface (Laptop 1)
1. Ensure the `static/` directory contains `index.html`, `style.css`, and `script.js`.
2. Modify `api.py` on Laptop 1 to serve static files:
   ```python
   from fastapi.staticfiles import StaticFiles
   app.mount("/", StaticFiles(directory="static", html=True), name="static")
   ```
3. Start the API server on Laptop 1:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

### 7. Testing
1. Open a web browser and navigate to:
   ```
   http://<Laptop 1's IP>:8000
   ```
2. Enter queries and verify chatbot responses.

---

## Contributing
Feel free to submit issues and pull requests to improve this project!

## License
This project is licensed under the MIT License.
