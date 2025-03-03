# Team---C

**Project Structure:**
llm_project/
├── data/             # Dummy data (and later, your preprocessed data)
│   └── dummy_data.txt
├── models/           # Trained models and checkpoints
├── onnx_model/    # ONNX exported models
├── .venv/            # Virtual environment (DO NOT COMMIT THIS)
├── api.py          # FastAPI inference server
├── convert_to_onnx.py  # Model conversion script
├── preprocess.py   # Data preprocessing script
├── train.py        # Distributed training script
├── requirements.txt # Python dependencies
└── static/         #Web Interface
      └── index.html
      └── style.css
      └── script.js

**requirements.txt:**
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

onnxruntime-gpu # Or onnxruntime, depending on GPU availability

optimum

deepspeed

scipy

scikit-learn

pypdf2

python-docx

Jinja2


**Instructions for Running (Step-by-Step):**
Setup:
Ensure all laptops are connected to the switch and can communicate.
Set up NFS as described previously.
Create the project directory structure on each laptop.
Create and activate virtual environments on each laptop.
Install the required packages (using pip install -r requirements.txt) inside the virtual environments.
Data Preprocessing (Continued):

Run the preprocess.py script on Laptop 1:
Bash
python preprocess.py
This will process your raw data and save the preprocessed data (e.g., preprocessed.json) to the data/preprocessed/ directory. Make sure this directory is accessible via NFS.

Distributed Training:
Copy train.py: Make sure train.py is present on Laptop 1.  You might also want to copy it to Laptops 2 and 3 for convenience, but the main execution will be controlled from Laptop 1.
Run train.py (from Laptop 1): This is where the magic of distributed training happens.  Use deepspeed to launch the training script.  You'll need to create a DeepSpeed configuration file (I've already included deepspeed_config in train.py).  Run the following command from your project directory on Laptop 1
Bash
deepspeed --hostfile hostfile train.py --deepspeed deepspeed_config
Create a hostfile: Before you can run it, you will need a hostfile. This file tells DeepSpeed which machines to use.  Create a file named hostfile in your project directory with the following content:
192.168.1.2 slots=1 # Assuming Laptop 2's IP is 192.168.1.2
192.168.1.3 slots=1 # Assuming Laptop 3's IP is 192.168.1.3
192.168.1.2 and 192.168.1.3: Replace these with the actual IP addresses of Laptop 2 and Laptop 3, respectively.
slots=1: This indicates that each laptop has one GPU to use.
--deepspeed deepspeed_config: It is very important and should be used.

Monitoring: Monitor the training progress.  The script is set up to log information every 10 steps and save checkpoints every 500 steps.  You can use TensorBoard to visualize the training logs:
Bash
tensorboard --logdir models/logs  # Or wherever your logs are saved
Open a web browser and go to http://<Laptop 1's IP>:6006 to view the TensorBoard dashboard.

ONNX Conversion (After Training - Laptop 2 or 3):
Once training is complete (or you have a checkpoint you want to use), copy the convert_to_onnx.py script to Laptop 2 or 3 (one with a GPU).
Modify convert_to_onnx.py: Change the model_id variable to point to the directory where your trained model (or LoRA adapters) are saved. This will likely be a subdirectory within the models/ directory on the NFS share (e.g., /mnt/llm_data/models/final_checkpoint).
Run the script:
Bash
python convert_to_onnx.py
This will create an onnx_model directory containing the ONNX model and tokenizer.

Inference Server (Laptop 2 or 3):
Copy the api.py to the onnx model generated laptop.
Make sure the model_path variable in api.py points to the onnx_model directory.
Run the FastAPI server:
Bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000

Web Interface (Laptop 1):
Make sure you have created the static folder with index.html,style.css and script.js files inside your project directory on Laptop 1.
Make sure your api.py on Laptop 1, has these lines. This will serve the static files, and your webpage will be shown:
Python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")
Now run the api.py in the Laptop 1.

Testing:
Open a web browser and navigate to http://<Laptop 1's IP>:8000. You should see your web interface.
Enter questions and verify that the chatbot is responding.
