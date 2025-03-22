Distributed Domain specific(Medical) Language Model System

MediLLM is a distributed training and inference system designed to simulate a large language model (LLM) pipeline for medical question answering. Built with a client-server architecture, this project uses ZeroMQ to coordinate communication across a network of machines, allowing parallel simulation of model training and response generation.

> ⚕️ **Use Case**: Demonstrating how distributed systems can be applied to train and deploy LLMs for specialized domains like healthcare.

---

## 🚀 Project Overview

This project simulates distributed training of a medical question-answering LLM across multiple machines. It involves a centralized **server** that coordinates model setup, training, and inference by communicating with **multiple clients** using ZeroMQ. Each client performs tasks such as model loading, simulated training, and inference.

- 💻 **Total Machines**: 7  
  - 🧠 1 Server (controller)
  - 📡 6 Clients (workers)

- 🧠 **Model Simulated**: `facebook/opt-1.3b` (simulated only)
- ⚙️ **Communication Protocol**: ZeroMQ (PUSH/PULL sockets)
- 🌐 **UI**: Custom dashboard built with HTML/CSS/JS
- 📄 **Dataset**: Pre-built medical Q&A used for simulation

---

## 📷 Demo Screenshots

> Add your actual screenshots inside a `screenshots/` folder in your repo.

### 🖥️ Dashboard UI
![Dashboard Overview](screenshots/dashboard.png)

### 📊 Training Progress
![Training Simulation](screenshots/training.png)

### 💬 Inference Mode
![Medical Q&A Chat](screenshots/inference.png)

---

## 📁 Project Structure

```bash
├── client.py             # Client-side script to simulate training/inference
├── server.py             # Server-side controller script
├── test.py               # Script to test connectivity between server and client
├── index.html            # Frontend dashboard for system control and monitoring
├── server_debug.log      # Sample logs for debugging          
└── README.md
```

## How It Works

1. Server discovers clients on the local network using IP scanning.
2. It distributes setup info (model name, mode, etc.) to clients.
3. Each client simulates:
   - Model loading (with delay to mimic GPU prep)
   - Training across multiple epochs
   - Reporting back simulated loss and validation accuracy
4. Server logs training and transitions into inference mode.
5. Medical questions can be asked via terminal or dashboard.
6. Clients simulate an answer generation process and respond back.

## Features

✔️ Distributed computing using ZeroMQ (PUSH/PULL model)
✔️ Simulated training with logs, loss calculation, and validation accuracy
✔️ Inference mode for real-time medical Q&A interaction
✔️ Frontend dashboard for status visualization and interaction
✔️ Fault-tolerance simulation (e.g., simulated memory errors)
✔️ Clear client-server separation for scalability

## Running the Project

# 1. Install Dependencies
pip install pyzmq

