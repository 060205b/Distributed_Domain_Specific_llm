# Distributed Medical LLM System

This project implements a **Distributed Medical Language Model (LLM)** system using a client-server architecture. The goal is to perform real-time training and inference on medical question-answering tasks by distributing the workload across multiple machines. Each client node runs a real transformer model, coordinated centrally by a server.

---

## 🚀 Project Overview

The system leverages distributed computing to train and run a medical Q&A model across multiple machines using **ZeroMQ** for communication. It is designed to demonstrate how real transformer-based LLMs can be fine-tuned and queried in a distributed setup.

- 🧠 **Model Used**: `distilbert-base-uncased`
- 🖥️ **Distributed Setup**:
  - 1 Server (controller)
  - 6 Clients (workers)
- 🔗 **Communication Protocol**: ZeroMQ
- 💬 **Frontend**: HTML-based dashboard for control & visualization
- 📚 **Dataset**: Preloaded Q&A pairs for training (can be extended)

---

## 🧱 Architecture

```bash
📡 Server
 ├── Sends setup, training, and inference commands
 ├── Receives results from all clients
 └── Displays logs & status updates

🧠 Clients (x6)
 ├── Load transformer model
 ├── Fine-tune on provided medical data
 ├── Respond to inference queries
```

---

## 📁 Project Structure

```bash
├── client.py           # Real training + inference client
├── server.py           # Main controller to manage all clients
├── test.py             # Utility to send a test inference to a client
├── index.html          # Dashboard UI
├── server_debug.log    # Sample log file from the server
├── README.md           # This file
```

---

## 🔧 Technologies Used

- **Python 3**
- **HuggingFace Transformers** (`distilbert-base-uncased`)
- **Datasets** (for Q&A)
- **PyTorch**
- **ZeroMQ** (for messaging)
- **HTML/CSS/JS** (UI dashboard)

---

## 💬 Sample Questions Supported

```text
- What is diabetes?
- What are the symptoms of a heart attack?
- How does the COVID-19 vaccine work?
- What causes high blood pressure?
- What is Alzheimer’s disease?
```

---

## 🎯 Features

- ✅ Distributed transformer model training
- ✅ Real-time inference from multiple nodes
- ✅ ZeroMQ-based message passing
- ✅ Dashboard for question input and logs
- ✅ Logging and test utilities

---

## 🛠️ How to Run

### 1. Install Requirements
```bash
pip install transformers datasets torch pyzmq
```

### 2. Start Clients (on each of the 6 client machines)
```bash
python client.py --port 5555 --server_ip <server-ip>
```

### 3. Start Server
```bash
python server.py --client_ips 192.168.1.101 192.168.1.102 ...
```

### 4. Launch Dashboard
Just open `index.html` in your browser to interact with the system.

### 5. Run Test
```bash
python test.py --client <client-ip> --question "What is diabetes?"
```

---

## 📷 Screenshot

![Dashboard](screenshots/dashboard.png)

---

## 🐛 Known Issues

- If `bitsandbytes` isn't installed, quantized model loading may fail (not needed for distilbert).
- Some clients may not connect if network config is incorrect.
- Inference must be triggered after all clients are marked ready.

---

## 📌 Future Enhancements

- Web API layer for frontend-backend connection
- Docker containers for easier deployment
- Advanced model options (e.g., Mistral, LLaMA)
- Automatic data sharding across clients

---

## 👨‍💻 Contributors

- **Lead Developer**: [Your Name Here]
- **Team**: [Add team members if applicable]

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, adapt, and contribute!
