# 🧠 Distributed Medical LLM System

A distributed medical question-answering system that runs real-time training and inference of transformer models across multiple machines. Built using **ZeroMQ**, **HuggingFace Transformers**, and a custom **HTML-based dashboard**, this setup demonstrates a scalable LLM micro-infrastructure tailored for healthcare Q&A.

---

## 🚀 Project Overview

This system is structured using a **client-server architecture**:

- **Server**: Sends instructions, receives model results, monitors system health.
- **Clients (x6)**: Load a real transformer model, train on medical Q&A data, respond to inference queries.
- **Frontend**: Browser-based dashboard for user input, status monitoring, and result display.

---

## ⚙️ System Specs

- 🧠 **Model**: `distilbert-base-uncased` (can be swapped with larger models)
- 🌐 **Communication**: ZeroMQ sockets (REQ/REP)
- 🖥️ **Clients**: 6 nodes doing real work (not mockups)
- 📡 **Server**: Central orchestrator with logging
- 🧾 **Dataset**: Predefined medical Q&A pairs
- 💻 **Frontend**: HTML/CSS + JS Dashboard (for triggering & visualizing inference)

---

## 📁 Updated Project Structure

```bash
project-root/
│
├── client.py              # Handles model loading, training, and inference
├── server.py              # Controls clients, coordinates training/inference
├── test.py                # Utility script for testing single-client inference
│
├── index.html             # UI dashboard for sending questions and viewing results
├── server_debug.log       # Server logs for debugging/tracking
│
├── static/
│   ├── css/               # (Optional) Styling for the dashboard
│   └── js/                # JS logic for dashboard interactivity
│
├── screenshots/           # UI snapshots of system in action
│   ├── Dashboard.png
│   ├── Dashboard2.png
│   ├── Form.png
│   ├── Table.png
│   ├── DetailView.png
│   └── Modal.png
│
└── README.md              # This file
```

## UI Snapshots

#  Dashboard View
![Demo_Screenshot](https://github.com/user-attachments/assets/9c3c6b11-9b26-4414-b05c-80cee1bc737d)

# Table View
![Demo_Screenshot1](https://github.com/user-attachments/assets/34e04d69-a0d7-404b-a65e-e257616a9159)

# Detail View
![Demo_Screenshot2](https://github.com/user-attachments/assets/d7bcc271-8271-47f2-be5c-e22ccb1d4a7b)

# Form Page 
![Demo_Screenshot3](https://github.com/user-attachments/assets/77305fa0-b016-4ec4-92d1-54992c92318c)

# Modal (Popup) 
![Demo_Screenshot4](https://github.com/user-attachments/assets/3bc1a998-8f59-4291-ab6b-da91fffe106f)

# Dashboard Analytics 
![Demo_Screenshot6](https://github.com/user-attachments/assets/83578baf-19f2-4b11-891f-0503ec33b759)
![Demo_Screenshot5](https://github.com/user-attachments/assets/1bee894f-a2b8-4667-ba27-393609fc7d00)

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

## 🖼️ UI Snapshots

#### 🔧 Dashboard View *(Screenshot: `Demo_Screenshot.png`)*
Shows model activity, client status, and system-wide controls.  
![Dashboard](./screenshots/Demo_Screenshot.png)

---

#### 📋 Table View *(Screenshot: `Demo_Screenshot1.png`)*
Displays history of questions asked and their answers.  
![Table](./screenshots/Demo_Screenshot1.png)

---

#### 📄 Detail View *(Screenshot: `Demo_Screenshot2.png`)*
Expanded single Q&A record with model response details.  
![Detail View](./screenshots/Demo_Screenshot2.png)

---

#### ➕ Form Page *(Screenshot: `Demo_Screenshot3.png`)*
For submitting new questions or test cases.  
![Form Page](./screenshots/Demo_Screenshot3.png)

---

#### 💬 Modal (Popup) *(Screenshot: `Demo_Screenshot4.png`)*
Used for client status alerts or response previews.  
![Modal](./screenshots/Demo_Screenshot4.png)

---

#### 📊 Dashboard Analytics (Alt View) *(Screenshot: `Demo_Screenshot5.png`)*
Widget-based view of client connections, model load, and activity logs.  
![Alt Dashboard](./screenshots/Demo_Screenshot5.png)

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
