import zmq
import logging
import argparse
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse arguments for client IP
parser = argparse.ArgumentParser(description="Test inference on a specific client")
parser.add_argument("--client", default="192.168.1.102", help="Client IP address")
parser.add_argument("--question", default="What is diabetes?", help="Medical question to ask")
args = parser.parse_args()

# Setup ZeroMQ PUSH socket to send to client
context = zmq.Context()
socket = context.socket(zmq.PUSH)
client_ip = args.client
socket.connect(f"tcp://{client_ip}:5555")

logger.info(f"Connected to client at {client_ip}:5555")

# Build an inference command
inference_request = {
    "command": "infer",
    "input": args.question
}

# Send the command
socket.send_json(inference_request)
logger.info(f"Sent inference request: \"{args.question}\"")

