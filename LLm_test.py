import zmq
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse arguments for client IP
parser = argparse.ArgumentParser()
parser.add_argument("--client", default="192.168.1.102", help="Client IP address")
args = parser.parse_args()

# Setup socket to send to client
context = zmq.Context()
socket = context.socket(zmq.PUSH)
client_ip = args.client
socket.connect(f"tcp://{client_ip}:5555")

logger.info(f"Server connected to client at {client_ip}:5555")

# Send a test message
test_message = {"command": "test", "message": "Hello from server!"}
socket.send_json(test_message)
logger.info("Test message sent!")