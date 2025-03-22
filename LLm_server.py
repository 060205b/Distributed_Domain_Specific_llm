
import zmq
import logging
import time
import random
import json
import argparse
import signal
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RealLLMServer:
    def __init__(self, clients):
        self.clients = clients
        self.context = zmq.Context()
        self.command_sockets = {}
        self.result_socket = self.context.socket(zmq.PULL)
        self.result_socket.bind("tcp://*:5557")
        self.running = True

        for client in self.clients:
            socket = self.context.socket(zmq.PUSH)
            socket.connect(f"tcp://{client['ip']}:{client['port']}")
            self.command_sockets[client['id']] = socket

        logger.info("Server initialized and ready")
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        logger.info("Interrupt received. Shutting down.")
        self.running = False
        sys.exit(0)

    def send_command(self, client_id, command_dict):
        socket = self.command_sockets[client_id]
        socket.send_json(command_dict)
        logger.info(f"Sent command to client {client_id}: {command_dict['command']}")

    def wait_for_response(self):
        logger.info("Waiting for client response...")
        message = self.result_socket.recv_json()
        logger.info(f"Received: {message}")
        return message

    def setup_all_clients(self):
        for client in self.clients:
            self.send_command(client['id'], {
                "command": "setup",
                "model_name": "facebook/opt-1.3b",
                "mode": "train",
                "client_id": client['id'],
                "total_clients": len(self.clients)
            })
        for _ in self.clients:
            self.wait_for_response()

    def start_training(self):
        for client in self.clients:
            self.send_command(client['id'], {
                "command": "train",
                "epochs": 2
            })

        completed = 0
        while completed < len(self.clients):
            msg = self.wait_for_response()
            if msg["command"] == "training_complete":
                completed += 1

        logger.info("All clients completed training.")

    def prepare_inference(self):
        for client in self.clients:
            self.send_command(client['id'], {"command": "prepare_inference"})

        for _ in self.clients:
            self.wait_for_response()

    def run_inference_loop(self):
        while self.running:
            question = input("Ask a medical question (or type 'exit'): ")
            if question.lower() == "exit":
                break

            client = random.choice(self.clients)
            self.send_command(client['id'], {
                "command": "infer",
                "input": question
            })

            response = self.wait_for_response()
            if response["command"] == "inference_result":
                print(f"\n🧠 Answer from client {response['client_id']}: {response['output']}\n")

    def shutdown(self):
        for client in self.clients:
            self.send_command(client['id'], {"command": "shutdown"})
        logger.info("Shutdown command sent to all clients.")

def main():
    parser = argparse.ArgumentParser(description="LLM Distributed Server")
    parser.add_argument("--client_ips", nargs="+", required=True, help="List of client IPs")
    args = parser.parse_args()

    clients = [{"id": idx + 1, "ip": ip, "port": 5555} for idx, ip in enumerate(args.client_ips)]
    server = RealLLMServer(clients)

    logger.info("Starting distributed setup...")
    server.setup_all_clients()

    logger.info("Starting training on all clients...")
    server.start_training()

    logger.info(" Preparing clients for inference...")
    server.prepare_inference()

    logger.info(" Entering inference mode. Ask your questions below:")
    server.run_inference_loop()

    server.shutdown()

if __name__ == "__main__":
    main()
