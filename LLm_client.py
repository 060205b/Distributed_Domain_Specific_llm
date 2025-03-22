import zmq
import logging
import time
import os
import json
import random
import argparse
import signal
import sys

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import Dataset
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealLLMClient:
    def __init__(self, listen_port=5555, server_ip="192.168.1.100", data_dir="./client_data"):
        self.listen_port = listen_port
        self.server_ip = server_ip
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.context = zmq.Context()

        self.command_socket = self.context.socket(zmq.PULL)
        self.command_socket.bind(f"tcp://*:{listen_port}")

        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.connect(f"tcp://{server_ip}:5557")

        self.client_id = None
        self.total_clients = None
        self.model_name = "distilbert-base-uncased"
        self.mode = None
        self.model = None
        self.tokenizer = None

        signal.signal(signal.SIGINT, self.handle_interrupt)

        logger.info(f"Client initialized, listening on port {listen_port}")

    def handle_interrupt(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    def run(self):
        logger.info("Client waiting for commands...")
        running = True

        while running:
            try:
                command = self.command_socket.recv_json()
                cmd = command.get('command')

                if cmd == 'setup':
                    self.handle_setup(command)
                elif cmd == 'csv_data':
                    self.load_csv_data()
                elif cmd == 'train':
                    self.train_model(command)
                elif cmd == 'prepare_inference':
                    self.prepare_inference()
                elif cmd == 'infer':
                    self.run_inference(command)
                elif cmd == 'shutdown':
                    logger.info("Received shutdown command")
                    running = False
                else:
                    logger.warning(f"Unknown command: {cmd}")

            except KeyboardInterrupt:
                logger.info("Client interrupted by user")
                running = False
            except Exception as e:
                logger.error(f"Error in client run loop: {e}")

        logger.info("Client shutting down")

    def handle_setup(self, command):
        self.model_name = command.get('model_name', self.model_name)
        self.mode = command.get('mode')
        self.client_id = command.get('client_id')
        self.total_clients = command.get('total_clients')

        logger.info(f"Setup received: model={self.model_name}, mode={self.mode}")
        logger.info("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        logger.info("Model ready.")

        self.result_socket.send_json({
            'command': 'ready',
            'client_id': self.client_id
        })

    def load_csv_data(self):
        logger.info("Simulating CSV load with basic examples...")
        filepath = os.path.join(self.data_dir, "medical_qa.csv")
        with open(filepath, "w") as f:
            f.write("question,answer,label\n")
            f.write("What is diabetes?,A chronic condition,1\n")
            f.write("How does COVID vaccine work?,Triggers immunity,0\n")
        logger.info("CSV dataset saved")

    def train_model(self, command):
        logger.info("Beginning actual training...")
        epochs = command.get("epochs", 3)

        try:
            data = {
                "question": ["What is diabetes?", "How does COVID vaccine work?"],
                "answer": ["A chronic condition", "Triggers immune response"],
                "label": [1, 0]
            }
            dataset = Dataset.from_dict(data)

            def preprocess(example):
                return self.tokenizer(example["question"], truncation=True, padding="max_length")

            tokenized = dataset.map(preprocess)
            split = tokenized.train_test_split(test_size=0.2)

            training_args = TrainingArguments(
                output_dir="./model_output",
                per_device_train_batch_size=4,
                num_train_epochs=epochs,
                evaluation_strategy="epoch",
                logging_dir="./logs",
                save_strategy="no",
                report_to="none"
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=split["train"],
                eval_dataset=split["test"]
            )

            trainer.train()
            trainer.save_model("./model_output")
            logger.info("Training finished!")

            self.result_socket.send_json({
                'command': 'training_complete',
                'client_id': self.client_id
            })

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.result_socket.send_json({
                'command': 'training_error',
                'client_id': self.client_id,
                'error': str(e)
            })

    def prepare_inference(self):
        logger.info("Preparing for inference...")
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained("./model_output")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info("Model loaded and ready for inference")

        self.result_socket.send_json({
            'command': 'inference_ready',
            'client_id': self.client_id
        })

    def run_inference(self, command):
        user_input = command.get('input', '')
        logger.info(f"Inference request: {user_input}")

        try:
            clf = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
            result = clf(user_input)[0]
            logger.info(f"Inference output: {result}")

            self.result_socket.send_json({
                'command': 'inference_result',
                'client_id': self.client_id,
                'output': f"Prediction: {result['label']} (confidence: {result['score']:.2f})"
            })

        except Exception as e:
            logger.error(f"Inference error: {e}")
            self.result_socket.send_json({
                'command': 'inference_error',
                'client_id': self.client_id,
                'error': str(e)
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real LLM Client")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--server_ip", type=str, default="192.168.1.100")
    args = parser.parse_args()

    client = RealLLMClient(listen_port=args.port, server_ip=args.server_ip)
    client.run()
