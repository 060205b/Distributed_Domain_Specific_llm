#!/usr/bin/env python3
import zmq
import logging
import time
import os
import json
import random
import argparse
import signal
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoLLMClient:
    def __init__(self, listen_port=5555, server_ip="192.168.1.100", data_dir="./client_data"):
        """Initialize the demo LLM client"""
        self.listen_port = listen_port
        self.server_ip = server_ip
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create ZeroMQ context and sockets
        self.context = zmq.Context()
        
        # Command socket (for receiving commands from server)
        self.command_socket = self.context.socket(zmq.PULL)
        self.command_socket.bind(f"tcp://*:{listen_port}")
        
        # Result socket (for sending results to server)
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.connect(f"tcp://{server_ip}:5557")
        
        # Client state
        self.client_id = None
        self.total_clients = None
        self.model_name = None
        self.mode = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
        logger.info(f"Client initialized, listening on port {listen_port}")
    
    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)
    
    def run(self):
        """Run the client, simulating processing commands from the server"""
        logger.info("Client starting, waiting for commands...")
        
        running = True
        while running:
            try:
                # Wait for a command from the server
                command = self.command_socket.recv_json()
                cmd = command.get('command')
                
                if cmd == 'setup':
                    self.handle_setup(command)
                elif cmd == 'csv_data':
                    self.simulate_csv_data(command)
                elif cmd == 'train':
                    self.simulate_training(command)
                elif cmd == 'prepare_inference':
                    self.simulate_prepare_inference(command)
                elif cmd == 'infer':
                    self.simulate_inference(command)
                elif cmd == 'shutdown':
                    logger.info("Received shutdown command")
                    running = False
                else:
                    logger.warning(f"Unknown command: {cmd}")
            
            except KeyboardInterrupt:
                logger.info("Client interrupted by user")
                running = False
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                # Continue running despite errors
        
        logger.info("Client shutting down")
    
    def handle_setup(self, command):
        """Handle initial setup command from server"""
        self.model_name = command.get('model_name')
        self.mode = command.get('mode')
        self.client_id = command.get('client_id')
        self.total_clients = command.get('total_clients')
        
        logger.info(f"Received setup command: model={self.model_name}, mode={self.mode}")
        logger.info(f"I am client {self.client_id} of {self.total_clients}")
        
        # Simulate model loading
        logger.info(f"Loading model {self.model_name}...")
        
        # Simulate loading steps
        logger.info("Initializing model parameters...")
        time.sleep(1)
        
        logger.info("Loading tokenizer...")
        time.sleep(0.5)
        
        logger.info("Moving model to GPU...")
        time.sleep(1.5)
        
        # Tell server we're ready
        logger.info("Model loaded successfully")
        self.result_socket.send_json({
            'command': 'ready',
            'client_id': self.client_id
        })
    
    def simulate_csv_data(self, command):
        """Simulate receiving CSV data"""
        logger.info("Receiving medical dataset from server")
        
        # Simulate file saving
        time.sleep(0.5)
        logger.info("Dataset saved to client_data/medical_qa.csv")
        
        # Create a simulated CSV file
        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, "medical_qa.csv"), "w") as f:
            f.write("type,question,answer\n")
            f.write("medical,What is diabetes?,Diabetes is a chronic condition...\n")
            f.write("medical,What are the symptoms of a heart attack?,Common symptoms include...\n")
            # Add more simulated rows
        
        # Log sample info
        logger.info("Dataset contains 100 medical Q&A pairs")
        logger.info("Sample: Q: What is diabetes? A: Diabetes is a chronic condition...")
    
    def simulate_training(self, command):
        """Simulate training process"""
        logger.info("Starting training process")
        epochs = command.get('epochs', 3)
        
        try:
            # Simulate dataset preparation
            logger.info("Preparing dataset...")
            time.sleep(1)
            logger.info("Tokenizing examples...")
            time.sleep(1)
            
            # Simulate CUDA issues first time (will succeed on retry)
            if random.random() < 0.5 and not os.path.exists(os.path.join(self.data_dir, "retry_flag")):
                # Create a flag file to avoid repeating the error
                with open(os.path.join(self.data_dir, "retry_flag"), "w") as f:
                    f.write("retry_attempted")
                    
                logger.error("CUDA out of memory. Attempting to recover...")
                time.sleep(1)
                logger.info("Reducing batch size and enabling gradient checkpointing")
                time.sleep(1)
                
                self.result_socket.send_json({
                    'command': 'training_error',
                    'client_id': self.client_id,
                    'error': "Training failed, see client logs for details"
                })
                return
            
            # Simulate successful training
            logger.info(f"Starting training for {epochs} epochs...")
            
            # Simulate GPU memory usage
            memory_usage = random.uniform(2.5, 3.2)
            logger.info(f"GPU Memory allocated: {memory_usage:.2f} GB")
            
            # Notify server that training is complete
            self.result_socket.send_json({
                'command': 'training_complete',
                'client_id': self.client_id
            })
            
        except Exception as e:
            logger.error(f"Error during training simulation: {e}")
            # Notify server of failure
            self.result_socket.send_json({
                'command': 'training_error',
                'client_id': self.client_id,
                'error': str(e)
            })
    
    def simulate_prepare_inference(self, command):
        """Simulate preparing for inference"""
        logger.info("Preparing for inference...")
        
        # Simulate loading model for inference
        logger.info("Loading trained medical model...")
        time.sleep(1.5)
        logger.info("Setting model to evaluation mode")
        time.sleep(0.5)
        
        # Notify server we're ready for inference
        self.result_socket.send_json({
            'command': 'inference_ready',
            'client_id': self.client_id
        })
    
    def simulate_inference(self, command):
        """Simulate inference request"""
        user_input = command.get('input', '')
        logger.info(f"Received inference request: {user_input[:50]}...")
        
        try:
            # Simulate processing
            logger.info("Tokenizing input...")
            time.sleep(0.2)
            
            logger.info("Running model inference...")
            time.sleep(random.uniform(0.5, 1.5))
            
            logger.info("Decoding output...")
            time.sleep(0.2)
            
            # Simulated response (this won't be used, the server has the actual responses)
            output_text = "Based on medical knowledge, this involves a complex interaction of factors..."
            
            # Send response to server
            self.result_socket.send_json({
                'command': 'inference_result',
                'client_id': self.client_id,
                'output': output_text
            })
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            # Send error to server
            self.result_socket.send_json({
                'command': 'inference_error',
                'client_id': self.client_id,
                'error': str(e)
            })

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo LLM Client")
    parser.add_argument("--port", type=int, default=5555,
                      help="Port to listen on")
    parser.add_argument("--server", default="192.168.1.100",
                      help="Server IP address")
    parser.add_argument("--data-dir", default="./client_data",
                      help="Directory to store data")
    
    args = parser.parse_args()
    
    # Create client
    client = DemoLLMClient(
        listen_port=args.port,
        server_ip=args.server,
        data_dir=args.data_dir
    )
    
    # Run client
    client.run()

if __name__ == "__main__":
    main()