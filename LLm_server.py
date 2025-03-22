#!/usr/bin/env python3
import zmq
import logging
import time
import os
import json
import argparse
import random
import signal
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-built medical QA dataset
MEDICAL_QA = {
    "What is diabetes?": "Diabetes is a chronic condition where the body either doesn't produce enough insulin or cannot effectively use the insulin it produces, resulting in elevated blood sugar levels. There are two main types: Type 1 diabetes (where the body doesn't produce insulin) and Type 2 diabetes (where the body becomes resistant to insulin).",
    
    "What are the symptoms of a heart attack?": "Common symptoms of a heart attack include chest pain or discomfort, shortness of breath, pain or discomfort in the jaw, neck, back, arm, or shoulder, and feeling nauseous, light-headed, or unusually tired. Women may experience different symptoms than men, such as more subtle chest pain and more pronounced fatigue and shortness of breath.",
    
    "How does the COVID-19 vaccine work?": "COVID-19 vaccines work by instructing our cells to produce a harmless piece of the virus (like the spike protein), triggering an immune response. This creates antibodies and memory cells that will recognize and fight the actual virus if encountered in the future. Different vaccines use different mechanisms (mRNA, viral vector, etc.) but all aim to build immunity without causing disease.",
    
    "What causes high blood pressure?": "High blood pressure can be caused by multiple factors including unhealthy lifestyle choices (like poor diet, lack of exercise, excessive salt intake, alcohol consumption, and smoking), genetic factors, underlying medical conditions (like kidney disease or thyroid disorders), certain medications, stress, and advancing age.",
    
    "What is the treatment for strep throat?": "Strep throat is typically treated with antibiotics, most commonly penicillin or amoxicillin. These help reduce the duration and severity of symptoms, prevent complications, and reduce transmission to others. Supportive care includes rest, hydration, gargling with saltwater, and using over-the-counter pain relievers for fever and throat pain.",
    
    "What is cancer and how does it develop?": "Cancer is a disease characterized by abnormal cell growth that can invade and spread to other parts of the body. It develops when genetic mutations affect cells' ability to control growth and division. These mutations may be inherited, caused by environmental factors, or occur spontaneously during cell replication. As mutations accumulate, cells can become increasingly abnormal, potentially leading to malignant tumors.",
    
    "How do antibiotics work?": "Antibiotics work by targeting structures or processes that are essential for bacterial survival but not found in human cells. Some antibiotics disrupt the bacterial cell wall synthesis, others interfere with protein synthesis, DNA replication, or other vital cellular processes. This selectivity allows antibiotics to kill or inhibit bacteria without harming human cells.",
    
    "What is Alzheimer's disease?": "Alzheimer's disease is a progressive neurological disorder that causes brain cells to degenerate and die, leading to a decline in cognitive function and memory. It's characterized by abnormal protein deposits in the brain (amyloid plaques and tau tangles), which disrupt communication between neurons and eventually cause cell death. As the disease progresses, brain tissue shrinks significantly.",
    
    "What causes allergies?": "Allergies occur when your immune system reacts to a foreign substance (allergen) that's typically harmless to most people. When first exposed to an allergen, the immune system produces antibodies that identify the particular allergen as harmful, even though it isn't. Upon subsequent exposure, these antibodies signal the immune system to release chemicals like histamine, causing allergy symptoms.",
    
    "How does vaccination provide immunity?": "Vaccination provides immunity by introducing a safe, modified version of a pathogen (or parts of it) to the body. This stimulates the immune system to produce antibodies and memory cells specific to that pathogen, without causing the actual disease. When the real pathogen is encountered later, the immune system recognizes it and mounts a rapid, effective response, preventing infection or reducing its severity."
}

# Additional questions for variety
GENERAL_QUESTIONS = [
    "What is a virus?",
    "How does chemotherapy work?",
    "What are the symptoms of appendicitis?",
    "How does insulin regulate blood sugar?",
    "What causes migraines?",
    "What is the function of antibodies?",
    "How do vaccines prevent disease?",
    "What is the difference between a bacterial and viral infection?",
    "How does blood pressure work?",
    "What is the purpose of cholesterol in the body?"
]

# Generic answers for questions not in our dataset
GENERIC_ANSWERS = [
    "Based on medical literature, this condition involves complex interactions between genetic and environmental factors. Treatment typically follows a multifaceted approach focusing on symptom management and addressing underlying causes when possible.",
    
    "According to recent medical research, this involves a cascade of physiological responses regulated by multiple body systems. Healthcare providers typically diagnose this through a combination of patient history, physical examination, and laboratory tests.",
    
    "Medical consensus indicates this is a multifactorial process with both modifiable and non-modifiable risk factors. Prevention strategies focus on lifestyle modifications while treatment approaches are tailored to individual patient characteristics.",
    
    "Current medical guidelines recommend a stepwise approach to management, beginning with non-pharmacological interventions before considering medication. Patient education plays a crucial role in successful outcomes.",
    
    "The pathophysiology involves complex interactions between various biological systems. Early detection is key to improving outcomes, and treatment approaches continue to evolve as new research emerges."
]

class DemoDistributedLLMServer:
    def __init__(self, clients, mode="train"):
        """Initialize the demo distributed LLM server"""
        self.clients = clients
        self.mode = mode
        self.running = True
        self.model_name = "facebook/opt-1.3b (Simulated)"
        
        # Create ZeroMQ context and sockets for realistic networking
        self.context = zmq.Context()
        self.command_sockets = {}
        self.result_socket = self.context.socket(zmq.PULL)
        self.result_socket.bind("tcp://*:5557")
        
        for client in self.clients:
            socket = self.context.socket(zmq.PUSH)
            socket.connect(f"tcp://{client['ip']}:{client['port']}")
            self.command_sockets[client['id']] = socket
            
        logger.info(f"Server initialized with simulated model {self.model_name}")
        logger.info(f"Clients: {', '.join([c['ip'] for c in self.clients])}")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
    
    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.shutdown_clients()
        sys.exit(0)
    
    def simulate_training(self):
        """Simulate distributed training process"""
        logger.info("Preparing to distribute model to clients...")
        
        # Send setup info to clients
        for client in self.clients:
            client_id = client['id']
            socket = self.command_sockets[client_id]
            
            logger.info(f"Sending setup info to client {client_id}")
            socket.send_json({
                'command': 'setup',
                'model_name': self.model_name,
                'mode': 'train',
                'client_id': client_id,
                'total_clients': len(self.clients)
            })
        
        logger.info("Sending medical training data to clients...")
        
        # Simulate waiting for clients to load models
        logger.info("Waiting for clients to load models...")
        for i, client in enumerate(self.clients, 1):
            # Simulate time delay for model loading
            time.sleep(2)
            logger.info(f"Client {client['id']} is ready ({i}/{len(self.clients)})")
        
        # Start training simulation
        logger.info("\n" + "="*60)
        logger.info("STARTING DISTRIBUTED TRAINING ON MEDICAL DATASET")
        logger.info("="*60)
        
        # Show progress
        epochs = 3
        for epoch in range(1, epochs+1):
            logger.info(f"\nEpoch {epoch}/{epochs}:")
            
            # Simulate progress for each client
            for client in self.clients:
                client_id = client['id']
                
                # Simulate training steps
                steps = 10
                for step in range(1, steps+1):
                    loss = 2.5 - (0.2 * epoch) - (0.02 * step) + (random.random() * 0.1)
                    loss = max(0.5, loss)  # Keep loss reasonable
                    
                    logger.info(f"Client {client_id} - Step {step}/{steps} - Loss: {loss:.4f}")
                    time.sleep(0.5)  # Slow down to make it look real
            
            # End of epoch
            logger.info(f"Epoch {epoch} complete. Validation accuracy: {50 + 15*epoch + random.randint(0,5)}%")
            
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("Model saved to ./trained_models/medical_model")
        logger.info("="*60 + "\n")
        
        # Save placeholder files to simulate output
        os.makedirs("trained_models", exist_ok=True)
        os.makedirs("trained_models/medical_model", exist_ok=True)
        
        with open("trained_models/medical_model/training_complete.txt", "w") as f:
            f.write(f"Training completed on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Medical dataset with {len(MEDICAL_QA)} examples\n")
            f.write(f"Final validation accuracy: {95 + random.randint(0,4)}%\n")
        
        logger.info("Press Enter to continue to inference mode...")
        input()
        
        # Switch to inference mode
        self.mode = "inference"
        self.run_inference()
    
    def run_inference(self):
        """Run simulated inference"""
        logger.info("\n" + "="*60)
        logger.info("STARTING DISTRIBUTED INFERENCE")
        logger.info("="*60 + "\n")
        
        logger.info("Loading trained medical model...")
        time.sleep(2)
        
        for client in self.clients:
            logger.info(f"Client {client['id']} ready for inference")
        
        # Show available questions
        logger.info("\nAvailable medical questions you can ask:")
        for i, question in enumerate(list(MEDICAL_QA.keys())[:5], 1):
            logger.info(f"{i}. {question}")
        logger.info("(You can also ask your own medical questions)")
        
        # Start inference loop
        while self.running:
            try:
                user_input = input("\nEnter your medical question (or 'exit' to quit): ")
                
                if user_input.lower() == 'exit':
                    break
                
                if not user_input.strip():
                    continue
                
                # Select a client (round-robin)
                client_id = random.choice(self.clients)['id']
                logger.info(f"Sending query to client {client_id}...")
                
                # Simulate processing time
                processing_time = random.uniform(1.0, 3.0)
                time.sleep(processing_time)
                
                # Get answer from our dataset or generate a generic one
                if user_input in MEDICAL_QA:
                    answer = MEDICAL_QA[user_input]
                else:
                    # Check for similar questions (case-insensitive)
                    found = False
                    for q, a in MEDICAL_QA.items():
                        if user_input.lower() in q.lower() or q.lower() in user_input.lower():
                            answer = a
                            found = True
                            break
                    
                    if not found:
                        # Generate a generic answer
                        answer = random.choice(GENERIC_ANSWERS)
                
                # Print the answer
                print(f"\nResponse: {answer}")
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
        
        logger.info("Inference session ended")
    
    def shutdown_clients(self):
        """Simulate shutting down clients"""
        logger.info("Shutting down distributed system...")
        time.sleep(1)
        logger.info("All clients have disconnected")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo Distributed Medical LLM")
    parser.add_argument("--mode", choices=["train", "inference"], default="train",
                      help="Mode of operation (train or inference)")
    parser.add_argument("--clients", type=int, default=2,
                      help="Number of simulated clients")
    
    args = parser.parse_args()
    
    # Create fake client list
    clients = []
    for i in range(1, args.clients + 1):
        clients.append({
            'id': i,
            'ip': f'192.168.1.{100+i}',
            'port': 5555
        })
    
    # Create server
    server = DemoDistributedLLMServer(
        clients=clients,
        mode=args.mode
    )
    
    try:
        # Run simulation based on mode
        if args.mode == "train":
            server.simulate_training()
        else:
            server.run_inference()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        server.shutdown_clients()

if __name__ == "__main__":
    main()