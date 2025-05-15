#!/usr/bin/env python3
"""
Simple chat interface for Phi-4-mini model.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phi4_chat")

def main():
    # Import necessary modules
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python not installed. Please install it with: pip install llama-cpp-python")
        return 1
    
    # Model path
    models_dir = Path("models")
    model_path = models_dir / "phi4-mini" / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Please run setup_phi4.py first.")
        return 1
    
    # Initialize model
    logger.info(f"Loading Phi-4-mini model from {model_path}...")
    
    try:
        model = Llama(
            model_path=str(model_path),
            n_ctx=4096,  # Context window size
            n_gpu_layers=-1,  # Use all available GPU layers
            verbose=False
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return 1
    
    # Chat interface
    print("\n=== Phi-4-mini Chat Interface ===")
    print("Type your questions below. Type 'exit' or 'quit' to end the chat.")
    print("Type 'clear' to clear the conversation history.")
    print("-------------------------------------")
    
    # Conversation context
    conversation = []
    
    # System message
    system_message = "You are Phi-4, a helpful, harmless, and honest AI assistant developed by Microsoft. You answer questions accurately, objectively, and helpfully."
    
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Check for clear command
        if user_input.lower() == "clear":
            conversation = []
            print("Conversation history cleared.")
            continue
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})
        
        # Format prompt
        prompt = f"<|system|>\n{system_message}\n\n"
        
        for message in conversation:
            if message["role"] == "user":
                prompt += f"<|user|>\n{message['content']}\n\n"
            else:
                prompt += f"<|assistant|>\n{message['content']}\n\n"
        
        # Add final assistant prompt
        prompt += "<|assistant|>\n"
        
        # Generate response
        print("Phi-4 is thinking...")
        start_time = time.time()
        
        result = model(
            prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=["<|user|>", "<|system|>"],
            echo=False
        )
        
        # Extract the response
        response = result["choices"][0]["text"].strip()
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # Print response
        print(f"\nPhi-4: {response}")
        print(f"\n[Generated in {generation_time:.2f} seconds]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())