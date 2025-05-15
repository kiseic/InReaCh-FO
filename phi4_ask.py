#!/usr/bin/env python3
"""
Ask a single question to Phi-4-mini model and get the response.
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phi4_ask")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Ask Phi-4-mini a question")
    parser.add_argument("question", help="The question to ask", nargs="?", default=None)
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    # Check if question is provided
    if args.question is None:
        print("Please provide a question. Example:")
        print("python phi4_ask.py \"What is anomaly detection?\"")
        return 1
    
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
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return 1
    
    # System message
    system_message = "You are Phi-4, a helpful, harmless, and honest AI assistant developed by Microsoft. You answer questions accurately, objectively, and helpfully."
    
    # Format prompt
    question = args.question
    
    prompt = f"""<|system|>
{system_message}

<|user|>
{question}

<|assistant|>
"""
    
    # Generate response
    logger.info(f"Asking: {question}")
    print("\nPhi-4 is thinking...")
    
    start_time = time.time()
    
    result = model(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=0.9,
        stop=["<|user|>", "<|system|>"],
        echo=False
    )
    
    # Extract the response
    response = result["choices"][0]["text"].strip()
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Print response
    print(f"\nPhi-4's response (generated in {generation_time:.2f} seconds):")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())