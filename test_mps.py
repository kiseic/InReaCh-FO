#!/usr/bin/env python3
"""
Test script to verify if llama-cpp-python is installed with MPS (Metal) support.
"""

import sys
import importlib.util
from pathlib import Path

def main():
    # Check if llama-cpp-python is installed
    if importlib.util.find_spec("llama_cpp") is None:
        print("❌ llama_cpp module not found. Please install llama-cpp-python.")
        return 1
    
    # Import llama_cpp module
    import llama_cpp
    print(f"✅ Found llama_cpp version: {llama_cpp.__version__}")
    
    # Check if llama_cpp was compiled with Metal support
    print("\nChecking Metal (MPS) support:")
    
    # Get the path to the shared library
    for attribute in dir(llama_cpp.llama_cpp):
        if attribute.startswith("_GGML_"):
            print(f"  Found build attribute: {attribute}")
    
    # Check if n_gpu_layers works
    from llama_cpp import Llama
    try:
        # Create a small dummy model config to test MPS support
        model_config = {
            "n_gpu_layers": -1,  # This should use all available GPU layers if MPS is supported
            "verbose": True,
        }
        
        print("\nTesting Llama initialization with n_gpu_layers:")
        print(f"  Config: {model_config}")
        print("  (Note: This will fail safely if no model is found, we're just checking if the MPS parameter is recognized)")
        
        try:
            # This will fail since we don't have a model, but we just want to see if n_gpu_layers is accepted
            model = Llama(model_path="nonexistent_model.gguf", **model_config)
        except FileNotFoundError:
            print("  ✅ n_gpu_layers parameter was accepted (file not found error is expected)")
        except TypeError as e:
            if "unexpected keyword argument 'n_gpu_layers'" in str(e):
                print("  ❌ n_gpu_layers not recognized - Metal/MPS support might not be enabled")
            else:
                print(f"  ⚠️ Unexpected TypeError: {e}")
        except Exception as e:
            print(f"  ⚠️ Exception when testing: {e}")
        
        # Just check if the parameter exists in the __init__ signature
        import inspect
        sig = inspect.signature(Llama.__init__)
        if "n_gpu_layers" in sig.parameters:
            print("  ✅ n_gpu_layers parameter exists in Llama.__init__")
        else:
            print("  ❌ n_gpu_layers parameter not found in Llama.__init__")
            
    except Exception as e:
        print(f"❌ Error testing MPS support: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())