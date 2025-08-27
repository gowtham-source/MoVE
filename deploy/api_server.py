"""MoVE Model API Server

Provides REST API endpoints for MoVE model inference:
- Text completion
- Chat interface
- Batch processing
- Model management

Built with FastAPI for high performance.
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from move import create_move_model
from move_large import create_move_model_large
from inference_optimized import OptimizedMoVE, InferenceOptimizer, GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
model_store = {}
tokenizer_store = {}

# Request/Response Models
class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text completion")
    max_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Stream response")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat conversation history")
    max_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    stream: bool = Field(False, description="Stream response")

class BatchCompletionRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts for batch processing")
    max_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")

class CompletionResponse(BaseModel):
    text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    model_name: str = Field(..., description="Model used for generation")

class ChatResponse(BaseModel):
    message: ChatMessage = Field(..., description="Assistant's response message")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    model_name: str = Field(..., description="Model used for generation")

class BatchCompletionResponse(BaseModel):
    results: List[CompletionResponse] = Field(..., description="Batch completion results")
    total_time: float = Field(..., description="Total processing time")
    average_time_per_prompt: float = Field(..., description="Average time per prompt")

class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    parameters: int = Field(..., description="Number of parameters")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    optimization_level: str = Field(..., description="Optimization level")
    loaded: bool = Field(..., description="Whether model is loaded")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    models_loaded: int = Field(..., description="Number of models loaded")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_memory_used: Optional[float] = Field(None, description="GPU memory used (MB)")
    gpu_memory_total: Optional[float] = Field(None, description="Total GPU memory (MB)")

# Model management functions
def load_model(model_name: str, model_path: str, model_type: str = 'move', 
               optimization_level: str = 'medium') -> bool:
    """Load a model into memory."""
    try:
        logger.info(f"Loading model {model_name} from {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        if model_type == 'move':
            base_model = create_move_model('medium')
        elif model_type == 'move_large':
            base_model = create_move_model_large('1b')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optimize model
        optimized_model = InferenceOptimizer.optimize_model(base_model, optimization_level)
        optimized_model = optimized_model.cuda() if torch.cuda.is_available() else optimized_model
        optimized_model.eval()
        
        # Store model and tokenizer
        model_store[model_name] = {
            'model': optimized_model,
            'type': model_type,
            'optimization_level': optimization_level,
            'parameters': sum(p.numel() for p in base_model.parameters()),
            'loaded_at': time.time()
        }
        tokenizer_store[model_name] = tokenizer
        
        logger.info(f"Model {model_name} loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return False

def unload_model(model_name: str) -> bool:
    """Unload a model from memory."""
    try:
        if model_name in model_store:
            del model_store[model_name]
            del tokenizer_store[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Model {model_name} unloaded")
            return True
        return False
    except Exception as e:
        logger.error(f"Error unloading model {model_name}: {e}")
        return False

def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get information about a loaded model."""
    if model_name not in model_store:
        return None
    
    model_data = model_store[model_name]
    model = model_data['model']
    
    return ModelInfo(
        name=model_name,
        type=model_data['type'],
        parameters=model_data['parameters'],
        memory_usage=model.get_memory_usage(),
        optimization_level=model_data['optimization_level'],
        loaded=True
    )

# Generation functions
def generate_completion(model_name: str, request: CompletionRequest) -> CompletionResponse:
    """Generate text completion."""
    if model_name not in model_store:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = model_store[model_name]['model']
    tokenizer = tokenizer_store[model_name]
    
    # Create generation config
    config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Tokenize input
    device = next(model.parameters()).device
    input_ids = tokenizer(request.prompt, return_tensors='pt')['input_ids'].to(device)
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(input_ids, config)
    
    generation_time = time.time() - start_time
    
    # Decode output
    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    # Apply stop sequences
    if request.stop_sequences:
        for stop_seq in request.stop_sequences:
            if stop_seq in generated_text:
                generated_text = generated_text.split(stop_seq)[0]
                break
    
    tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
    
    return CompletionResponse(
        text=generated_text,
        prompt=request.prompt,
        tokens_generated=tokens_generated,
        generation_time=generation_time,
        model_name=model_name
    )

def generate_chat_response(model_name: str, request: ChatRequest) -> ChatResponse:
    """Generate chat response."""
    if model_name not in model_store:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Format chat messages into prompt
    prompt = ""
    for message in request.messages:
        if message.role == "system":
            prompt += f"System: {message.content}\n"
        elif message.role == "user":
            prompt += f"User: {message.content}\n"
        elif message.role == "assistant":
            prompt += f"Assistant: {message.content}\n"
    
    prompt += "Assistant:"
    
    # Create completion request
    completion_request = CompletionRequest(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        stop_sequences=["User:", "System:"],
        stream=request.stream
    )
    
    # Generate completion
    completion = generate_completion(model_name, completion_request)
    
    # Create chat message
    assistant_message = ChatMessage(
        role="assistant",
        content=completion.text.strip()
    )
    
    return ChatResponse(
        message=assistant_message,
        tokens_generated=completion.tokens_generated,
        generation_time=completion.generation_time,
        model_name=model_name
    )

# Startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting MoVE API Server")
    
    # Load default model if available
    default_model_path = os.getenv('DEFAULT_MODEL_PATH')
    if default_model_path and os.path.exists(default_model_path):
        load_model('default', default_model_path, 'move', 'medium')
    
    yield
    
    # Shutdown
    logger.info("Shutting down MoVE API Server")
    
    # Unload all models
    for model_name in list(model_store.keys()):
        unload_model(model_name)

# Create FastAPI app
app = FastAPI(
    title="MoVE Model API",
    description="REST API for MoVE (Mixture of Vector Experts) language models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_memory_used = None
    gpu_memory_total = None
    
    if gpu_available:
        gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(model_store),
        gpu_available=gpu_available,
        gpu_memory_used=gpu_memory_used,
        gpu_memory_total=gpu_memory_total
    )

# Model management endpoints
@app.post("/models/{model_name}/load")
async def load_model_endpoint(model_name: str, model_path: str, 
                            model_type: str = 'move', optimization_level: str = 'medium'):
    """Load a model."""
    success = load_model(model_name, model_path, model_type, optimization_level)
    if success:
        return {"message": f"Model {model_name} loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")

@app.delete("/models/{model_name}")
async def unload_model_endpoint(model_name: str):
    """Unload a model."""
    success = unload_model(model_name)
    if success:
        return {"message": f"Model {model_name} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded models."""
    models = []
    for model_name in model_store.keys():
        model_info = get_model_info(model_name)
        if model_info:
            models.append(model_info)
    return models

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info_endpoint(model_name: str):
    """Get information about a specific model."""
    model_info = get_model_info(model_name)
    if model_info:
        return model_info
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

# Generation endpoints
@app.post("/models/{model_name}/completions", response_model=CompletionResponse)
async def create_completion(model_name: str, request: CompletionRequest):
    """Generate text completion."""
    return generate_completion(model_name, request)

@app.post("/models/{model_name}/chat", response_model=ChatResponse)
async def create_chat_completion(model_name: str, request: ChatRequest):
    """Generate chat completion."""
    return generate_chat_response(model_name, request)

@app.post("/models/{model_name}/batch", response_model=BatchCompletionResponse)
async def create_batch_completion(model_name: str, request: BatchCompletionRequest):
    """Generate batch completions."""
    start_time = time.time()
    results = []
    
    for prompt in request.prompts:
        completion_request = CompletionRequest(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        result = generate_completion(model_name, completion_request)
        results.append(result)
    
    total_time = time.time() - start_time
    average_time = total_time / len(request.prompts) if request.prompts else 0
    
    return BatchCompletionResponse(
        results=results,
        total_time=total_time,
        average_time_per_prompt=average_time
    )

# Convenience endpoints
@app.post("/completions", response_model=CompletionResponse)
async def create_completion_default(request: CompletionRequest):
    """Generate completion using default model."""
    if 'default' not in model_store:
        raise HTTPException(status_code=404, detail="No default model loaded")
    return generate_completion('default', request)

@app.post("/chat", response_model=ChatResponse)
async def create_chat_completion_default(request: ChatRequest):
    """Generate chat completion using default model."""
    if 'default' not in model_store:
        raise HTTPException(status_code=404, detail="No default model loaded")
    return generate_chat_response('default', request)

# Main function
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MoVE Model API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--model_path', type=str, help='Path to default model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['move', 'move_large'], 
                       default='move', help='Type of default model')
    parser.add_argument('--optimization_level', type=str, 
                       choices=['none', 'light', 'medium', 'aggressive'],
                       default='medium', help='Optimization level for default model')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Set default model path as environment variable
    if args.model_path:
        os.environ['DEFAULT_MODEL_PATH'] = args.model_path
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == '__main__':
    main()