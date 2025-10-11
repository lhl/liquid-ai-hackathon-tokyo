#!/usr/bin/env python3

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
import uuid
import argparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="LFM2 Translation Server", version="1.0.0")

# Global variables for model and tokenizer
model = None
tokenizer = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "LiquidAI/LFM2-350M-ENJP-MT"
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

def load_model():
    global model, tokenizer
    print("Loading LFM2 model...")
    
    model_id = "LiquidAI/LFM2-350M-ENJP-MT"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Model loaded successfully!")

def determine_translation_direction(messages: List[ChatMessage]) -> str:
    """Determine translation direction based on the conversation"""
    user_content = ""
    for msg in messages:
        if msg.role == "user":
            user_content += msg.content + " "
    
    # Simple heuristic: if we find Japanese characters, translate to English
    # Otherwise, translate to Japanese
    japanese_chars = any('\u3040' <= char <= '\u309f' or  # Hiragana
                        '\u30a0' <= char <= '\u30ff' or   # Katakana
                        '\u4e00' <= char <= '\u9faf'      # Kanji
                        for char in user_content)
    
    if japanese_chars:
        return "Translate to English."
    else:
        return "Translate to Japanese."

def generate_translation(messages: List[ChatMessage], temperature: float = 0.0, max_tokens: int = 512) -> str:
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Determine translation direction
    system_prompt = determine_translation_direction(messages)
    
    # Get the user's text to translate
    user_text = ""
    for msg in messages:
        if msg.role == "user":
            user_text = msg.content
            break
    
    # Create chat messages with system prompt
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    
    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)
    
    # Generate translation
    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            min_p=0.15 if temperature > 0 else None,
            repetition_penalty=1.05,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (excluding the input)
    generated_tokens = output[0][input_ids.shape[-1]:]
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return translation.strip()

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model)

@app.get("/")
async def root():
    return {"message": "LFM2 Translation Server is running"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "LiquidAI/LFM2-350M-ENJP-MT",
                "object": "model",
                "created": 1234567890,
                "owned_by": "liquid-ai"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
        # Run translation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        translation = await loop.run_in_executor(
            None, 
            generate_translation,
            request.messages,
            request.temperature,
            request.max_tokens
        )
        
        # Create response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=translation),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=0,  # Would need tokenizer to calculate accurately
                completion_tokens=0,  # Would need tokenizer to calculate accurately  
                total_tokens=0
            )
        )
        
        return response
        
    except Exception as e:
        print(f"Error during translation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFM2 Translation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to (default: 8888)")
    
    args = parser.parse_args()
    
    print(f"Starting LFM2 Translation Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
