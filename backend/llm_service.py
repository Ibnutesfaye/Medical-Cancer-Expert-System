"""
LLM service client for answer generation.

Supports Groq and OpenAI with streaming responses.
"""

from typing import AsyncGenerator, Optional
import os
from groq import AsyncGroq
from openai import AsyncOpenAI


class LLMService:
    """
    LLM service client with streaming support.
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 4096
    ):
        """
        Initialize LLM service.
        
        Args:
            groq_api_key: Groq API key
            openai_api_key: OpenAI API key (fallback)
            model_name: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        # Avoid using placeholder dummy keys
        opt_openai = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.openai_api_key = opt_openai if opt_openai and "your_" not in opt_openai.lower() else None
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize clients
        self.groq_client = None
        self._get_groq_client()
        
        self.openai_client = AsyncOpenAI(
            api_key=self.openai_api_key,
            timeout=120.0,
            max_retries=3
        ) if self.openai_api_key else None
        
        # Determine which client to use
        self.use_groq = self.groq_api_key is not None and len(self.groq_api_key.strip()) > 0
    
    def _get_groq_client(self, force_refresh=False) -> Optional[AsyncGroq]:
        """Get or recreate the AsyncGroq client."""
        if not self.groq_api_key:
            return None
        if self.groq_client is None or force_refresh:
            self.groq_client = AsyncGroq(
                api_key=self.groq_api_key,
                timeout=120.0,
                max_retries=3
            )
        return self.groq_client
    
    async def generate_streaming(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate response with streaming.
        
        Args:
            prompt: Input prompt
            
        Yields:
            Response tokens
        """
        if self.use_groq:
            try:
                async for token in self._generate_groq_streaming(prompt):
                    yield token
                return
            except Exception as e:
                if self.openai_client:
                    print(f"Groq API failed ({e}), falling back to OpenAI...")
                else:
                    raise
                    
        if self.openai_client:
            async for token in self._generate_openai_streaming(prompt):
                yield token
        else:
            raise ValueError("No LLM API key configured")
    
    async def _generate_groq_streaming(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate response using Groq with streaming."""
        client = self._get_groq_client()
        if not client:
            raise ValueError("Groq client not initialized")
        try:
            # Groq's streaming is asynchronous here, effectively avoiding event loop blocks
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
        except Exception as e:
            # Recreate client on connection/initial failure and retry once
            print(f"Groq API connection failed: {e}. Retrying with a fresh client connection...")
            try:
                client = self._get_groq_client(force_refresh=True)
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
            except Exception as retry_err:
                raise RuntimeError(f"Groq API error: {str(retry_err)}")

        try:
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
        except Exception as stream_err:
            raise RuntimeError(f"Groq streaming error: {str(stream_err)}")
    
    async def _generate_openai_streaming(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate response using OpenAI with streaming."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    async def generate(self, prompt: str) -> str:
        """
        Generate complete response (non-streaming).
        
        Args:
            prompt: Input prompt
            
        Returns:
            Complete response text
        """
        if self.use_groq:
            client = self._get_groq_client()
            if client:
                try:
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Groq API connection failed: {e}. Retrying with a fresh client connection...")
                    try:
                        client = self._get_groq_client(force_refresh=True)
                        response = await client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        return response.choices[0].message.content
                    except Exception as retry_err:
                        if self.openai_client:
                            print(f"Groq API failed after retry ({retry_err}), falling back to OpenAI...")
                        else:
                            raise RuntimeError(f"Groq API error: {str(retry_err)}")
                    
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"OpenAI API error: {str(e)}")
        else:
            raise ValueError("No LLM API key configured")
