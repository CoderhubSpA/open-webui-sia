"""
title: OpenAI O1-Preview Pipe
version: 0.1.0
license: MIT
"""
import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from open_webui.utils.misc import pop_system_message

class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="")
        
    def __init__(self):
        self.type = "manifold"
        self.id = "openai-o1"
        self.name = "openai/"
        self.valves = self.Valves(
            **{"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")}
        )
        self.llm = ChatOpenAI(
            model="o1-preview",
            temperature=1,
            api_key=self.valves.OPENAI_API_KEY
        )
        
    def get_openai_models(self):
        return [
            {"id": "o1-preview", "name": "o1-preview"}
        ]
        
    def pipes(self) -> List[dict]:
        return self.get_openai_models()
        
    def _convert_messages_to_memory(self, messages: List[dict]) -> ConversationBufferMemory:
        memory = ConversationBufferMemory(return_messages=True)
        for message in messages:
            if message["role"] == "user":
                memory.chat_memory.add_user_message(message.get("content", ""))
            elif message["role"] == "assistant":
                memory.chat_memory.add_ai_message(message.get("content", ""))
        return memory

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])
        
        try:
            # Configurar la memoria con el historial de conversación
            memory = self._convert_messages_to_memory(messages)
            
            # Configurar el modelo con los parámetros del body
            self.llm.temperature = body.get("temperature", 1)
            self.llm.max_tokens = body.get("max_tokens", 4096)
            
            # Crear la cadena de conversación
            conversation = ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=True
            )

            # Obtener el último mensaje del usuario
            last_message = messages[-1].get("content", "") if messages else ""
            
            # Si hay un mensaje del sistema, añadirlo al contexto
            if system_message:
                last_message = f"System: {system_message}\nUser: {last_message}"
            
            # Obtener la respuesta
            response = conversation.predict(input=last_message)
            
            return response
                
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"