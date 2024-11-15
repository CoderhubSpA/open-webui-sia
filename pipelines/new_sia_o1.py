from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import json
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = ""
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "OpenAI Pipeline"
        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv(
                    "OPENAI_API_KEY", "your-openapi-key"
                )
            }
        )
        print("loaded openai_api_key:", self.valves.OPENAI_API_KEY)
        # Inicializar el modelo de Langchain
        self.llm = ChatOpenAI(
            model="o1-preview",
            temperature=1,
            api_key=self.valves.OPENAI_API_KEY
        )
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def _convert_messages_to_memory(self, messages: List[dict]) -> ConversationBufferMemory:
        # Crear una nueva instancia de memoria
        memory = ConversationBufferMemory(return_messages=True)
        
        # Iterar sobre cada mensaje en el historial
        for message in messages:
            # Si el mensaje es del usuario
            if message["role"] == "user":
                memory.chat_memory.add_user_message(message["content"])
            # Si el mensaje es del asistente
            elif message["role"] == "assistant":
                memory.chat_memory.add_ai_message(message["content"])
                
        return memory

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print("> messages")
        print(messages)
        print("> user_message")
        print(user_message)

        OPENAI_API_KEY = self.valves.OPENAI_API_KEY
        os.environ['OPENAI_API_KEY'] = self.valves.OPENAI_API_KEY
        MODEL = "o1-preview"

        headers = {}
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        headers["Content-Type"] = "application/json"

        payload = {**body, "model": MODEL}

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print("> Payload")
        print(payload)

        try:
            '''# Crear la cadena de conversación
            conversation = ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=True
            )
            
            # Obtener la respuesta
            response = conversation.predict(input=user_message)'''

            # Configurar la memoria con el historial de conversación
            memory: ConversationBufferMemory = self._convert_messages_to_memory(messages)
            

            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser

            # Crear el template del prompt
            prompt = ChatPromptTemplate.from_template(
                "Explica y detalla tus pensamientos paso a paso para llegar a la respuesta del usuario. Pregunta del usuario: {user_message}."
            )

            # Crear la cadena
            chain = (
                memory
                | prompt 
                | self.llm 
                | StrOutputParser()
            )

            # Ejemplo de uso
            response = chain.invoke({
                "user_message": user_message
            })
            
            return response
                
        except Exception as e:
            return f"Error: {e}"
