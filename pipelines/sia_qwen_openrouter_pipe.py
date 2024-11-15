"""
title: OpenRouter Qwen Pipe
version: 0.1.0
license: MIT
"""
import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message

class Pipe:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(default="")
        SITE_URL: str = Field(default="http://localhost")
        SITE_NAME: str = Field(default="Local Development")

    def __init__(self):
        self.type = "manifold"
        self.id = "openrouter"
        self.name = "openrouter/"
        self.valves = self.Valves(
            **{
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", "")
            }
        )

    def get_qwen_models(self):
        return [
            {"id": "qwen/qwen-2.5-coder-32b-instruct", "name": "qwen-2.5-coder"}
        ]

    def pipes(self) -> List[dict]:
        return self.get_qwen_models()

    def process_image(self, image_data):
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["image_url"]["url"]},
            }

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])
        processed_messages = []
        
        # Agregar mensaje del sistema si existe
        if system_message:
            processed_messages.append({
                "role": "system",
                "content": str(system_message)
            })

        # Procesar mensajes
        for message in messages:
            if isinstance(message.get("content"), list):
                # Procesar contenido multimedia
                content_parts = []
                for item in message["content"]:
                    if item["type"] == "text":
                        content_parts.append(item["text"])
                    elif item["type"] == "image_url":
                        # Por ahora, omitimos las imágenes ya que Qwen no las procesa
                        continue
                processed_messages.append({
                    "role": message["role"],
                    "content": " ".join(content_parts)
                })
            else:
                # Contenido simple de texto
                processed_messages.append({
                    "role": message["role"],
                    "content": message.get("content", "")
                })

        payload = {
            "model": body["model"],
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 0.8),
            "top_k": body.get("top_k", 40),
            "top_p": body.get("top_p", 0.9),
            "stream": body.get("stream", False)
        }

        headers = {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        url = "https://openrouter.ai/api/v1/chat/completions"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=(3.05, 60)
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )
                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                                time.sleep(0.01)
                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                print(f"Unexpected data structure: {e}")
                                print(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, headers, payload):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60)
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")
            res = response.json()
            return res["choices"][0]["message"]["content"] if "choices" in res else ""
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"