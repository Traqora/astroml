"""OpenAI Provider implementation."""
import json
from typing import Any, Dict, Iterator, List
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)

    def _generate_raw(self, prompt: str, tools: list[dict] | None = None, **kwargs: Any) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key)

        messages: list[dict[str, Any]]
        try:
            messages = json.loads(prompt) if prompt.startswith("[") or prompt.startswith("{") else [{"role": "user", "content": prompt}]
        except json.JSONDecodeError:
            messages = [{"role": "user", "content": prompt}]

        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = client.chat.completions.create(**params)

        if response.usage is not None:
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        else:
            p_tokens = self.count_tokens(prompt)
            c_tokens = 100
            self.last_usage = {
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "total_tokens": p_tokens + c_tokens,
            }

        message = response.choices[0].message
        if message.tool_calls:
            tool_calls_list = []
            for tc in message.tool_calls:
                tool_calls_list.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                    },
                })
            return json.dumps({
                "content": message.content or "",
                "tool_calls": tool_calls_list,
            })

        return message.content or ""

    def get_token_usage(self) -> Dict[str, int]:
        return self.last_usage

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(self, text: str, **kwargs: Any) -> List[float]:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            input=[text],
            model=kwargs.pop("model", "text-embedding-3-small"),
            **kwargs
        )
        return response.data[0].embedding

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception:
            # Approximate (~4 chars/token)
            return max(1, len(text) // 4)
