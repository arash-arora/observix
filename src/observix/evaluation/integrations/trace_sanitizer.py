import os
from dotenv import load_dotenv
from fastapi import HTTPException

from .prompts import TRACE_SANITIZATION_PROMPT

load_dotenv()

class TraceSanitizer:
    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider
        self.model_name = model
        self.metric_name = "TraceSanitizer"
        self.temperature = kwargs.get("temperature", 0.0)
        self.llm_client = self._get_llm()

    def _get_llm(self):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(400, "OPENAI_API_KEY is required")

            from observix.llm.openai import OpenAI
            return OpenAI(name=self.metric_name, api_key=api_key)

        elif self.provider == "azure":
            api_base = os.getenv("AZURE_API_BASE")
            api_version = os.getenv("AZURE_API_VERSION")
            api_key = os.getenv("AZURE_OPENAI_KEY")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

            if not all([api_base, api_version, api_key, deployment]):
                raise HTTPException(
                    400,
                    "Azure requires AZURE_API_BASE, AZURE_API_VERSION, "
                    "AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME",
                )

            from observix.llm.openai import AzureOpenAI
            return AzureOpenAI(
                name=self.metric_name,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base
            )

        elif self.provider == "langchain":
            api_key = os.getenv("GROQ_API_KEY")
            model = self.model_name or "openai/gpt-oss-120b"

            if not api_key:
                raise HTTPException(400, "GROQ_API_KEY is required")

            from observix.llm.langchain import ChatGroq
            return ChatGroq(
                model=model,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=2500,
            )

        raise HTTPException(400, f"Unsupported provider: {self.provider}")

    async def _generate_response(self, prompt: str) -> str:
        if self.provider in {"openai", "azure"}:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat.completions.create(
                model=self.model_name or (self.llm_client.deployment_name if hasattr(self.llm_client, "deployment_name") else "gpt-4"),
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        elif self.provider == "langchain":
            from langchain_core.messages import HumanMessage
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            return response.content

        raise RuntimeError("LLM provider interface mismatch")

    async def sanitize(self, trace_data: str, agents: list, tools: list):
        import json
        
        prompt = TRACE_SANITIZATION_PROMPT.format(
            agents=str(agents),
            tools=str(tools),
            trace=trace_data
        )

        try:
            response_text = await self._generate_response(prompt)
            
            # Parse JSON response
            try:
                data = json.loads(response_text)
                return data.get("sanitized_trace", [])
            except json.JSONDecodeError:
                # Fallback: try to find JSON block if simple parse fails
                import re
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    return data.get("sanitized_trace", [])
                raise ValueError("Could not parse JSON from sanitization response")
                
        except Exception as e:
            print(f"Sanitization failed: {e}")
            return []