# Observix

**Observix** is a Python SDK for **observability, tracing, and evaluation of LLM applications**, designed to be embedded directly into your codebase as a **git submodule**.

It provides:

- End-to-end tracing for LLM calls and agent workflows
- Unified evaluation across popular LLM evaluation frameworks
- Lightweight HTTP exporters
- Scalable storage with ClickHouse and Postgres
- First-class integrations with LangChain-based LLMs

---

## Why Observix?

- 🧩 **Submodule-first**: Version observability alongside your application code
- 🔍 **Deep tracing**: Automatic instrumentation for LLM calls and workflows
- 📊 **Evaluation-native**: Built-in support for multiple evaluation frameworks
- ⚡ **Low overhead**: Async, HTTP-based exporters
- 🧠 **LLM-aware**: Native wrappers for LangChain + Groq / OpenAI

---

## Features

### Tracing

- Automatic tracing via decorators
- LLM call metadata (model, tokens, latency)
- Workflow and agent-level traces

### Evaluation

Integrated support for:

- [**DeepEval**](https://github.com/confident-ai/deepeval)
- [**Ragas**](https://github.com/explodinggradients/ragas)
- [**Arize Phoenix**](https://github.com/Arize-ai/phoenix)
- **Observix Evaluation** (agent & tool-based evaluation)

### Storage

- **ClickHouse** → traces, spans, metrics
- **PostgreSQL** → users, projects, API keys

### LLM Integrations

- LangChain-compatible wrappers
- Supported providers:
  - Groq
  - OpenAI
  - Azure OpenAI

---

## Installation (Git Submodule – Recommended)

Observix is intended to be used **as a git submodule**, not as a standalone pip dependency.

```bash
git submodule add https://github.com/your-org/observix.git observix
```

## Quick Start

```python
from observix.llm.langchain import ChatGroq
from observix import observe, init_observability
from observix.llm.openai import OpenAI, AzureOpenAI

# Use the observe decorator
@observe(agent_name)
def my_llm_function(prompt: str) -> str:
    # Your LLM logic here

    # Langchain
    llm = ChatGroq(model="openai/gpt-oss/120b", temperature=0)
    response = llm.invoke("Hi")

    # OpenAI
    llm = OpenAI()
    response = llm.responses.create(
      model="gpt-4o",
      messages=[{
        "role": "assistant", "content": "You're a helpful agent",
        "role": "user", "content": "Hi"
      }]
    )

    # Azure OpenAI
    llm = AzureOpenAI(api_key="", azure_endpoint="", api_version="")
    response = llm.chat.completions.create(
      model=<deployment_name>, 
      messages=[{
        "role": "assistant", "content": "You're a helpful agent",
        "role": "user", "content": "Hi"
      }]
    )

    return response
```

## Configuration

Configure via environment variables:

```bash
export OBSERVIX_URL=http://localhost:8000
export OBSERVIX_API_KEY=your-api-key
```

Or programmatically:

```python
init_observability(url="http://localhost:8000", api_key="your-api-key")
```

## Git Submodule Usage

This repository can be used as a git submodule in your project:

```bash
git submodule add https://github.com/your-org/observix.git path/to/observix
```

Then install as an editable package:

```bash
pip install -e path/to/observix
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/observix.git
cd observix

# Install dependencies
pip install -e .
```

### Testing

```bash
pytest
```

### Linting

```bash
ruff check .
mypy src/
```