# Sekha Python SDK

> **Official Python Client for Sekha Memory System**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-coming--soon-orange.svg)](https://pypi.org)

---

## What is Sekha Python SDK?

Official Python client library for interacting with Sekha Controller.

**Features:**

- ✅ Full REST API coverage
- ✅ Type hints & autocompletion
- ✅ Async/await support
- ✅ Retry logic & error handling
- ✅ Pydantic models
- ✅ 100% test coverage

---

## 📚 Documentation

**Complete SDK docs: [docs.sekha.dev/sdks/python-sdk](https://docs.sekha.dev/sdks/python-sdk/)**

- [Python SDK Guide](https://docs.sekha.dev/sdks/python-sdk/)
- [API Reference](https://docs.sekha.dev/api-reference/rest-api/)
- [Code Examples](https://docs.sekha.dev/sdks/examples/)
- [Getting Started](https://docs.sekha.dev/getting-started/quickstart/)

---

## 🚀 Quick Start

### Installation

```bash
# From PyPI (coming soon)
pip install sekha-sdk

# From source (current)
git clone https://github.com/sekha-ai/sekha-python-sdk.git
cd sekha-python-sdk
pip install -e .
```

### Basic Usage

```python
from sekha import SekhaClient

# Initialize client
client = SekhaClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

# Store a conversation
conversation = client.conversations.create(
    label="My First Conversation",
    messages=[
        {"role": "user", "content": "Hello Sekha!"},
        {"role": "assistant", "content": "Hello! I'll remember this."}
    ]
)

# Search semantically
results = client.query(
    query="What did we discuss?",
    limit=5
)

# Get context for next LLM call
context = client.context.assemble(
    query="Continue our conversation",
    context_budget=8000
)
```

### Async Usage

```python
from sekha import AsyncSekhaClient

async with AsyncSekhaClient(base_url="http://localhost:8080", api_key="key") as client:
    conversation = await client.conversations.create(
        label="Async Conversation",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    results = await client.query("search query")
```

**[Full examples](https://docs.sekha.dev/sdks/python-sdk/)**

---

## 📋 API Coverage

- ✅ Conversations (CRUD)
- ✅ Semantic query
- ✅ Full-text search
- ✅ Context assembly
- ✅ Summarization
- ✅ Labels & folders
- ✅ Pruning
- ✅ Import/export
- ✅ Stats & health

**[Complete API Reference](https://docs.sekha.dev/api-reference/rest-api/)**

---

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy sekha/

# Linting
ruff check .
black --check .

# Build docs
mkdocs serve
```

---

## 🔗 Links

- **Main Repo:** [sekha-controller](https://github.com/sekha-ai/sekha-controller)
- **Docs:** [docs.sekha.dev](https://docs.sekha.dev)
- **Website:** [sekha.dev](https://sekha.dev)
- **Discord:** [discord.gg/sekha](https://discord.gg/sekha)

---

## 📄 License

AGPL-3.0 - **[License Details](https://docs.sekha.dev/about/license/)**
