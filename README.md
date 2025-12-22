# Sekha Python SDK

Python client library for the Sekha AI Memory System.

## Installation

```bash
pip install sekha-memory

## Quick Start

import asyncio
from sekha import SekhaClient, ClientConfig, NewConversation, MessageDto

async def main():
    # Configure client
    config = ClientConfig(
        api_key="sk-sekha-your-api-key-here",
        base_url="http://localhost:8080"
    )
    
    async with SekhaClient(config) as client:
        # Create a conversation
        conv = NewConversation(
            label="Project:AI",
            folder="/work",
            messages=[
                MessageDto(role="user", content="What is token limit for Claude?"),
                MessageDto(role="assistant", content="200K tokens"),
            ]
        )
        
        result = await client.create_conversation(conv)
        print(f"Created: {result.id}")
        
        # Smart query
        response = await client.smart_query("token limits")
        for msg in response.results:
            print(f"- {msg.content} (from {msg.label})")

# Run
asyncio.run(main())


Features

Core Operations
  Create/Get/Delete conversations
  Smart context assembly - Automatically ranks by importance, recency, and relevance
  Semantic search - Vector-based search across all messages
  Label management - Organize conversations hierarchically

Intelligence (Requires LLM Bridge)
  Importance scoring - LLM rates messages 1-10
  Hierarchical summarization - Daily/weekly/monthly summaries
  Pruning suggestions - Intelligent archive recommendations
  Auto-labeling - LLM suggests and applies labels

Advanced
  Rate limiting - Automatic (1000 req/min default)
  Retry logic - Exponential backoff with jitter
  Connection pooling - Reuses connections for performance
  Type hints - Full IDE support with autocomplete


## Configuration

config = ClientConfig(
    api_key="sk-sekha-...",           # Required
    base_url="http://localhost:8080",  # Default
    timeout=30.0,                      # Request timeout (seconds)
    max_retries=3,                     # Retry attempts
    rate_limit_requests=1000,          # Per minute
)

## Synchronous Usage

from sekha import SyncSekhaClient

with SyncSekhaClient(config) as client:
    # All async methods available synchronously
    result = client.create_conversation(conv)
    response = client.smart_query("token limits")

## Error Handling

from sekha import SekhaError, SekhaNotFoundError, SekhaAPIError

try:
    conv = await client.get_conversation("invalid-id")
except SekhaNotFoundError:
    print("Conversation not found")
except SekhaAPIError as e:
    print(f"API Error {e.status_code}: {e.message}")
except SekhaError as e:
    print(f"Unexpected error: {e}")


### **Pinning and Archiving Conversations**

```python
from sekha import SekhaClient

async with SekhaClient(config) as client:
    # Pin an important conversation
    await client.pin("conversation-uuid-here")
    
    # Archive an old conversation
    await client.archive("conversation-uuid-here")
    
    # Pinned conversations get boosted in search results
    # Archived conversations are excluded from default searches


### **Exporting Conversations**

from sekha import SekhaClient

async with SekhaClient(config) as client:
    # Export all conversations with a specific label as markdown
    markdown = await client.export(label="Project:AI", format="markdown")
    with open("backup.md", "w") as f:
        f.write(markdown)
    
    # Export all conversations as JSON
    json_data = await client.export(format="json")
    with open("backup.json", "w") as f:
        f.write(json_data)

    # Export without label filter (includes all conversations)
    all_markdown = await client.export(format="markdown")

## Development

git clone https://github.com/sekha-ai/sekha-controller
cd sekha-controller/sekha-python-sdk
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type check
mypy sekha/

# Format code
black sekha/
isort sekha/

git clone https://github.com/sekha-ai/sekha-controller
cd sekha-controller/sekha-python-sdk
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type check
mypy sekha/

# Format code
black sekha/
isort sekha/

## License
AGPL-3.0 license






Claude's entry:

📚 DOCUMENTATION
Create sekha-python-sdk/README.md:

text
# Sekha Python SDK

Official Python SDK for Sekha AI Memory Controller

## Installation

pip install sekha-sdk

text

## Quick Start

from sekha import MemoryController, MemoryConfig

config = MemoryConfig(
base_url="http://localhost:8080",
api_key="sk-your-api-key-here",
default_label="Work"
)

memory = MemoryController(config)

Create conversation
conversation = memory.create(
messages=[
{"role": "user", "content": "How do we handle authentication?"},
{"role": "assistant", "content": "We use JWT tokens..."}
],
label="Project:AI",
folder="Work/2025"
)

Assemble context
context = memory.assemble_context(
query="authentication patterns",
labels=["Project:AI", "Security"],
token_budget=8000
)

Search
results = memory.search("JWT implementation", limit=10)

Memory management
memory.pin(conversation.id)
memory.update_label(conversation.id, "Project:AI-Archive")
memory.archive(conversation.id)

Pruning
suggestions = memory.get_pruning_suggestions()
for suggestion in suggestions:
print(f"Consider archiving: {suggestion.id} - {suggestion.reason}")

Export
markdown = memory.export("Project:AI", format="markdown")

text

## Async Support

async with memory.async_client() as client:
conversation = await client.create(messages=[...], label="Test")
context = await client.assemble_context(query="test", token_budget=5000)

text

## Error Handling

from sekha.errors import SekhaAPIError, SekhaAuthError, SekhaConnectionError

try:
conversation = memory.create(messages=[...])
except SekhaAuthError:
print("Invalid API key")
except SekhaConnectionError:
print("Cannot connect to Sekha controller")
except SekhaAPIError as e:
print(f"API error: {e.message}")

text

## API Reference

### MemoryConfig

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| base_url | str | Yes | Controller URL |
| api_key | str | Yes | API key (min 32 chars) |
| default_label | str | No | Default label for conversations |
| timeout | int | No | Request timeout (default: 30s) |
| max_retries | int | No | Max retry attempts (default: 3) |

### MemoryController.create()

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| messages | List[Dict] | Yes | Conversation messages |
| label | str | No | Conversation label |
| folder | str | No | Organization folder |

Returns: Dict with id, label, created_at

### MemoryController.assemble_context()

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | str | Yes | Search query |
| labels | List[str] | No | Filter by labels |
| token_budget | int | No | Max tokens (default: 8000) |

Returns: Dict with context, token_count, conversations_used

### MemoryController.search()

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | str | Yes | Search query |
| label | str | No | Filter by single label |
| limit | int | No | Max results (default: 10) |

Returns: List of conversations with scores

## Testing

pip install -e ".[dev]"
pytest
pytest --cov=sekha --cov-report=html

text

## License

AGPL-3.0
Create sekha-js-sdk/README.md:

text
# Sekha JavaScript/TypeScript SDK

Official TypeScript SDK for Sekha AI Memory Controller

## Installation

npm install @sekha/sdk

or
pnpm add @sekha/sdk

or
yarn add @sekha/sdk

text

## Quick Start

import { MemoryController } from '@sekha/sdk';

const memory = new MemoryController({
baseURL: 'http://localhost:8080',
apiKey: 'sk-your-api-key-here',
defaultLabel: 'Work'
});

// Create conversation
const conversation = await memory.create({
messages: [
{ role: 'user', content: 'How do we handle auth?' },
{ role: 'assistant', content: 'We use JWT tokens...' }
],
label: 'Project:AI',
folder: 'Work/2025'
});

// Assemble context
const context = await memory.assembleContext({
query: 'authentication patterns',
labels: ['Project:AI', 'Security'],
tokenBudget: 8000
});

// Search
const results = await memory.search('JWT implementation', { limit: 10 });

// Memory management
await memory.pin(conversation.id);
await memory.updateLabel(conversation.id, 'Project:AI-Archive');
await memory.archive(conversation.id);

// Pruning
const suggestions = await memory.getPruningSuggestions();

// Export
const markdown = await memory.export('Project:AI', 'markdown');

// Streaming export
const stream = memory.exportStream('Project:AI');
for await (const chunk of stream) {
console.log(chunk);
}

text

## Request Cancellation

const controller = new AbortController();

const promise = memory.create(
{ messages: [...], label: 'Test' },
{ signal: controller.signal }
);

// Cancel request
controller.abort();

text

## Error Handling

import { SekhaAPIError, SekhaAuthError, SekhaConnectionError } from '@sekha/sdk';

try {
const conversation = await memory.create({...});
} catch (error) {
if (error instanceof SekhaAuthError) {
console.error('Invalid API key');
} else if (error instanceof SekhaConnectionError) {
console.error('Connection failed');
} else if (error instanceof SekhaAPIError) {
console.error(API error: ${error.message});
}
}

text

## TypeScript Types

interface MemoryConfig {
baseURL: string;
apiKey: string;
defaultLabel?: string;
timeout?: number;
maxRetries?: number;
}

interface CreateConversationRequest {
messages: Message[];
label?: string;
folder?: string;
}

interface Message {
role: 'user' | 'assistant' | 'system';
content: string;
}

interface ContextRequest {
query: string;
labels?: string[];
tokenBudget?: number;
}

interface Conversation {
id: string;
label: string;
folder?: string;
messages: Message[];
created_at: string;
}

text

## Build

pnpm install
pnpm build # Builds ESM + CJS
pnpm test # Run tests
pnpm lint # Lint code

text

## Tree-shaking

This package is fully tree-shakeable. Only import what you need:

import { MemoryController } from '@sekha/sdk/client';
import { SekhaAPIError } from '@sekha/sdk/errors';

text

## License

AGPL-3.0
Create sekha-vscode/README.md:

text
# Sekha VS Code Extension

Connect your AI conversations to VS Code with persistent memory management

## Features

- Save AI conversations directly from VS Code
- Search your conversation history
- Insert relevant context into new chats
- Browse conversations in tree view
- Auto-save conversations every 5 minutes
- Label and organize conversations

## Installation

1. Install from VS Code Marketplace
2. Open Settings (Cmd/Ctrl + ,)
3. Search for "Sekha"
4. Configure API URL and API key

## Configuration

{
"sekha.apiUrl": "http://localhost:8080",
"sekha.apiKey": "sk-your-api-key",
"sekha.autoSave": true,
"sekha.defaultLabel": "Work"
}

text

## Commands

| Command | Description |
|---------|-------------|
| `Sekha: Save this conversation` | Save current chat to memory |
| `Sekha: Search memory` | Search conversation history |
| `Sekha: Insert context` | Insert relevant context at cursor |
| `Sekha: Refresh` | Refresh tree view |

## Usage

### Saving Conversations

1. Open command palette (Cmd/Ctrl + Shift + P)
2. Run "Sekha: Save this conversation"
3. Select label from quick pick
4. Conversation is saved

### Searching Memory

1. Run "Sekha: Search memory"
2. Enter search query in webview
3. Click result to view details
4. Click "Insert" to add to editor

### Tree View

- Click Sekha icon in activity bar
- Browse conversations by label
- Click conversation to view details
- Right-click for context menu options

### Auto-save

Enable `sekha.autoSave` to automatically save conversations every 5 minutes

## Development

npm install
npm run compile
npm run watch # Watch mode
npm test # Run tests
vsce package # Build VSIX

text

## Publishing

vsce publish

text

## License

AGPL-3.0