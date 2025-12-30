"""Tests for untested client methods"""

import pytest
from unittest.mock import Mock, AsyncMock
from sekha import SekhaClient, ClientConfig, NewConversation, MessageDto, MessageRole

@pytest.fixture
def mock_client():
    config = ClientConfig(api_key="sk-sekha-test-12345678901234567890123456789012")
    client = SekhaClient(config)
    
    # Mock the httpx client
    mock_httpx = AsyncMock()
    default_response = Mock()
    default_response.raise_for_status = Mock()
    default_response.json = Mock(return_value={
        "id": "conv-123",
        "label": "Test",
        "folder": "/",
        "status": "active",
        "message_count": 1,
        "created_at": "2025-12-30T10:00:00Z",
        "updated_at": "2025-12-30T10:00:00Z",
    })
    
    mock_httpx.get = AsyncMock(return_value=default_response)
    mock_httpx.delete = AsyncMock(return_value=default_response)
    mock_httpx.post = AsyncMock(return_value=default_response)
    
    client.client = mock_httpx
    return client

# At the top level, add:
@pytest.fixture
def config():
    """Test configuration"""
    return ClientConfig(
        base_url="http://localhost:8080",
        api_key="sk-sekha-test-12345678901234567890123456789012",
        default_label="Test",
    )

@pytest.mark.asyncio
async def test_async_context_manager_cleanup(config):
    """Test proper resource cleanup in async context manager"""
    client = SekhaClient(config)
    async with client:
        assert client.client is not None
    
    # After exit, should be closed
    await client.close()  # Explicit close for cleanup

# ============== Untested Methods ==============

@pytest.mark.asyncio
async def test_get_conversation(mock_client):
    result = await mock_client.get_conversation("conv-123")
    assert result.id == "conv-123"
    assert mock_client.client.get.called

@pytest.mark.asyncio
async def test_list_conversations(mock_client):
    result = await mock_client.list_conversations(label="Work", page=1, page_size=10)
    assert isinstance(result, list)
    assert mock_client.client.get.called

@pytest.mark.asyncio
async def test_delete_conversation(mock_client):
    await mock_client.delete_conversation("conv-123")
    assert mock_client.client.delete.called

@pytest.mark.asyncio
async def test_score_message_importance(mock_client):
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={
        "score": 8.5,
        "reasoning": "Critical security information",
        "model": "gpt-4"
    })
    mock_client.client.post = AsyncMock(return_value=mock_response)
    
    result = await mock_client.score_message_importance("msg-456")
    assert result.score == 8.5
    assert result.model == "gpt-4"

@pytest.mark.asyncio
async def test_generate_summary(mock_client):
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={
        "summary": "Conversation summary",
        "tokens_used": 150,  # Changed from token_count
        "level": "daily",     # Must be enum value
        "model": "gpt-4"      # Required field
    })
    mock_client.client.post = AsyncMock(return_value=mock_response)
    
    result = await mock_client.generate_summary("conv-123")
    assert result.summary == "Conversation summary"
    assert result.tokens_used == 150

@pytest.mark.asyncio
async def test_suggest_labels(mock_client):
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value=[
        {"label": "Project:AI", "confidence": 0.92, "is_existing": True, "reason": "Matches context"}
    ])
    mock_client.client.post = AsyncMock(return_value=mock_response)
    
    result = await mock_client.suggest_labels("conv-123")
    assert len(result) == 1
    assert result[0].label == "Project:AI"
    assert result[0].confidence > 0.9

@pytest.mark.asyncio
async def test_auto_label_threshold_not_met(mock_client):
    # Setup: low confidence suggestion
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value=[
        {"label": "Uncertain", "confidence": 0.5, "is_existing": False, "reason": "Uncertain"}
    ])
    mock_client.client.post = AsyncMock(return_value=mock_response)
    mock_client.client.put = AsyncMock(return_value=Mock())
    
    result = await mock_client.auto_label("conv-123", threshold=0.8)
    assert result is None  # No label applied

@pytest.mark.asyncio
async def test_auto_label_threshold_met(mock_client):
    # Setup: high confidence suggestion
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value=[
        {"label": "High Confidence", "confidence": 0.95, "is_existing": True, "reason": "Clear match"}
    ])
    mock_client.client.post = AsyncMock(return_value=mock_response)
    mock_client.client.put = AsyncMock(return_value=Mock())
    
    result = await mock_client.auto_label("conv-123", threshold=0.8)
    assert result == "High Confidence"

@pytest.mark.asyncio
async def test_get_mcp_tools(mock_client):
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value=[
        {"name": "search_memory", "description": "Search conversation memory"}
    ])
    mock_client.client.get = AsyncMock(return_value=mock_response)
    
    result = await mock_client.get_mcp_tools()
    assert len(result) == 1
    assert result[0]["name"] == "search_memory"

# ============== SyncSekhaClient Tests ==============

def test_sync_client_wrapper(config):
    """Test SyncSekhaClient wraps async methods"""
    from sekha.client import SyncSekhaClient
    
    sync_client = SyncSekhaClient(config)
    
    # Test that it has async method names
    assert hasattr(sync_client, 'create_conversation')
    assert hasattr(sync_client, 'smart_query')
    
    sync_client._async_client.close()  # Cleanup

@pytest.mark.asyncio
async def test_async_context_manager_cleanup(config):
    """Test proper resource cleanup"""
    async with SekhaClient(config) as client:
        assert client.client is not None
    
    # After exit, client should be closed
    # (This would raise an error if used, which is expected)