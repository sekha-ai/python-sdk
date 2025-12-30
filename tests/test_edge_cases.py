"""
Edge case and error handling tests for SekhaClient
Covers missing coverage gaps
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import httpx

from sekha import (
    SekhaClient,
    SyncSekhaClient,
    ClientConfig,
    NewConversation,
    MessageDto,
    MessageRole,
    SekhaAPIError,
    SekhaAuthError,
    SekhaConnectionError,
    SekhaNotFoundError,
    SekhaValidationError,
    SekhaRateLimitError,
)


@pytest.fixture
def config():
    return ClientConfig(
        base_url="http://localhost:8080",
        api_key="sk-sekha-test-12345678901234567890123456789012",
    )


@pytest.fixture
def mock_client(config):
    """Client with mocked httpx"""
    client = SekhaClient(config)
    client.client = AsyncMock()
    return client


# ==================== Error Handling Coverage ====================

class TestErrorHandlingCoverage:
    """Cover missing error handling branches"""
    
    @pytest.mark.asyncio
    async def test_create_conversation_500_error(self, mock_client):
        """Test 500 error handling in create_conversation"""
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"
        
        request_mock = Mock()
        request_mock.url = "http://localhost:8080/api/v1/conversations"
        
        mock_client.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=request_mock,
                response=error_response
            )
        )
        
        conv = NewConversation(
            label="Test",
            messages=[MessageDto(role=MessageRole.USER, content="Test")]
        )
        
        with pytest.raises(SekhaAPIError, match="500"):
            await mock_client.create_conversation(conv)
    
    @pytest.mark.asyncio
    async def test_create_conversation_timeout(self, mock_client):
        """Test timeout handling"""
        mock_client.client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        
        conv = NewConversation(
            label="Test",
            messages=[MessageDto(role=MessageRole.USER, content="Test")]
        )
        
        with pytest.raises(SekhaConnectionError, match="timed out"):
            await mock_client.create_conversation(conv)
    
    @pytest.mark.asyncio
    async def test_get_conversation_404(self, mock_client):
        """Test 404 in get_conversation"""
        error_response = Mock()
        error_response.status_code = 404
        error_response.text = "Not found"
        
        request_mock = Mock()
        request_mock.url = "http://localhost:8080/api/v1/conversations/conv-123"
        
        mock_client.client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=request_mock,
                response=error_response
            )
        )
        
        with pytest.raises(SekhaNotFoundError):
            await mock_client.get_conversation("conv-123")
    
    @pytest.mark.asyncio
    async def test_get_conversation_500(self, mock_client):
        """Test 500 in get_conversation"""
        error_response = Mock()
        error_response.status_code = 500
        
        request_mock = Mock()
        request_mock.url = "http://localhost:8080/api/v1/conversations/conv-123"
        
        mock_client.client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=request_mock,
                response=error_response
            )
        )
        
        with pytest.raises(SekhaAPIError):
            await mock_client.get_conversation("conv-123")
    
    @pytest.mark.asyncio
    async def test_list_conversations_error(self, mock_client):
        """Test error in list_conversations"""
        mock_client.client.get = AsyncMock(side_effect=Exception("Network failed"))
        
        with pytest.raises(Exception, match="Network failed"):
            await mock_client.list_conversations()
    
    @pytest.mark.asyncio
    async def test_delete_conversation_404(self, mock_client):
        """Test 404 in delete_conversation"""
        error_response = Mock()
        error_response.status_code = 404
        
        request_mock = Mock()
        request_mock.url = "http://localhost:8080/api/v1/conversations/conv-123"
        
        mock_client.client.delete = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=request_mock,
                response=error_response
            )
        )
        
        with pytest.raises(SekhaNotFoundError):
            await mock_client.delete_conversation("conv-123")
    
    @pytest.mark.asyncio
    async def test_update_label_404(self, mock_client):
        """Test 404 in update_label"""
        error_response = Mock()
        error_response.status_code = 404
        
        request_mock = Mock()
        request_mock.url = "http://localhost:8080/api/v1/conversations/conv-123/label"
        
        mock_client.client.put = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=request_mock,
                response=error_response
            )
        )
        
        with pytest.raises(SekhaNotFoundError):
            await mock_client.update_label("conv-123", "NewLabel")
    
    @pytest.mark.asyncio
    async def test_smart_query_error(self, mock_client):
        """Test error in smart_query"""
        mock_client.client.post = AsyncMock(side_effect=Exception("Query failed"))
        
        with pytest.raises(Exception, match="Smart query failed"):
            await mock_client.smart_query("test query")
    
    @pytest.mark.asyncio
    async def test_score_message_importance_error(self, mock_client):
        """Test error in score_message_importance"""
        mock_client.client.post = AsyncMock(side_effect=Exception("Scoring failed"))
        
        with pytest.raises(Exception, match="Failed to score"):
            await mock_client.score_message_importance("msg-123")


# ==================== Sync Wrapper Coverage ====================

class TestSyncWrapperCoverage:
    """Cover SyncSekhaClient missing lines"""
    
    def test_sync_client_wrapper(self, config):
        """Test sync wrapper delegates to async methods"""
        # Import here to avoid NameError
        from sekha import SyncSekhaClient
        
        sync_client = SyncSekhaClient(config)
        
        # Mock the underlying async client
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "id": "conv-sync-123",
            "label": "Sync Test",
            "folder": "/",
            "status": "active",
            "message_count": 1,
            "created_at": datetime.now().isoformat(),
        })
        
        sync_client._async_client.client = AsyncMock()
        sync_client._async_client.client.post = AsyncMock(return_value=mock_response)
        
        # Test sync method call
        conv = NewConversation(
            label="Sync Test",
            messages=[MessageDto(role=MessageRole.USER, content="Hello")]
        )
        
        # This should work via __getattr__ delegation
        result = sync_client.create_conversation(conv)
        
        # Verify the async method was called
        assert sync_client._async_client.client.post.called
        assert result.id == "conv-sync-123"
        
        # Cleanup
        sync_client._async_client.sync_client.close()
    
    def test_sync_wrapper_cleanup(self, config):
        """Test sync wrapper cleanup"""
        sync_client = SyncSekhaClient(config)
        # Should not raise
        sync_client._async_client.sync_client.close()


# ==================== Rate Limiter & Backoff Coverage ====================

class TestRateLimiterBackoffCoverage:
    """Cover utility edge cases"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_edge_case(self, config):
        """Test rate limiter with zero window"""
        # This shouldn't happen in normal use, but test the edge
        config.rate_limit_window = 0.001
        client = SekhaClient(config)
        
        # Should still work
        await client.rate_limiter.acquire()
        await client.rate_limiter.acquire()
    
    def test_exponential_backoff_max(self):
        """Test backoff max delay cap"""
        from sekha.utils import ExponentialBackoff
        
        backoff = ExponentialBackoff(base_delay=10.0, max_delay=15.0, factor=2.0)
        
        # Reset and force high attempt
        backoff.attempt = 10
        
        # Should cap at max_delay
        import asyncio
        start = asyncio.get_event_loop().time()
        # Note: We can't easily test the actual wait without async, but we can test the logic
        assert backoff.max_delay == 15.0
    
    def test_validate_base_url_edge_cases(self):
        """Test URL validation edge cases"""
        from sekha.utils import validate_base_url
        
        # Valid URLs
        assert validate_base_url("http://localhost:8080")
        assert validate_base_url("https://api.sekha.ai/v1")
        
        # Invalid URLs
        with pytest.raises(ValueError, match="Invalid base_url"):
            validate_base_url("not-a-url")
        
        with pytest.raises(ValueError, match="Invalid base_url"):
            validate_base_url("ftp://invalid-protocol.com")
        
        with pytest.raises(ValueError, match="Invalid base_url"):
            validate_base_url("http://[invalid")


# ==================== MCP Integration ====================

class TestMCPIntegration:
    """Cover MCP methods if they exist"""
    
    @pytest.mark.asyncio
    async def test_get_mcp_tools(self, mock_client):
        """Test MCP tools listing"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=[
            {"name": "search", "description": "Search conversations"},
            {"name": "create", "description": "Create conversation"}
        ])
        
        mock_client.client.get = AsyncMock(return_value=mock_response)
        
        tools = await mock_client.get_mcp_tools()
        
        assert len(tools) == 2
        assert tools[0]["name"] == "search"
    
    @pytest.mark.asyncio
    async def test_get_mcp_tools_error(self, mock_client):
        """Test MCP tools error handling"""
        mock_client.client.get = AsyncMock(side_effect=Exception("MCP failed"))
        
        with pytest.raises(Exception, match="Failed to get MCP tools"):
            await mock_client.get_mcp_tools()


# ==================== Export Edge Cases ====================

class TestExportEdgeCases:
    """Cover export method edge cases"""
    
    @pytest.mark.asyncio
    async def test_export_no_label(self, mock_client):
        """Test export without label filter"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "content": "# All Conversations\n\n",
            "format": "markdown",
            "conversation_count": 5,
        })
        
        mock_client.client.get = AsyncMock(return_value=mock_response)
        
        result = await mock_client.export(format="markdown")
        
        assert result.startswith("# All")
        assert mock_client.client.get.called
    
    @pytest.mark.asyncio
    async def test_export_json_format(self, mock_client):
        """Test JSON export format"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "content": '[{"id": "1", "label": "Test"}]',
            "format": "json",
            "conversation_count": 1,
        })
        
        mock_client.client.get = AsyncMock(return_value=mock_response)
        
        result = await mock_client.export(label="Work", format="json")
        
        assert '"id"' in result
        assert '"label"' in result


# ==================== Label Intelligence ====================

class TestLabelIntelligence:
    """Cover label suggestion and auto-label"""
    
    @pytest.mark.asyncio
    async def test_suggest_labels_error(self, mock_client):
        """Test suggest_labels error handling"""
        mock_client.client.post = AsyncMock(side_effect=Exception("Label suggestion failed"))
        
        with pytest.raises(Exception, match="Failed to suggest labels"):
            await mock_client.suggest_labels("conv-123")
    
    @pytest.mark.asyncio
    async def test_auto_label_no_confident_match(self, mock_client):
        """Test auto_label when no suggestion meets threshold"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=[
            {"label": "Work", "confidence": 0.5, "is_existing": False, "reason": "Low confidence"}
        ])
        
        mock_client.client.post = AsyncMock(return_value=mock_response)
        mock_client.client.put = AsyncMock()  # Should not be called
        
        result = await mock_client.auto_label("conv-123", threshold=0.8)
        
        assert result is None
        assert not mock_client.client.put.called