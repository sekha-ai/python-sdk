"""
Complete integration tests for SekhaClient
Tests all major API endpoints with realistic scenarios
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import httpx

from sekha import (
    SekhaClient,
    ClientConfig,
    NewConversation,
    MessageDto,
    MessageRole,
    ConversationStatus,
    QueryRequest,
    SekhaAPIError,
    SekhaAuthError,
    SekhaConnectionError,
    SekhaNotFoundError,
    SekhaValidationError,
    QueryResponse,
)


# ==================== Fixtures ====================

@pytest.fixture
def config():
    """Test configuration with valid API key"""
    return ClientConfig(
        base_url="http://localhost:8080",
        api_key="sk-sekha-test-12345678901234567890123456789012",
        default_label="Test",
    )


@pytest.fixture
def memory(config):
    """Create client instance (no mocking yet)"""
    return SekhaClient(config)


@pytest.fixture
def mock_client():
    """Create a client with mocked httpx - synchronous fixture"""
    config = ClientConfig(api_key="sk-sekha-test-12345678901234567890123456789012")
    client = SekhaClient(config)

    # Create a mock that tracks calls but allows method assignment
    mock_httpx_client = AsyncMock()

    # Set up default mock response
    default_response = Mock()
    default_response.raise_for_status = Mock()
    
    # Make json dynamic to return what was sent
    def dynamic_json():
        # Try to get the request data
        try:
            call_args = mock_httpx_client.post.call_args
            if call_args and len(call_args.args) > 0:
                # For GET requests, check query params
                if mock_httpx_client.get.called:
                    get_args = mock_httpx_client.get.call_args
                    if "params" in get_args.kwargs and "label" in get_args.kwargs["params"]:
                        return {
                            "content": f"# Test Export for {get_args.kwargs['params']['label']}\n\n## Conversation 1\n\nThis is the content.",
                            "format": "markdown",
                            "conversation_count": 1,
                        }
                
                # For POST requests, get the JSON body
                if "json" in call_args.kwargs:
                    request_data = call_args.kwargs["json"]
                    label = request_data.get("label", "Test")
                else:
                    label = "Test"
            else:
                label = "Test"
        except:
            label = "Test"
            
        return {
            "id": "conv-123",
            "label": label,
            "folder": "/work",
            "status": "active",
            "message_count": 2,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    default_response.json = dynamic_json

    # Set default return values for all common methods
    mock_httpx_client.post = AsyncMock(return_value=default_response)
    mock_httpx_client.get = AsyncMock(return_value=default_response)
    mock_httpx_client.put = AsyncMock(return_value=default_response)
    mock_httpx_client.delete = AsyncMock(return_value=default_response)

    # Replace the actual client with mock
    client.client = mock_httpx_client

    return client  # Return the client directly, not a generator


# ==================== Initialization Tests ====================

class TestMemoryControllerInit:
    """Test client initialization and configuration"""
    
    def test_init_with_config(self, config):
        """Test initialization with full config"""
        client = SekhaClient(config)
        assert client.config.api_key == config.api_key
        assert client.config.base_url == config.base_url
        
    def test_init_validates_api_key_length(self):
        """Test that short API keys raise error"""
        with pytest.raises(ValueError, match="too short"):
            ClientConfig(api_key="sk-sekha-short")
            
    def test_init_validates_base_url_format(self):
        """Test that invalid URLs raise error"""
        with pytest.raises(ValueError, match="Invalid base_url"):
            ClientConfig(api_key="sk-sekha-" + "x" * 32, base_url="not-a-url")
            
    def test_init_with_default_label(self, config):
        """Test config includes default_label"""
        assert config.default_label == "Test"


# ==================== Conversation Creation Tests ====================

class TestCreateConversation:
    """Test conversation creation workflows"""
    
    @pytest.mark.asyncio
    async def test_create_conversation_success(self, mock_client):
        """Test basic conversation creation"""
        conv = NewConversation(
            label="Test Conversation",
            folder="/work",
            messages=[
                MessageDto(role=MessageRole.USER, content="Hello"),
                MessageDto(role=MessageRole.ASSISTANT, content="Hi there!")
            ]
        )
        
        result = await mock_client.create_conversation(conv)
        
        assert result.label == "Test Conversation"
        assert result.folder == "/work"
        assert mock_client.client.post.called
        
    @pytest.mark.asyncio
    async def test_create_conversation_minimal(self, mock_client):
        """Test creating conversation with minimal data"""
        conv = NewConversation(
            label="Minimal Test",  # Provide minimal required data
            messages=[MessageDto(role=MessageRole.USER, content="Test")]
        )
        
        result = await mock_client.create_conversation(conv)
        assert result.id == "conv-123"

    @pytest.mark.asyncio
    async def test_create_conversation_auth_error(self, config):
        """Test 401 error handling"""
        # Create a real client but mock the httpx client directly
        client = SekhaClient(config)
        
        error_response = Mock()
        error_response.status_code = 401
        error_response.text = "Invalid API key"
        
        # Create a proper httpx request mock
        request_mock = Mock()
        request_mock.url = "http://localhost:8080/api/v1/conversations"
        
        # Create the HTTPStatusError
        auth_error = httpx.HTTPStatusError(
            "Unauthorized",
            request=request_mock,
            response=error_response
        )
        
        # Mock the client's httpx client
        client.client = AsyncMock()
        client.client.post = AsyncMock(side_effect=auth_error)
        
        conv = NewConversation(
            label="Test",
            messages=[MessageDto(role=MessageRole.USER, content="Test")]
        )
        
        with pytest.raises(SekhaAuthError):
            await client.create_conversation(conv)

    @pytest.mark.asyncio
    async def test_400_bad_request(self, mock_client):
        """Test 400 error handling"""
        error_response = Mock()
        error_response.status_code = 400
        error_response.text = "Invalid request payload"

        mock_client.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=Mock(),
                response=error_response
            )
        )

        conv = NewConversation(
            label="Test",
            messages=[MessageDto(role=MessageRole.USER, content="Test")]
        )

        # Change from SekhaAPIError to SekhaValidationError
        with pytest.raises(SekhaValidationError, match="Invalid conversation data"):
            await mock_client.create_conversation(conv)

    @pytest.mark.asyncio
    async def test_429_rate_limit(self, mock_client):
        """Test rate limit error handling"""
        error_response = Mock()
        error_response.status_code = 429
        error_response.headers = {"Retry-After": "60"}

        mock_client.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Too Many Requests",
                request=Mock(),
                response=error_response
            )
        )

        conv = NewConversation(
            label="Test",
            messages=[MessageDto(role=MessageRole.USER, content="Test")]
        )

        # FIX: This should raise SekhaAPIError, not SekhaValidationError
        with pytest.raises(SekhaAPIError, match="429"):
            await mock_client.create_conversation(conv)


# ==================== Smart Query Tests ====================

class TestSmartQuery:
    """Test intelligent context assembly"""
    
    @pytest.mark.asyncio
    async def test_smart_query_success(self, mock_client):
        """Test successful smart query"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "results": [
                {
                    "conversation_id": "conv-456",
                    "message_id": "msg-789",
                    "score": 0.92,
                    "content": "Authentication is handled via API keys",
                    "label": "Project:Auth",
                    "folder": "/work",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "total": 1,
            "page": 1,
            "page_size": 10,
        })
        
        mock_client.client.post = AsyncMock(return_value=mock_response)
        
        result = await mock_client.smart_query(
            query="How to handle authentication?",
            limit=5,
            filters={"label": "Project:Auth"}
        )
        
        assert result.total == 1
        assert len(result.results) == 1
        assert result.results[0].score > 0.9
        
    @pytest.mark.asyncio
    async def test_smart_query_empty_results(self, mock_client):
        """Test query with no results"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "results": [],
            "total": 0,
            "page": 1,
            "page_size": 10,
        })
        
        mock_client.client.post = AsyncMock(return_value=mock_response)
        
        result = await mock_client.smart_query(query="non-existent topic")
        
        assert result.total == 0
        assert len(result.results) == 0


# ==================== Memory Management Tests ====================

class TestMemoryManagement:
    """Test pin, archive, label operations"""
    
    @pytest.mark.asyncio
    async def test_pin_conversation(self, mock_client):
        """Test pinning a conversation"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={"success": True})
        
        mock_client.client.put = AsyncMock(return_value=mock_response)
        
        await mock_client.pin("conv-123")
        
        # Verify the call was made
        assert mock_client.client.put.called
        call_args = mock_client.client.put.call_args
        assert "conv-123/status" in call_args[0][0]
        assert call_args[1]["json"]["status"] == "pinned"
        
    @pytest.mark.asyncio
    async def test_archive_conversation(self, mock_client):
        """Test archiving a conversation"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        
        mock_client.client.put = AsyncMock(return_value=mock_response)
        
        await mock_client.archive("conv-456")
        
        call_args = mock_client.client.put.call_args
        assert call_args[1]["json"]["status"] == "archived"
        
    @pytest.mark.asyncio
    async def test_update_label(self, mock_client):
        """Test updating conversation label and folder"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        
        mock_client.client.put = AsyncMock(return_value=mock_response)
        
        await mock_client.update_label(
            conversation_id="conv-123",
            new_label="Updated Label",
            new_folder="/archive"
        )
        
        call_args = mock_client.client.put.call_args
        assert call_args[1]["json"]["label"] == "Updated Label"
        assert call_args[1]["json"]["folder"] == "/archive"


# ==================== Export Tests ====================

class TestExport:
    """Test export functionality"""
    
    @pytest.mark.asyncio
    async def test_export_markdown(self, mock_client):
        """Test markdown export"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "content": "# Test Export\n\n## Conversation 1\n\nThis is the content.",
            "format": "markdown",
            "conversation_count": 1,
        })
        
        mock_client.client.get = AsyncMock(return_value=mock_response)
        
        result = await mock_client.export(
            label="Project:AI",
            format="markdown"
        )
        
        assert result.startswith("# Test Export")
        assert "conversation 1" in result.lower()
        
    @pytest.mark.asyncio
    async def test_export_json(self, mock_client):
        """Test JSON export"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={
            "content": '[{"id": "conv-1", "label": "Test"}]',
            "format": "json",
            "conversation_count": 1,
        })
        
        mock_client.client.get = AsyncMock(return_value=mock_response)
        
        result = await mock_client.export(format="json")
        
        assert '"id"' in result
        assert '"label"' in result


# ==================== Pruning Tests ====================

class TestPruning:
    """Test pruning suggestions"""
    
    @pytest.mark.asyncio
    async def test_get_pruning_suggestions(self, mock_client):
        """Test pruning suggestions retrieval"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=[
            {
                "conversation_id": "old-conv-1",
                "conversation_label": "Old Project",
                "last_accessed": datetime.now().isoformat(),
                "message_count": 150,
                "token_estimate": 4500,
                "importance_score": 2.1,
                "preview": "Old conversation content...",
                "recommendation": "archive",
            },
            {
                "conversation_id": "old-conv-2",
                "conversation_label": "Legacy Code",
                "last_accessed": datetime.now().isoformat(),
                "message_count": 200,
                "token_estimate": 6000,
                "importance_score": 1.8,
                "preview": "Deprecated code discussion...",
                "recommendation": "delete",
            }
        ])
        
        mock_client.client.get = AsyncMock(return_value=mock_response)
        
        suggestions = await mock_client.get_pruning_suggestions(
            threshold_days=90,
            importance_threshold=3.0
        )
        
        assert len(suggestions) == 2
        assert suggestions[0].recommendation == "archive"
        assert suggestions[0].importance_score < 3.0


# ==================== Error Handling Tests ====================

class TestErrorHandling:
    """Test comprehensive error scenarios"""
    
    @pytest.mark.asyncio
    async def test_400_bad_request(self, mock_client):
        """Test 400 error handling"""
        error_response = Mock()
        error_response.status_code = 400
        error_response.text = "Invalid request payload"
        
        # Create proper request mock
        request_mock = Mock()
        request_mock.url = "http://localhost:8080/api/v1/conversations"
        
        mock_client.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=request_mock,
                response=error_response
            )
        )
        
        conv = NewConversation(
            label="Test",
            messages=[MessageDto(role=MessageRole.USER, content="Test")]
        )
        
        # FIX: Change SekhaAPIError to SekhaValidationError
        with pytest.raises(SekhaValidationError, match="Invalid conversation data"):
            await mock_client.create_conversation(conv)
            
    @pytest.mark.asyncio
    async def test_404_not_found(self, mock_client):
        """Test 404 error handling"""
        error_response = Mock()
        error_response.status_code = 404
        
        mock_client.client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=Mock(),
                response=error_response
            )
        )
        
        with pytest.raises(SekhaNotFoundError):
            await mock_client.get_conversation("non-existent")
            
    @pytest.mark.asyncio
    async def test_429_rate_limit(self, mock_client):
        """Test rate limit error handling"""
        error_response = Mock()
        error_response.status_code = 429
        error_response.headers = {"Retry-After": "60"}
        
        request_mock = Mock()
        request_mock.url = "http://localhost:8080/api/v1/conversations"
        
        mock_client.client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Too Many Requests",
                request=request_mock,
                response=error_response
            )
        )
        
        conv = NewConversation(
            label="Test",
            messages=[MessageDto(role=MessageRole.USER, content="Test")]
        )
        
        # This one correctly expects SekhaAPIError for 429
        with pytest.raises(SekhaAPIError, match="429"):
            await mock_client.create_conversation(conv)

class TestErrorBranches:
    """Test specific error handling branches"""
    
    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, mock_client):
        """Test 404 on get_conversation"""
        error_response = Mock()
        error_response.status_code = 404
        
        mock_client.client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=Mock(),
                response=error_response
            )
        )
        
        with pytest.raises(SekhaNotFoundError):
            await mock_client.get_conversation("non-existent")

    @pytest.mark.asyncio
    async def test_smart_query_connection_error(self, mock_client):
        """Test connection error in smart_query"""
        mock_client.client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection failed")
        )
        
        with pytest.raises(SekhaConnectionError):
            await mock_client.smart_query("test query")

    @pytest.mark.asyncio
    async def test_export_invalid_format(self, mock_client):
        """Test export with invalid format parameter"""
        error_response = Mock()
        error_response.status_code = 400
        
        mock_client.client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=Mock(),
                response=error_response
            )
        )
        
        with pytest.raises(SekhaValidationError):
            await mock_client.export(format="invalid")

# ==================== Async Context Manager Tests ====================

class TestAsyncClient:
    """Test async context manager functionality"""
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, config):
        """Test async context manager usage"""
        async with SekhaClient(config) as client:
            assert isinstance(client, SekhaClient)
            assert client.client is not None
            
    @pytest.mark.asyncio
    async def test_async_cleanup(self, config):
        """Test proper cleanup of async resources"""
        client = SekhaClient(config)
        await client.close()
        # Should not raise any exceptions


# ==================== Rate Limiting Tests ====================

class TestRateLimiting:
    """Test rate limiter integration"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquires(self, config):
        """Test rate limiter is called"""
        client = SekhaClient(config)
        
        # Access the rate limiter
        assert client.rate_limiter.max_requests == 1000
        assert client.rate_limiter.window_seconds == 60.0