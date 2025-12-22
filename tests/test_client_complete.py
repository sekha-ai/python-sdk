import pytest
import httpx
from unittest.mock import Mock, AsyncMock, patch
from sekha import MemoryController, MemoryConfig
from sekha.errors import SekhaAPIError, SekhaConnectionError, SekhaAuthError


@pytest.fixture
def config():
    return MemoryConfig(
        base_url="http://localhost:8080",
        api_key="sk-test-12345678901234567890123456789012",
        default_label="Test"
    )


@pytest.fixture
def memory(config):
    return MemoryController(config)


@pytest.fixture
def mock_response():
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"id": "conv_123", "label": "Test"}
    return response


class TestMemoryControllerInit:
    def test_init_with_config(self, config):
        memory = MemoryController(config)
        assert memory.config.base_url == "http://localhost:8080"
        assert memory.config.api_key.startswith("sk-test")
    
    def test_init_validates_api_key_length(self):
        with pytest.raises(ValueError, match="API key must be at least 32 characters"):
            MemoryConfig(base_url="http://localhost:8080", api_key="short")
    
    def test_init_validates_base_url_format(self):
        with pytest.raises(ValueError, match="Invalid base_url"):
            MemoryConfig(base_url="not-a-url", api_key="sk-" + "x" * 32)


class TestCreateConversation:
    @patch('httpx.Client.post')
    def test_create_conversation_success(self, mock_post, memory):
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {
            "id": "conv_123",
            "label": "Test",
            "created_at": "2025-12-21T19:00:00Z"
        }
        
        result = memory.create(
            messages=[{"role": "user", "content": "Hello"}],
            label="Test"
        )
        
        assert result["id"] == "conv_123"
        assert result["label"] == "Test"
        mock_post.assert_called_once()
    
    @patch('httpx.Client.post')
    def test_create_conversation_with_folder(self, mock_post, memory):
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": "conv_123"}
        
        memory.create(
            messages=[{"role": "user", "content": "Test"}],
            label="Work",
            folder="Projects/2025"
        )
        
        call_args = mock_post.call_args
        assert call_args[1]['json']['folder'] == "Projects/2025"
    
    @patch('httpx.Client.post')
    def test_create_conversation_auth_error(self, mock_post, memory):
        mock_post.return_value.status_code = 401
        
        with pytest.raises(SekhaAuthError):
            memory.create(messages=[{"role": "user", "content": "Test"}])
    
    @patch('httpx.Client.post')
    def test_create_conversation_connection_error(self, mock_post, memory):
        mock_post.side_effect = httpx.ConnectError("Connection failed")
        
        with pytest.raises(SekhaConnectionError):
            memory.create(messages=[{"role": "user", "content": "Test"}])


class TestAssembleContext:
    @patch('httpx.Client.post')
    def test_assemble_context_success(self, mock_post, memory):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "context": "Assembled context",
            "token_count": 500,
            "conversations_used": ["conv_1", "conv_2"]
        }
        
        result = memory.assemble_context(
            query="How do we handle authentication?",
            token_budget=8000
        )
        
        assert result["context"] == "Assembled context"
        assert result["token_count"] == 500
        assert len(result["conversations_used"]) == 2
    
    @patch('httpx.Client.post')
    def test_assemble_context_with_labels(self, mock_post, memory):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"context": "test"}
        
        memory.assemble_context(
            query="test",
            labels=["Project:AI", "Work"],
            token_budget=5000
        )
        
        call_args = mock_post.call_args[1]['json']
        assert call_args['labels'] == ["Project:AI", "Work"]
        assert call_args['token_budget'] == 5000


class TestMemoryManagement:
    @patch('httpx.Client.post')
    def test_pin_conversation(self, mock_post, memory):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"success": True}
        
        result = memory.pin("conv_123")
        assert result is True
    
    @patch('httpx.Client.post')
    def test_archive_conversation(self, mock_post, memory):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"success": True}
        
        result = memory.archive("conv_123")
        assert result is True
    
    @patch('httpx.Client.patch')
    def test_update_label(self, mock_patch, memory):
        mock_patch.return_value.status_code = 200
        mock_patch.return_value.json.return_value = {"success": True}
        
        result = memory.update_label("conv_123", "NewLabel")
        assert result is True
        
        call_args = mock_patch.call_args
        assert "conv_123" in call_args[0][0]
        assert call_args[1]['json']['label'] == "NewLabel"


class TestSearch:
    @patch('httpx.Client.get')
    def test_search_success(self, mock_get, memory):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "results": [
                {"id": "conv_1", "label": "Test", "score": 0.95},
                {"id": "conv_2", "label": "Work", "score": 0.88}
            ]
        }
        
        results = memory.search("authentication", limit=10)
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]
    
    @patch('httpx.Client.get')
    def test_search_with_label_filter(self, mock_get, memory):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"results": []}
        
        memory.search("test", label="Work", limit=5)
        
        call_args = mock_get.call_args
        params = call_args[1]['params']
        assert params['label'] == "Work"
        assert params['limit'] == 5


class TestPruning:
    @patch('httpx.Client.get')
    def test_get_pruning_suggestions(self, mock_get, memory):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "suggestions": [
                {"id": "conv_1", "reason": "Low importance", "score": 2},
                {"id": "conv_2", "reason": "Redundant", "score": 3}
            ]
        }
        
        suggestions = memory.get_pruning_suggestions()
        assert len(suggestions) == 2
        assert suggestions[0]["score"] == 2


class TestExport:
    @patch('httpx.Client.get')
    def test_export_markdown(self, mock_get, memory):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "content": "# Exported Content\n\nTest"
        }
        
        result = memory.export("Project:AI", format="markdown")
        assert result.startswith("# Exported")
    
    @patch('httpx.Client.get')
    def test_export_json(self, mock_get, memory):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "conversations": [{"id": "conv_1"}]
        }
        
        result = memory.export("Work", format="json")
        assert "conversations" in result


class TestRetryLogic:
    @patch('httpx.Client.post')
    def test_retry_on_500_error(self, mock_post, memory):
        # First two calls fail, third succeeds
        mock_post.side_effect = [
            Mock(status_code=500),
            Mock(status_code=500),
            Mock(status_code=200, json=lambda: {"id": "conv_123"})
        ]
        
        result = memory.create(messages=[{"role": "user", "content": "Test"}])
        assert result["id"] == "conv_123"
        assert mock_post.call_count == 3
    
    @patch('httpx.Client.post')
    def test_max_retries_exceeded(self, mock_post, memory):
        mock_post.return_value.status_code = 500
        
        with pytest.raises(SekhaAPIError):
            memory.create(messages=[{"role": "user", "content": "Test"}])
        
        assert mock_post.call_count == 3  # Default max retries


@pytest.mark.asyncio
class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_async_context_manager(self, config):
        async with MemoryController(config).async_client() as client:
            assert client is not None
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_async_create(self, mock_post, config):
        mock_post.return_value.status_code = 201
        mock_post.return_value.json = AsyncMock(return_value={"id": "conv_123"})
        
        async with MemoryController(config).async_client() as client:
            result = await client.create(messages=[{"role": "user", "content": "Test"}])
            assert result["id"] == "conv_123"


class TestErrorHandling:
    @patch('httpx.Client.post')
    def test_400_bad_request(self, mock_post, memory):
        mock_post.return_value.status_code = 400
        mock_post.return_value.json.return_value = {"error": "Invalid payload"}
        
        with pytest.raises(SekhaAPIError, match="Invalid payload"):
            memory.create(messages=[])
    
    @patch('httpx.Client.post')
    def test_404_not_found(self, mock_post, memory):
        mock_post.return_value.status_code = 404
        
        with pytest.raises(SekhaAPIError, match="404"):
            memory.create(messages=[{"role": "user", "content": "Test"}])
    
    @patch('httpx.Client.post')
    def test_429_rate_limit(self, mock_post, memory):
        mock_post.return_value.status_code = 429
        mock_post.return_value.headers = {"Retry-After": "60"}
        
        with pytest.raises(SekhaAPIError, match="Rate limit"):
            memory.create(messages=[{"role": "user", "content": "Test"}])


class TestConnectionPooling:
    def test_client_reuse(self, memory):
        # Should reuse the same client instance
        client1 = memory._client
        client2 = memory._client
        assert client1 is client2
    
    def test_client_cleanup(self, memory):
        memory.close()
        # Should be able to create new client after close
        memory.create(messages=[{"role": "user", "content": "Test"}])
