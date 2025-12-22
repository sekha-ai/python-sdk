import pytest
from sekha.models import Conversation, Message, ContextRequest, PruningSuggestion
from pydantic import ValidationError


class TestMessage:
    def test_valid_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            Message(role="invalid", content="Test")
    
    def test_message_serialization(self):
        msg = Message(role="assistant", content="Response")
        data = msg.dict()
        assert data == {"role": "assistant", "content": "Response"}


class TestConversation:
    def test_valid_conversation(self):
        conv = Conversation(
            id="conv_123",
            label="Test",
            messages=[Message(role="user", content="Hi")],
            created_at="2025-12-21T19:00:00Z"
        )
        assert conv.id == "conv_123"
        assert len(conv.messages) == 1
    
    def test_conversation_with_folder(self):
        conv = Conversation(
            id="conv_123",
            label="Work",
            folder="Projects/AI",
            messages=[],
            created_at="2025-12-21T19:00:00Z"
        )
        assert conv.folder == "Projects/AI"


class TestContextRequest:
    def test_valid_context_request(self):
        req = ContextRequest(query="test", token_budget=8000)
        assert req.token_budget == 8000
    
    def test_negative_token_budget(self):
        with pytest.raises(ValidationError):
            ContextRequest(query="test", token_budget=-100)
    
    def test_context_request_with_labels(self):
        req = ContextRequest(
            query="auth patterns",
            labels=["Project:AI", "Security"],
            token_budget=5000
        )
        assert len(req.labels) == 2


class TestPruningSuggestion:
    def test_pruning_suggestion(self):
        sug = PruningSuggestion(
            id="conv_123",
            reason="Low importance score",
            score=2
        )
        assert sug.score == 2
        assert "importance" in sug.reason.lower()
