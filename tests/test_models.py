"""Tests for Pydantic models"""

import pytest
from datetime import datetime
from sekha.models import (
    MessageRole,
    ConversationStatus,
    SummaryLevel,
    MessageDto,
    NewConversation,
    ConversationResponse,
    QueryRequest,
    QueryResult,
    QueryResponse,
    LabelSuggestion,
    PruningSuggestion,
    ImportanceScore,
    SummaryResponse,
)
from pydantic import ValidationError


class TestMessageDto:
    def test_valid_message(self):
        msg = MessageDto(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_message_with_metadata(self):
        msg = MessageDto(
            role=MessageRole.ASSISTANT,
            content="Response",
            metadata={"confidence": 0.9}
        )
        assert msg.metadata["confidence"] == 0.9

    def test_invalid_role_enum(self):
        with pytest.raises(ValueError):
            MessageDto(role="invalid_role", content="Test")


class TestConversationResponse:
    def test_valid_response(self):
        conv = ConversationResponse(
            id="test-uuid-123",
            label="Project:AI",
            folder="/work",
            status=ConversationStatus.ACTIVE,
            message_count=5,
            created_at=datetime.now()
        )
        assert conv.id == "test-uuid-123"
        assert conv.message_count == 5

    def test_response_serialization(self):
        from datetime import timezone
        dt = datetime(2025, 12, 30, 10, 30, 0, tzinfo=timezone.utc)
        conv = ConversationResponse(
            id="conv-123",
            label="Test",
            folder="/",
            status=ConversationStatus.PINNED,
            message_count=1,
            created_at=dt
        )
        data = conv.model_dump()
        assert data["id"] == "conv-123"
        assert data["status"] == "pinned"


class TestQueryRequest:
    def test_basic_query(self):
        req = QueryRequest(query="token limits")
        assert req.query == "token limits"
        assert req.limit == 10  # default

    def test_query_with_limit(self):
        req = QueryRequest(query="auth patterns", limit=50)
        assert req.limit == 50

    def test_query_with_filters(self):
        req = QueryRequest(
            query="test",
            filters={"label": "Project:AI", "folder": "/work"}
        )
        assert req.filters["label"] == "Project:AI"


class TestQueryResult:
    def test_result_creation(self):
        dt = datetime.now()
        result = QueryResult(
            conversation_id="conv-456",
            message_id="msg-789",
            score=0.85,
            content="Important message",
            label="Project:AI",
            folder="/work",
            timestamp=dt
        )
        assert result.score == 0.85
        assert result.conversation_id == "conv-456"


class TestImportanceScore:
    def test_valid_score(self):
        score = ImportanceScore(score=8.5, reasoning="Critical information", model="gpt-4")
        assert score.score == 8.5
        assert 1.0 <= score.score <= 10.0

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            ImportanceScore(score=11.0, reasoning="Too high", model="test")


class TestLabelSuggestion:
    def test_suggestion_with_confidence(self):
        sugg = LabelSuggestion(
            label="Project:AI",
            confidence=0.92,
            is_existing=True,
            reason="Matches context"
        )
        assert sugg.confidence > 0.9
        assert sugg.is_existing is True


class TestPruningSuggestion:
    def test_pruning_suggestion(self):
        sugg = PruningSuggestion(
            conversation_id="old-conv",
            conversation_label="Old Project",
            last_accessed=datetime.now(),
            message_count=150,
            token_estimate=4500,
            importance_score=2.1,
            preview="Old conversation...",
            recommendation="archive"
        )
        assert sugg.recommendation == "archive"
        assert sugg.importance_score < 3.0