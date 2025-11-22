"""
Pytest tests for DialogueChunkerProcessor.

Tests the critical functionality of chunking dialogue messages while ensuring
no individual chunk exceeds the max_tokens limit, preventing OOM errors in
downstream BERT models.
"""

import pytest
from unittest.mock import Mock
from transformers import AutoTokenizer


# Import from the cursus processing module
import sys
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from cursus.processing.text.dialogue_processor import DialogueChunkerProcessor


class TestDialogueChunkerProcessor:
    """Test suite for DialogueChunkerProcessor."""

    @pytest.fixture
    def tokenizer(self):
        """Fixture providing a BERT tokenizer."""
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    @pytest.fixture
    def processor(self, tokenizer):
        """Fixture providing a DialogueChunkerProcessor with default settings."""
        return DialogueChunkerProcessor(
            tokenizer=tokenizer, max_tokens=512, truncate=False, max_total_chunks=None
        )

    @pytest.fixture
    def processor_with_truncate(self, tokenizer):
        """Fixture providing a processor with truncate enabled and max chunks."""
        return DialogueChunkerProcessor(
            tokenizer=tokenizer, max_tokens=512, truncate=True, max_total_chunks=3
        )

    def test_basic_chunking(self, processor):
        """Test basic chunking with multiple short messages."""
        messages = [
            "Hello, how are you?",
            "I'm doing well, thank you!",
            "That's great to hear.",
        ]

        chunks = processor.process(messages)

        # Should create a single chunk since all messages are short
        assert len(chunks) == 1
        assert "Hello" in chunks[0]
        assert "doing well" in chunks[0]
        assert "great to hear" in chunks[0]

    def test_chunking_with_long_messages(self, processor):
        """Test chunking when messages collectively exceed max_tokens."""
        # Create messages that will require multiple chunks
        messages = [
            "This is message one. " * 50,  # ~100 tokens
            "This is message two. " * 50,  # ~100 tokens
            "This is message three. " * 50,  # ~100 tokens
            "This is message four. " * 50,  # ~100 tokens
            "This is message five. " * 50,  # ~100 tokens
            "This is message six. " * 50,  # ~100 tokens
        ]

        chunks = processor.process(messages)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Verify each chunk is within token limit
        for chunk in chunks:
            token_count = len(
                processor.tokenizer.encode(chunk, add_special_tokens=False)
            )
            assert token_count <= 512, f"Chunk exceeds max_tokens: {token_count} > 512"

    def test_single_oversized_message_truncation(self, processor):
        """
        CRITICAL TEST: Verify that a single message exceeding max_tokens is truncated.
        This prevents the OOM error that was occurring in production.
        """
        # Create a message with ~1600 tokens (similar to production issue)
        oversized_message = (
            "This is a very long message that exceeds the token limit. " * 100
        )

        # Verify the message is indeed oversized
        original_token_count = len(
            processor.tokenizer.encode(oversized_message, add_special_tokens=False)
        )
        assert original_token_count > 512, "Test message should exceed 512 tokens"

        messages = [oversized_message]
        chunks = processor.process(messages)

        # Should create exactly one chunk
        assert len(chunks) == 1

        # CRITICAL: Verify the chunk does not exceed max_tokens
        chunk_token_count = len(
            processor.tokenizer.encode(chunks[0], add_special_tokens=False)
        )
        assert chunk_token_count <= 512, (
            f"Chunk should be truncated to max_tokens. "
            f"Got {chunk_token_count} tokens, expected <= 512"
        )

    def test_multiple_oversized_messages(self, processor):
        """Test handling of multiple oversized messages."""
        # Create multiple messages that each exceed max_tokens
        messages = [
            "First oversized message. " * 150,  # ~600 tokens
            "Second oversized message. " * 150,  # ~600 tokens
            "Third oversized message. " * 150,  # ~600 tokens
        ]

        chunks = processor.process(messages)

        # Each oversized message should become its own truncated chunk
        assert len(chunks) == 3

        # Verify each chunk is within limit
        for i, chunk in enumerate(chunks):
            token_count = len(
                processor.tokenizer.encode(chunk, add_special_tokens=False)
            )
            assert token_count <= 512, (
                f"Chunk {i} exceeds max_tokens: {token_count} > 512"
            )

    def test_mixed_sized_messages(self, processor):
        """Test chunking with a mix of normal and oversized messages."""
        messages = [
            "Short message.",
            "This is an oversized message that exceeds the token limit. "
            * 100,  # ~1600 tokens
            "Another short message.",
            "Final short message.",
        ]

        chunks = processor.process(messages)

        # Should create multiple chunks
        assert len(chunks) >= 2

        # Verify all chunks are within limit
        for i, chunk in enumerate(chunks):
            token_count = len(
                processor.tokenizer.encode(chunk, add_special_tokens=False)
            )
            assert token_count <= 512, (
                f"Chunk {i} exceeds max_tokens: {token_count} > 512"
            )

    def test_truncate_max_chunks(self, processor_with_truncate):
        """Test that truncate parameter limits the number of chunks created."""
        # Create many messages that would normally create >3 chunks
        messages = [f"Message number {i}. " * 50 for i in range(20)]

        chunks = processor_with_truncate.process(messages)

        # Should respect max_total_chunks=3
        assert len(chunks) <= 3, (
            f"Expected max 3 chunks with truncate=True, got {len(chunks)}"
        )

    def test_empty_messages(self, processor):
        """Test handling of empty or whitespace-only messages."""
        messages = ["", "  ", "Valid message", "", "Another valid"]

        chunks = processor.process(messages)

        # Should still create valid chunks
        assert len(chunks) >= 1
        assert "Valid message" in chunks[0] or "Another valid" in chunks[0]

    def test_single_short_message(self, processor):
        """Test handling of a single short message."""
        messages = ["Hello world"]

        chunks = processor.process(messages)

        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_no_messages_fallback(self, processor):
        """Test fallback behavior when no messages provided."""
        messages = []

        chunks = processor.process(messages)

        # Should return at least one non-empty chunk (fallback behavior)
        assert len(chunks) >= 1
        assert chunks[0] == "."

    def test_extreme_oversized_message(self, processor):
        """Test handling of extremely large messages (>2000 tokens)."""
        # Create an extremely large message
        extreme_message = "This is an extremely long message. " * 300  # ~3000 tokens

        original_count = len(
            processor.tokenizer.encode(extreme_message, add_special_tokens=False)
        )
        assert original_count > 2000, "Test message should be extremely large"

        messages = [extreme_message]
        chunks = processor.process(messages)

        # Should create one truncated chunk
        assert len(chunks) == 1

        chunk_count = len(
            processor.tokenizer.encode(chunks[0], add_special_tokens=False)
        )
        assert chunk_count <= 512, (
            f"Even extreme messages should be truncated to 512 tokens, got {chunk_count}"
        )

    def test_token_count_accuracy(self, processor):
        """Verify that token counting is accurate and matches tokenizer behavior."""
        message = "This is a test message for token counting accuracy."

        # Count tokens using the processor's method
        tokens = processor.tokenizer.encode(message, add_special_tokens=False)
        expected_count = len(tokens)

        # Process and verify
        chunks = processor.process([message])
        chunk_tokens = processor.tokenizer.encode(chunks[0], add_special_tokens=False)
        actual_count = len(chunk_tokens)

        assert actual_count <= expected_count
        assert actual_count <= 512

    def test_special_characters(self, processor):
        """Test handling of messages with special characters."""
        messages = [
            "Message with emojis 😊🎉",
            "Message with special chars: @#$%^&*()",
            "Message with unicode: café, naïve, 日本語",
        ]

        chunks = processor.process(messages)

        # Should handle without errors
        assert len(chunks) >= 1

        # Verify token limits
        for chunk in chunks:
            token_count = len(
                processor.tokenizer.encode(chunk, add_special_tokens=False)
            )
            assert token_count <= 512

    def test_punctuation_preservation(self, processor):
        """Test that punctuation is preserved during chunking."""
        messages = ["Hello! How are you?", "I'm great, thanks.", "What's your name?"]

        chunks = processor.process(messages)
        result = " ".join(chunks)

        # Key punctuation should be preserved
        assert "!" in result or "?" in result or "," in result

    @pytest.mark.parametrize("max_tokens", [128, 256, 512, 1024])
    def test_different_max_tokens(self, tokenizer, max_tokens):
        """Test processor with different max_token settings."""
        processor = DialogueChunkerProcessor(
            tokenizer=tokenizer, max_tokens=max_tokens, truncate=False
        )

        # Create an oversized message
        oversized = "Test message. " * 300

        chunks = processor.process([oversized])

        # Verify chunks respect the specified max_tokens
        for chunk in chunks:
            token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
            assert token_count <= max_tokens, (
                f"Chunk exceeds specified max_tokens={max_tokens}: {token_count}"
            )


class TestDialogueChunkerEdgeCases:
    """Additional edge case tests for DialogueChunkerProcessor."""

    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_whitespace_only_messages(self, tokenizer):
        """Test handling of whitespace-only messages."""
        processor = DialogueChunkerProcessor(tokenizer, max_tokens=512)
        messages = ["   ", "\t\t", "\n\n", "Valid"]

        chunks = processor.process(messages)

        # Should handle gracefully and include the valid message
        assert any("Valid" in chunk for chunk in chunks)

    def test_very_short_max_tokens(self, tokenizer):
        """Test with unrealistically small max_tokens."""
        processor = DialogueChunkerProcessor(tokenizer, max_tokens=10)
        messages = ["This is a message that will definitely exceed 10 tokens"]

        chunks = processor.process(messages)

        # Should still truncate to max_tokens
        for chunk in chunks:
            token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
            assert token_count <= 10

    def test_chunk_order_preservation(self, tokenizer):
        """Test that message order is preserved in chunks."""
        processor = DialogueChunkerProcessor(tokenizer, max_tokens=512)
        messages = ["First", "Second", "Third", "Fourth", "Fifth"]

        chunks = processor.process(messages)
        combined = " ".join(chunks)

        # Verify order
        first_idx = combined.find("First")
        second_idx = combined.find("Second")
        fifth_idx = combined.find("Fifth")

        assert first_idx < second_idx < fifth_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
