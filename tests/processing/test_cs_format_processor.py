"""
Unit tests for ``cursus.processing.text.cs_format_processor`` — the customer-service chat splitter
(``CSChatSplitterProcessor``) and its dialogue-chunker adapter (``CSAdapter``). Pure regex/string
logic (no heavy deps), previously 0% covered.
"""

from cursus.processing.text.cs_format_processor import (
    CSChatSplitterProcessor,
    CSAdapter,
)


class TestCSChatSplitter:
    def test_splits_roles_and_content(self):
        p = CSChatSplitterProcessor()
        out = p.process("[customer]: hello there [agent]: how can I help?")
        assert out == [
            {"role": "customer", "content": "hello there"},
            {"role": "agent", "content": "how can I help?"},
        ]

    def test_single_message(self):
        p = CSChatSplitterProcessor()
        out = p.process("[bot]: welcome")
        assert out == [{"role": "bot", "content": "welcome"}]

    def test_no_markers_yields_empty(self):
        p = CSChatSplitterProcessor()
        assert p.process("plain text with no role markers") == []

    def test_embedded_messages_are_expanded(self):
        p = CSChatSplitterProcessor()
        # An agent turn that itself contains an embedded customer marker splits into two messages.
        out = p.process("[agent]: main reply [customer]: embedded question")
        roles = [m["role"] for m in out]
        assert "agent" in roles and "customer" in roles
        assert any(m["content"] == "main reply" for m in out)

    def test_get_name_is_stable(self):
        assert CSChatSplitterProcessor().processor_name == "cs_chat_splitter_processor"


class TestCSAdapter:
    def test_roundtrips_splitter_output_to_strings(self):
        splitter = CSChatSplitterProcessor()
        adapter = CSAdapter()
        messages = splitter.process("[customer]: hi [agent]: hello")
        formatted = adapter.process(messages)
        assert formatted == ["[customer]: hi", "[agent]: hello"]

    def test_empty_list(self):
        assert CSAdapter().process([]) == []
