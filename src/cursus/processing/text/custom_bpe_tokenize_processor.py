"""
Custom BPE Tokenization Processor

Tokenization processor compatible with HuggingFace tokenizers.Tokenizer
(custom BPE tokenizers from tokenizer_training step).

This processor is specifically designed for lstm2risk/transformer2risk models
that use custom BPE tokenizers. Outputs token IDs, attention masks, and
text lengths required by bimodal models.

Key Features:
- Uses tokenizers.Tokenizer.encode() for BPE tokenization
- Outputs attention_mask for Transformer2Risk (masks padding tokens)
- Outputs text_length for LSTM2Risk (enables pack_padded_sequence)
- Configurable output key names for flexibility
- Padding applied when max_length is specified

Usage:
    from tokenizers import Tokenizer
    from processing.text.custom_bpe_tokenize_processor import CustomBPETokenizeProcessor

    tokenizer = Tokenizer.from_file("tokenizer.json")
    processor = CustomBPETokenizeProcessor(
        tokenizer=tokenizer,
        add_special_tokens=True,
        max_length=100,
        padding=True  # Pads to max_length
    )

    data = processor({"text_field": "0.5|0.3|0.8|0.2"})
    # Returns: {
    #   "text": tensor([101, 15, 234, 45, 102, 0, 0, ...]),  # Padded to max_length
    #   "attn_mask": tensor([1, 1, 1, 1, 1, 0, 0, ...]),    # 1=real, 0=padding
    #   "text_length": 5                                     # Real token count
    # }
"""

from typing import Optional, Dict, Any
import torch
from tokenizers import Tokenizer  # HuggingFace tokenizers library

from ..processors import Processor


class CustomBPETokenizeProcessor(Processor):
    """
    Tokenization processor for custom BPE tokenizer (HuggingFace tokenizers.Tokenizer).

    Outputs token IDs, attention masks, and sequence lengths for lstm2risk/transformer2risk.

    Args:
        tokenizer: HuggingFace Tokenizer instance (from tokenizers library)
        add_special_tokens: Whether to add special tokens like [CLS], [SEP]
        max_length: Maximum sequence length (truncates if longer, pads if padding=True)
        padding: Whether to pad sequences to max_length (default: True if max_length set)
        pad_token_id: Token ID for padding (default: 0)
        input_ids_key: Key name for token IDs in output dict (default: "text")
        attention_mask_key: Key name for attention mask (default: "attn_mask")
        text_length_key: Key name for text length (default: "text_length")

    Returns:
        Dictionary with token IDs, attention mask, and text length

    Example:
        >>> from tokenizers import Tokenizer
        >>> tokenizer = Tokenizer.from_file("tokenizer.json")
        >>> processor = CustomBPETokenizeProcessor(tokenizer, max_length=100)
        >>> data = processor({"text_field": "Alice|alice@email.com"})
        >>> data.keys()
        dict_keys(['text_field', 'text', 'attn_mask', 'text_length'])
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Optional[bool] = None,
        pad_token_id: int = 0,
        input_ids_key: str = "input_ids",
        attention_mask_key: str = "attention_mask",
        text_length_key: str = "text_length",
    ):
        super().__init__()
        self.processor_name = "custom_bpe_tokenization_processor"
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        # Default: pad if max_length is set
        self.padding = padding if padding is not None else (max_length is not None)
        self.pad_token_id = pad_token_id
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key
        self.text_length_key = text_length_key

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize input text using custom BPE tokenizer.

        Args:
            data: Dictionary containing input text (expects string under some key)

        Returns:
            Updated data dictionary with added keys:
                - input_ids_key: token IDs tensor (B, L)
                - attention_mask_key: attention mask (B, L) - 1=real, 0=padding
                - text_length_key: sequence length (scalar) - count before padding

        Note:
            - Returns empty tensors for empty/whitespace-only input
            - Applies truncation if max_length is set
            - Applies padding if padding=True and max_length is set
            - Text length always reflects real tokens (before padding)
        """
        # Get input text from data (first string value found)
        input_text = None
        for value in data.values():
            if isinstance(value, str):
                input_text = value
                break

        # Handle empty input
        if not input_text or not input_text.strip():
            # Return empty tensors
            data[self.input_ids_key] = torch.tensor([], dtype=torch.long)
            data[self.attention_mask_key] = torch.tensor([], dtype=torch.long)
            data[self.text_length_key] = 0
            return data

        # Encode using custom tokenizer
        # tokenizer.encode() returns Encoding object with .ids attribute
        encoding = self.tokenizer.encode(
            input_text,
            add_special_tokens=self.add_special_tokens,
        )

        # Extract token IDs as list
        ids = encoding.ids

        # Store original length (before truncation/padding)
        text_length = len(ids)

        # Apply truncation if max_length is specified
        if self.max_length and len(ids) > self.max_length:
            ids = ids[: self.max_length]
            text_length = self.max_length  # Update length after truncation

        # Apply padding if enabled
        if self.padding and self.max_length:
            # Pad to max_length
            padding_length = self.max_length - len(ids)
            if padding_length > 0:
                ids = ids + [self.pad_token_id] * padding_length

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * text_length
        if self.padding and self.max_length:
            padding_length = self.max_length - text_length
            if padding_length > 0:
                attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors and store in data dictionary
        data[self.input_ids_key] = torch.tensor(ids, dtype=torch.long)
        data[self.attention_mask_key] = torch.tensor(attention_mask, dtype=torch.long)
        data[self.text_length_key] = text_length

        return data
