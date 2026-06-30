from typing import List, Optional, Dict
from transformers import AutoTokenizer

from ..processors import Processor


# --- Processor 6: Tokenization Processor using AutoTokenizer ---
class BertTokenizeProcessor(Processor):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: str = "longest",
        input_ids_key: str = "input_ids",  # Added input_ids_key
        attention_mask_key: str = "attention_mask",  # Added attention_mask_key
    ):
        super().__init__()
        self.processor_name = "tokenization_processor"
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.input_ids_key = input_ids_key  # Store the key names
        self.attention_mask_key = attention_mask_key  # Store the key names

    def process(self, input_chunks: List[str]) -> List[Dict[str, List[int]]]:
        tokenized_output = []

        for chunk in input_chunks:
            # Tokenize empty/whitespace-only chunks too (as ""), rather than
            # silently dropping them. Skipping made the output shorter than the
            # input, breaking 1:1 cardinality for any consumer that zips them.
            if not chunk or not chunk.strip():
                chunk = ""

            encoded = self.tokenizer(
                chunk,
                add_special_tokens=self.add_special_tokens,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_attention_mask=True,
            )
            tokenized_output.append(
                {
                    self.input_ids_key: encoded["input_ids"],  # Use stored key
                    self.attention_mask_key: encoded[
                        "attention_mask"
                    ],  # Use stored key
                }
            )
        return tokenized_output
