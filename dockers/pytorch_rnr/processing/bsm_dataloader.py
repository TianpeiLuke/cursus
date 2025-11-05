from collections.abc import Callable, Mapping

import torch
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence


def build_collate_batch(
    input_ids_key: str = "input_ids", attention_mask_key: str = "attention_mask"
):
    def collate_batch(batch):
        if not isinstance(batch[0], dict):
            raise TypeError("Batch must contain dictionaries.")
        output = {}
        for key in batch[0]:
            if all(
                isinstance(item[key], list)
                and len(item[key]) > 0
                and isinstance(item[key][0], dict)
                and input_ids_key in item[key][0]
                for item in batch
            ):
                all_input_ids = []
                all_attention_masks = []
                for item in batch:
                    input_chunks = [
                        torch.tensor(chunk[input_ids_key], dtype=torch.long)
                        for chunk in item[key]
                    ]
                    mask_chunks = [
                        torch.tensor(chunk[attention_mask_key], dtype=torch.long)
                        for chunk in item[key]
                    ]
                    all_input_ids.append(pad_sequence(input_chunks, batch_first=True))
                    all_attention_masks.append(
                        pad_sequence(mask_chunks, batch_first=True)
                    )

                def pad_nested(tensors):
                    max_chunks = max(t.size(0) for t in tensors)
                    max_len = max(t.size(1) for t in tensors)
                    padded = []
                    for t in tensors:
                        pad_chunk = max_chunks - t.size(0)
                        pad_len = max_len - t.size(1)
                        padded.append(
                            torch.nn.functional.pad(t, (0, pad_len, 0, pad_chunk))
                        )
                    return torch.stack(padded)

                output[key + "_" + input_ids_key] = pad_nested(all_input_ids)
                output[key + "_" + attention_mask_key] = pad_nested(all_attention_masks)
            else:
                output[key] = [item[key] for item in batch]
        return output

    return collate_batch


def build_trimodal_collate_batch():
    """
    Enhanced collate function for tri-modal data that handles multiple text fields.
    Automatically detects and processes all text fields ending with '_processed'.
    """

    def collate_batch(batch):
        if not isinstance(batch[0], dict):
            raise TypeError("Batch must contain dictionaries.")
        output = {}

        for key in batch[0]:
            # Check if this is a processed text field (list of tokenized chunks)
            if key.endswith("_processed") and all(
                isinstance(item[key], list)
                and len(item[key]) > 0
                and isinstance(item[key][0], dict)
                and "input_ids" in item[key][0]
                and "attention_mask" in item[key][0]
                for item in batch
            ):
                # Process tokenized text field
                all_input_ids = []
                all_attention_masks = []

                for item in batch:
                    input_chunks = [
                        torch.tensor(chunk["input_ids"], dtype=torch.long)
                        for chunk in item[key]
                    ]
                    mask_chunks = [
                        torch.tensor(chunk["attention_mask"], dtype=torch.long)
                        for chunk in item[key]
                    ]
                    all_input_ids.append(pad_sequence(input_chunks, batch_first=True))
                    all_attention_masks.append(
                        pad_sequence(mask_chunks, batch_first=True)
                    )

                def pad_nested(tensors):
                    max_chunks = max(t.size(0) for t in tensors)
                    max_len = max(t.size(1) for t in tensors)
                    padded = []
                    for t in tensors:
                        pad_chunk = max_chunks - t.size(0)
                        pad_len = max_len - t.size(1)
                        padded.append(
                            torch.nn.functional.pad(t, (0, pad_len, 0, pad_chunk))
                        )
                    return torch.stack(padded)

                # Create the expected tensor keys
                output[key + "_input_ids"] = pad_nested(all_input_ids)
                output[key + "_attention_mask"] = pad_nested(all_attention_masks)

            else:
                # Handle non-text fields (tabular, labels, etc.)
                output[key] = [item[key] for item in batch]

        return output

    return collate_batch
