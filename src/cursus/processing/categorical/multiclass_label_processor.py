import numpy as np
from typing import List, Union, Optional, Dict


from ..processors import Processor


class MultiClassLabelProcessor(Processor):
    """
    Processes multi-class labels into a format suitable for machine learning models.

    This processor handles encoding categorical labels into numerical arrays
    and optionally provides one-hot encoding using numpy.

    Args:
        label_list (Optional[List[str]]): A list of unique label strings. If provided,
                                     the processor will learn this mapping; otherwise,
                                     it will learn the mapping from the data it processes.
        one_hot (bool): If True, output one-hot encoded labels.
        strict (bool): If True, raise error for unknown labels when label_list is provided.
    """

    def __init__(
        self,
        label_list: Optional[List[str]] = None,
        one_hot: bool = False,
        strict: bool = False,
    ):
        super().__init__()
        self.processor_name = "multiclass_label_processor"
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: List[str] = []
        self.one_hot = one_hot
        self.strict = strict

        # `is not None` (not truthiness): an explicitly-empty label_list=[] means
        # "fixed vocab, currently empty" and must be distinguishable from None
        # ("learn the vocab from data").
        if label_list is not None:
            self.label_to_id = {
                self._normalize(label): i for i, label in enumerate(label_list)
            }
            self.id_to_label = [self._normalize(label) for label in label_list]

    @staticmethod
    def _normalize(label) -> str:
        """Normalize a label to a stable string key.

        Coerce integer-valued floats to their int form first so that 1, 1.0,
        "1" all map to the same key "1" (str(1.0) would otherwise yield "1.0"
        and silently create a separate class from the int/str 1).
        """
        if isinstance(label, float) and label.is_integer():
            return str(int(label))
        return str(label)

    def process(self, labels: Union[str, List[str]]) -> np.ndarray:
        """
        Encodes the input labels.

        Args:
            labels (Union[str, List[str]]): A single label or a list of labels.

        Returns:
            np.ndarray: Encoded labels as a numpy array.
        """

        if isinstance(labels, (str, int, float)):
            labels = [labels]  # Wrap scalar in list

        encoded_labels = []
        for label in labels:
            label = self._normalize(label)  # consistent with __init__ mapping
            if label not in self.label_to_id:
                if self.strict:
                    raise ValueError(f"Label '{label}' not found in known label list.")
                self.label_to_id[label] = len(self.label_to_id)
                self.id_to_label.append(label)
            encoded_labels.append(self.label_to_id[label])

        encoded_array = np.array(encoded_labels, dtype=np.int64)

        if self.one_hot:
            # Create one-hot encoding using numpy
            num_classes = len(self.id_to_label)
            one_hot_labels = np.eye(num_classes)[encoded_array].astype(np.float32)
            return one_hot_labels
        else:
            return encoded_array
