from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CounterfactualGenerator(ABC):
    """Abstract base class for all NN-based counterfactual generators.

    Subclasses implement ``generate``, which takes a dataset and a single test
    sample index and returns a list of counterfactual candidate dicts.

    Each candidate dict has the keys:
        'static'     : np.ndarray, shape (D_static,)
        'meds'       : np.ndarray, shape (T, D_meds)
        'labs'       : np.ndarray, shape (T, D_labs)
        'text'       : raw text string / PID
        'tab'        : alias for 'static'  (compatibility)
        'ts'         : {'labs': ..., 'meds': ...}  (compatibility)
        'text_input' : alias for 'text'  (compatibility)
    """

    @abstractmethod
    def generate(
        self,
        dataset,  # cf_lib.dataset.MultimodalDataset
        sample_idx: int,
        model: Optional[Any] = None,
        target_value: int = 0,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual candidates for one test sample.

        Parameters
        ----------
        dataset      : MultimodalDataset
        sample_idx   : index into dataset.X_test_* arrays
        model        : trained Keras/PyTorch model (required only by IF)
        target_value : desired class for the counterfactuals (default 0)
        k            : number of candidates; overrides the instance default when set

        Returns
        -------
        List of candidate dicts (may be empty if no neighbors found).
        """
