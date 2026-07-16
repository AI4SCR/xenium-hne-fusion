import pandas as pd
import pytest

from xenium_hne_fusion.metadata import normalize_sample_metadata


def test_normalize_sample_metadata_rejects_plain_id_column():
    metadata = pd.DataFrame([{'id': 'S1', 'patient': 'P1'}])

    with pytest.raises(AssertionError, match='Metadata must contain sample_id column'):
        normalize_sample_metadata(metadata)
