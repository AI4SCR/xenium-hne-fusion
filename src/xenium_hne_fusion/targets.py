"""Fixed target-label vocabularies shared across datasets and training."""

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROTEIN_PANEL = [
    'Beta-catenin', 'CD11c', 'CD138', 'CD16', 'CD163-1', 'CD20', 'CD31', 'CD3E-1', 'CD4-1', 'CD45',
    'CD45RA', 'CD45RO', 'CD68-1', 'CD8A-1', 'E-Cadherin', 'GranzymeB', 'HLA-DR', 'Ki-67', 'LAG-3', 'PCNA',
    'PD-1', 'PD-L1', 'PTEN-1', 'PanCK', 'VISTA', 'Vimentin', 'alphaSMA',
]


def protein_base_name(protein: str) -> str:
    """Strip a trailing "-<digit>" batch suffix (e.g. "PCNA-1" -> "PCNA")."""
    return re.sub(r"-\d+$", "", protein)


def protein_base_to_panel(proteins: list[str]) -> dict[str, str]:
    """Map each protein's base name (see `protein_base_name`) back to its panel name.

    Marker naming (e.g. "PCNA" vs "PCNA-1") is inconsistent across sample batches;
    this lets callers normalize a batch's column names onto `proteins`.
    """
    base_to_panel = {protein_base_name(p): p for p in proteins}
    assert len(base_to_panel) == len(proteins), "panel has ambiguous base names"
    return base_to_panel


def load_sample_proteins(structured_dir: Path, sample_id: str, proteins: list[str]) -> pd.DataFrame:
    """Read a sample's structured `proteins.parquet`, normalized onto `proteins` columns.

    Skips the "geometry" column at read time rather than loading and dropping it.
    """
    base_to_panel = protein_base_to_panel(proteins)
    path = structured_dir / sample_id / "proteins.parquet"
    columns = [c for c in pq.ParquetFile(path).schema.names if c != "geometry"]
    df = pd.read_parquet(path, columns=columns)
    rename = {c: base_to_panel[protein_base_name(c)] for c in df.columns}
    return df.rename(columns=rename)[proteins]

# Index-ordered: conch_class labels (see scribble/create-conch-training.py) are argmax indices into this list.
CONCH_CLASSES = [
    "epithelioid nests",
    "tubulopapillary epithelioid",
    "solid sheets of epithelioid",
    "inflamed epithelioid",
    "cold, epithelioid, solid",
    "nests and trabeculae, cold",
    "cuboidal epithelioid",
    "papillae and micropapillae",
    "tubulopapillary",  # end of blues
    "disorderly spindle cells",
    "dense spindle cells",
    "desmoplastic sarcomatoid",
    "inflamed spindle cells",  # end of oranges
    "inflamed, fibrotic",
    "inflamed, malignant",  # end of reds
    "connective tissues",
    "epithelioid nests in bland stroma",
    "basket-weave collagen",
    "collagen, tiny nuclei",
    "bland spindle cells and collagen",  # end of greens
    "dense lymphocytes",
    "skeletal muscle",
    "muscle, transverse",  # end of pink
    "inflamed fat",
    "fat",
    "fibrotically infiltrated fat",  # end of yellows
    "diverse necrotic tissues",
    "diathermy and crush",  # end of turquoises
    "pleural plaque",
    "talc reaction",
    "vessels",
    "airways",
]
