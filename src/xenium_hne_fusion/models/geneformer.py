import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from anndata import AnnData
from helical.utils.mapping import convert_list_gene_symbols_to_ensembl_ids
from typing import Literal, Callable
from ai4bmr_learn.utils.device import get_device


class Geneformer(nn.Module):
    def __init__(
        self,
        gene_names: list[str],
        transform: Callable,
        model_name: str = "gf-12L-38M-i4096",
        emb_mode: Literal['cls', 'cell'] = 'cls',
        nproc: int = 8,
    ):
        super().__init__()
        from helical.models.geneformer import Geneformer, GeneformerConfig

        self.gene_names = gene_names
        self.transform = transform
        self.ensembl_ids = convert_list_gene_symbols_to_ensembl_ids(gene_names)

        # TODO: nproc could be a lever for speedup
        device = get_device()
        model_config = GeneformerConfig(model_name=model_name, emb_mode=emb_mode, device=device, nproc=nproc)
        self.model = Geneformer(model_config)
        # TODO set head to identity?

        self.embed_dim = self.model.model.config.hidden_size

        self.has_special_tokens = model_config.config['special_token']
        self.embed_layer = model_config.config['emb_layer']

    def forward(self, expr_tokens: torch.Tensor):
        # breakpoint()
        device = expr_tokens.device
        *original_shape, n_genes = expr_tokens.shape

        if n_genes == 0:
            logger.info('No genes in the expression tokens, returning zero embeddings.')
            # TODO: could per learnable params instead of zeros
            return torch.zeros(*expr_tokens.shape[:-1], self.embed_dim, device=device)

        expr_tokens = expr_tokens.reshape(-1, n_genes)
        expr_tokens = self.transform(expr_tokens)
        ad = AnnData(X=expr_tokens.cpu().numpy(force=True),
                     var=pd.DataFrame({'ensembl_id': self.ensembl_ids}),
                     obs=pd.DataFrame({'filter_pass': True}, index=[str(i) for i in range(len(expr_tokens))]))
        dataset = self.model.process_data(ad, gene_names='ensembl_id')
        # NOTE: we had to fix the source code and replace the np.array(embs_list) with torch.stack
        embeddings = self.model.get_embeddings(dataset)
        # embeddings = torch.tensor(embeddings, device=device)
        embeddings = embeddings.reshape(*original_shape, -1)
        return embeddings

