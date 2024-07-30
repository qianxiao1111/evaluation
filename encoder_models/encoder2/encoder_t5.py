from encoder_models.base_modules import (
    simple_MLP,
    Transformer,
    RowColTransformer,
)
import torch
import torch.nn as nn

def init_learned_embedding( 
    wte: nn.Embedding,
    config
):
    """
    Args
        `wte` (nn.Embedding): original transformer word embedding
        
        `config`:
            `n_tokens` (int, optional): number of tokens for task. Defaults to 10.
            `random_range` (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            `initialize_from_vocab` (bool, optional): initalizes from default vocab. Defaults to True.
    """
    n_tokens = getattr(config, 'n_tokens', 10)
    random_range = getattr(config, 'random_range', 0.5)
    initialize_from_vocab = getattr(config, 'initialize_from_vocab', True)
    ret = wte.weight[:n_tokens].clone().detach() # 深拷贝并且从计算图中分离
    ret = nn.parameter.Parameter(ret) # tensor -> parameter
    return ret

ROW_SIZE = 50
COLUMN_SIZE = 50
class TableEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_cols,
        dim,
        depth,
        heads,
        dim_head=16,
        attn_dropout=0.0,
        ff_dropout=0.0,
        attentiontype="col",
        final_mlp_style="common",
        decode=False,
        pred_type='generation'
    ):
        super().__init__()

        self.num_cols = num_cols
        self.dim = dim
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style
        self.decode = decode
        self.pred_type=pred_type
        

        # transformer
        if attentiontype == "col":
            self.transformer = Transformer(
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                dim=dim,
                nfeats=num_cols,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
            )
        self.mlp = simple_MLP([dim, dim, dim])
        self.projection_head = simple_MLP([dim, dim, 256])

    @classmethod
    def from_pretrained(cls, model_path = None, config_dict = None):
        args = {
            "num_cols": 100,
            "dim": 384,
            "depth": 12,
            "heads": 6,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
            "dim_head": 64,
        }
        model_path = "/home/xjgu/encoder_pt/Table-Encoder/checkpoints_save/encoder-contrastive-100d-50epochs-0.0001lr.pt"
        
        # init model with give args, then load state dict from model_path
        model = cls(**args)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        return model
    
        
    def forward(
        self,
        anchor_table,
        anchor_table_mask,
        shuffled_table=None,
        shuffled_table_mask=None,
        inference=False
    ):
        x = self.transformer(anchor_table, mask=anchor_table_mask)
        output = x[:,0,:,:]
        return output
