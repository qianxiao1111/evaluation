from encoder_models.table_encoder import TableEncoder
from transformers import AutoModel
import torch
import argparse

def load_encoder(sentence_transformer_path, encoder_path = None, **kwargs):
    args = {
        'num_columns': 20, 
        'embedding_size': 384, 
        'transformer_depth': 12, 
        'attention_heads': 16, 
        'attention_dropout': 0.1, 
        'ff_dropout': 0.1, 
        'dim_head': 64,
        'decode': True,
        'pred_type': 'contrastive'
    }
    # 用kwargs更新args
    args.update(kwargs)
    # 将args转换为命名空间
    args = argparse.Namespace(**args)
    st = AutoModel.from_pretrained(sentence_transformer_path)
    args.embedding_size = st.config.hidden_size
    model = TableEncoder(
            num_cols=args.num_columns,
            depth=args.transformer_depth,
            heads=args.attention_heads,
            attn_dropout=args.attention_dropout,
            ff_dropout=args.ff_dropout,
            attentiontype="colrow",
            # decode=args.decode,
            pred_type=args.pred_type,
            dim_head=args.dim_head,
            pooling='mean',
            col_name=False,
            numeric_mlp=False,
        ).cpu()
    if encoder_path is not None:
        model.load_state_dict(torch.load(encoder_path), strict=False)
        
    return model