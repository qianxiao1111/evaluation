'''
model that supports cross-table generation
'''
from encoder_models.encoder1.multihead_projector import MultiHeadProjector
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Tuple, Union
import json, os
import torch
from encoder_models.utils import tokenize_insert
from torch import nn
import numpy as np
from encoder_models.utils import *
import torch.nn.functional as F
import pandas as pd
from encoder_models.encoder1.load_encoder import load_encoder
from safetensors.torch import load_file
from encoder_models.encoder1.config import MAX_ROW, MAX_COL
if 'MAX_COL' in os.environ:
    MAX_COL = int(os.environ['MAX_COL'])
    print(f'find new MAX_COL in environ: {MAX_COL}')
    
class Model(nn.Module):

    # 将成员属性decoder指向model

    def __init__(self, *, encoder, projector, decoder, tokenizer, encoder_tokenizer, torch_dtype):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.torch_dtype = torch_dtype
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.gradient_checkpointing_enable = self.decoder.gradient_checkpointing_enable
        
        
    @classmethod
    def from_pretrained(cls, path, sentence_transformer_path, base_model_path):
        encoder = load_encoder(sentence_transformer_path).to(dtype = torch.bfloat16)
        projector = MultiHeadProjector(
            projector_type="mlp2x_gelu",
            encoder_hidden_size=3584,
            decoder_hidden_size=3584,
            num_heads=1,
            torch_dtype=torch.bfloat16,
            multihead=False
        )
        decoder = AutoModelForCausalLM.from_pretrained(base_model_path).to(dtype = torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(path)
        encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(sentence_transformer_path)
        model = cls(encoder=encoder, projector=projector, decoder=decoder, tokenizer=tokenizer, torch_dtype=torch.bfloat16, encoder_tokenizer=encoder_tokenizer).to(dtype = torch.bfloat16)
        print('model initialized')

        model.load_state_dict(load_file(os.path.join(path, 'model.safetensors'))) 
        print('model loaded')
        return model

    
    def get_embedded_table(self, path_csv, path_emb = None):
        def process_table_df(table_df):
            numeric_columns = table_df.select_dtypes(include=["number"]).columns
            numeric_indices = [
                table_df.columns.get_loc(col) for col in numeric_columns
            ]
            
            # fill missing values with mean
            table_df[numeric_columns] = table_df[numeric_columns].apply(
                lambda col: col.fillna(col.mean() if not col.isna().all() else 0)
            )
            if len(table_df) > MAX_ROW:
                table_df = table_df.sample(n=MAX_ROW)
                
            
            table_np = table_df.to_numpy().astype(str)
            
            return table_np
        def load_tokenized_table(anchor_table):
            anchor_table = process_table_df(anchor_table)
            num_rows, num_cols = anchor_table.shape[0], anchor_table.shape[1]
            anchor_row_num = anchor_table.shape[0]
            anchor_table = anchor_table.reshape(-1)
            max_length = 64
            tokenized_anchor_table = self.encoder_tokenizer(anchor_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')                
            tokenized_anchor_table = {k: v.reshape(anchor_row_num, num_cols, -1) for k, v in tokenized_anchor_table.items()}
            return tokenized_anchor_table

        try:
            raise
            assert os.path.exists(path_emb)
            # print('=' * 10)
            # print('path_emb', path_emb)
            # print('=' * 10)
            anchor_table_padded, anchor_table_mask, df_col_count = torch.load(path_emb)
        except:
            # print(f'loading csv from {path_csv}')
            table_df = pd.read_csv(
                path_csv,
                encoding="utf-8",
                low_memory=False,
                nrows=500
            )
            df_col_count = table_df.shape[1]
            anchor_table = load_tokenized_table(table_df)
            num_cols = anchor_table['input_ids'].shape[1]
            anchor_table_row_num = anchor_table['input_ids'].shape[0]
            anchor_table_padded = {k: F.pad(v, (0, 0, 0, MAX_COL - v.shape[1], 0, MAX_ROW - v.shape[0]), "constant", 1) for k, v in anchor_table.items()}
            # print('..', anchor_table_padded['input_ids'].shape, anchor_table_padded['attention_mask'].shape, anchor_table_padded['token_type_ids'].shape)
            anchor_table_mask = np.zeros((MAX_ROW, MAX_COL))
            anchor_table_mask[:anchor_table_row_num, : num_cols] = 1
            # torch.save((anchor_table_padded, anchor_table_mask, df_col_count), path_emb)
        ret = (
            anchor_table_padded['input_ids'].to(device = self.decoder.device),
            anchor_table_padded['attention_mask'].to(device = self.decoder.device),
            anchor_table_padded['token_type_ids'].to(device = self.decoder.device),
            torch.tensor(anchor_table_mask, device = self.decoder.device),
            df_col_count
        )
        return ret
            

    
    def get_encoder_output(self, path_csv, path_emb = None):
        anchor_table_input_ids = []
        anchor_table_attention_mask = []
        anchor_table_token_type_ids = []
        anchor_table_mask = []
        
        table_count = [len(c_list) for c_list in path_csv]
        column_count = []
        for c_list in path_csv:
            cur_column_count = []
            for c in c_list:
                p, q, r, s, cnt = self.get_embedded_table(c)
                cur_column_count.append(cnt)
                anchor_table_input_ids.append(p)
                anchor_table_attention_mask.append(q)
                anchor_table_token_type_ids.append(r)
                anchor_table_mask.append(s)
            column_count.append(cur_column_count)
        
        anchor_table_input_ids = torch.stack(anchor_table_input_ids, dim=0)
        anchor_table_attention_mask = torch.stack(anchor_table_attention_mask, dim=0)
        anchor_table_token_type_ids = torch.stack(anchor_table_token_type_ids, dim=0)
        anchor_table_mask = torch.stack(anchor_table_mask, dim=0)    
        table_embeds = self.encoder(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask, inference=True)            
        table_embeds = F.normalize(table_embeds,dim=-1)
        
        table_count = [0] + table_count
        table_count = np.cumsum(table_count)
        # if table_embeds.shape[0] > 2:
        #     print('tc', table_count)
        for question_id in range(len(table_count) - 1):
            tab_l, tab_r = table_count[question_id], table_count[question_id + 1]
            col_count = [0] + column_count[question_id]
            col_count = np.cumsum(col_count)
            # if table_embeds.shape[0] > 2:
            #     print(f'cc{question_id}, tab_l: {tab_l}, tab_r: {tab_r}, col_count: {col_count}')
            for table_id in range(len(col_count) - 1):
                col_l, col_r = col_count[table_id], col_count[table_id + 1]
                # assert col_r <= 50
                table_embeds[question_id][col_l:col_r] = table_embeds[tab_l + table_id][:col_r - col_l]
        # 删去多余的部分
        table_embeds = table_embeds[:len(table_count) - 1]
        return table_embeds
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        path_emb: Optional[str] = None,
        path_csv: Optional[str] = None,
        insert_embs = None # 是否往中间插入embedding
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # get table embeddings
        bs = input_ids.shape[0]
        table_embeds = self.get_encoder_output(path_csv, path_emb)
        prepare_embs_func = self.projector.prepare_embeds if insert_embs == None or insert_embs[0] == False else self.projector.prepare_insert_embeds
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = prepare_embs_func(
            decoder = self.decoder,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            table_embeds=table_embeds,
        )
        return self.decoder.forward(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.to(dtype = self.decoder.dtype),
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # cache_position=cache_position,
            return_dict=return_dict
        )
    
    @torch.inference_mode()
    def generate(self, input_str = None, input_ids = None, path_csv = None, max_new_tokens = 1024, **kwargs):
        if path_csv == None or (type(path_csv) == list and len(path_csv) == 0):
            input_tensor = self.tokenizer(input_str, return_tensors='pt').to(device = self.decoder.device)
            return self.decoder.generate(**input_tensor, max_new_tokens=max_new_tokens, **kwargs)
        
        input_ids = tokenize_insert(input_str, self.tokenizer).unsqueeze(0).to(device = self.decoder.device)
        path_csv = [path_csv]
            
        table_embeds = self.get_encoder_output(path_csv)
        
        prepare_embs_func = self.projector.prepare_insert_embeds
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = prepare_embs_func(
            decoder = self.decoder,
            input_ids=input_ids,
            table_embeds=table_embeds,
        )
        attention_mask = torch.ones(inputs_embeds.shape[:-1], device=inputs_embeds.device)
        
        return self.decoder.generate(
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds.to(dtype = self.decoder.dtype),
            use_cache=True,
            **kwargs
        )
    
    
    