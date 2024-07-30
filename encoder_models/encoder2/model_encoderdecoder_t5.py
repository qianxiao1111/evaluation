from encoder_models.encoder2.decoder_t5 import TableDecoder
from transformers import AutoTokenizer, PretrainedConfig, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Tuple, Union
import json, os
import torch
from torch import nn

# from sentence_transformers import SentenceTransformer
class MyConfig(PretrainedConfig):
    model_type = "my_emb_model"

    def __init__(self, device='cuda', mm_projector_type='linear', encoder_hidden_size=1024,
                 decoder_hidden_size=2048, base_model_name_or_path=None, model_path=None, db_path=None, torch_dtype="float32", **kwargs):
        super().__init__(torch_dtype=torch_dtype) 
        self.device = device
        self.mm_projector_type = mm_projector_type
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.base_model_name_or_path = base_model_name_or_path
        self.model_path = model_path
        self.db_path = db_path
        
    @classmethod
    def from_pretrained(cls, model_path = None, config_dict = None, **kwargs): # if model_path is provided, it means the config, model, learnable_embedding are pretrained; otherwise a new model will be created with the provided config_dict
        if model_path is not None:
            with open(os.path.join(model_path, 'config.json'), 'r') as f:
                config_dict = json.load(f)
        
        elif config_dict is None:
            raise ValueError("The 'config_dict' must be provided if 'model_path' is None.")            
        
        all_dict = {
            **config_dict,
            **kwargs,
            'model_path': model_path
        }
        return cls(**all_dict)
    
class Model(nn.Module):

    # 将成员属性decoder指向model

    def __init__(self, *, config, encoder, decoder, tokenizer_encoder, tokenizer_decoder):
        super().__init__()
        self.config = config
        self.gradient_checkpointing_enable = decoder.model.gradient_checkpointing_enable
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        self.torch_dtype = config.torch_dtype
        self.db_path = config.db_path

    @classmethod
    def from_pretrained(cls, encoder_path=None, decoder_path=None, projector_path=None, config_dict = None): # if model_path is provided, it means the config, model, learnable_embedding are pretrained; else a new model will be created with the provided config_dict

        config = MyConfig.from_pretrained(model_path=decoder_path, config_dict=config_dict)
        # encoder = torch.load(encoder_path).to(dtype=config.torch_dtype)
        print('load encoder from', encoder_path)
        decoder = TableDecoder.from_pretrained(decoder_path=decoder_path, config=config, projector_path=projector_path)
        decoder.projector = decoder.projector.to(decoder.model.dtype)
        print('load decoder from', decoder_path)
        # tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        if encoder_path is None:
            encoder = None
            tokenizer_encoder = None
        else:
            encoder = AutoModelForSeq2SeqLM.from_pretrained(encoder_path, trust_remote_code=True, torch_dtype=torch.float).encoder

            encoder = encoder.to(decoder.model.dtype)

            tokenizer_encoder = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)


        tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)

        model = cls(config=config, encoder=encoder, decoder=decoder, tokenizer_encoder=tokenizer_encoder, tokenizer_decoder=tokenizer_decoder)
        return model

    '''
    def get_embedded_table(self, db_id, table):
        torch.manual_seed(11)
        np.random.seed(11)
        row_size = 50
        column_size = 50
        # db_path = self.config.db_path
        db_path = '/home/xjgu/spider/database_csv'

        embedding_dim = self.sentencetransformer.get_sentence_embedding_dimension()
        
        table_df = pd.read_csv(os.path.join(db_path, db_id, table + '.csv'), encoding='utf-8', low_memory=False)
        
        if len(table_df) > row_size:
            table_df = table_df.sample(n=row_size)
        table = table_df.to_numpy()
        table = table.astype(str)
        table_emb = np.zeros((table.shape[0], table.shape[1], embedding_dim))
        for j, row in enumerate(table):
            row_emb = self.sentencetransformer.encode(row)
            table_emb[j] = row_emb
        
        column_truncation = np.random.permutation(range(table_emb.shape[1]))[:column_size]
        table_emb = table_emb[:, column_truncation, :]

        table_emb_padded = np.zeros((row_size, 100, table_emb.shape[2]))
        table_emb_padded[:table_emb.shape[0], :table_emb.shape[1], :] = table_emb
        table_mask = np.zeros((row_size, 100))
        table_mask[:table_emb.shape[0], :table_emb.shape[1]] = 1

        
        return torch.tensor(table_emb_padded, device=self.decoder.model.device, dtype = self.torch_dtype), torch.tensor(table_mask, device=self.decoder.model.device, dtype = self.torch_dtype)
    '''

    
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
        db_ids = None,
        tables = None,
        encoder_input_ids = None,
        decoder_input_ids = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]: # TODO: 清理一下参数
        '''
        if db_ids != None:
            bs = input_ids.shape[0]
            table_embeds = []
            for i in range(bs):
                table_embeds.append(self.get_embedded_table(db_ids[i], tables[i]))
                
            table_embeds = []
            table_masks = []
            for db_id, table in zip(db_ids, tables):
                table_emb, table_mask = self.get_embedded_table(db_id, table)
                table_embeds.append(table_emb)
                table_masks.append(table_mask)
            table_embeds = torch.stack(table_embeds).detach()
            table_masks = torch.stack(table_masks).detach()
            table_embeds = self.encoder(anchor_table=table_embeds, anchor_table_mask=table_masks, inference=True)
        '''
        if self.encoder is not None and encoder_input_ids is not None:
            encoder_outputs = self.encoder(input_ids=encoder_input_ids).last_hidden_state
        else:
            encoder_outputs = None

        return self.decoder.forward(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            table_embeds=encoder_outputs
        )
    
    def generate(self, input_str = None, input_ids = None, table_str = None, db_id = None, table = None, max_new_tokens = 114514, **kwargs):
        if input_ids is None:
            input_ids = self.tokenizer_decoder(input_str, return_tensors='pt')['input_ids'].to(self.decoder.model.device)
        bs = input_ids.shape[0]
        assert bs == 1, "batch size must be 1"
        
        '''
        table_emb, table_mask = self.get_embedded_table(db_id, table)
        table_embeds = self.encoder(anchor_table=table_emb.unsqueeze(0), anchor_table_mask=table_mask.unsqueeze(0), inference=True)
        # table_embeds = F.normalize(table_embeds,dim=-1)
        '''
        if self.encoder is not None and table_str is not None:
            encoder_input_ids = self.tokenizer_encoder(table_str, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(self.encoder.device)
            encoder_outputs = self.encoder(input_ids=encoder_input_ids).last_hidden_state
            assert len(encoder_outputs.shape) == 3, "encoder_outputs must be 3D" # bs * n_tokens * hidden_size
        else:
            encoder_outputs = None
        return self.decoder.generate(input_ids, max_new_tokens, encoder_outputs, **kwargs)
    
    
    