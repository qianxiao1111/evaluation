from typing import Optional, List, Tuple, Union
import torch
import torch.distributed
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from encoder_models.encoder2.projector_t5 import Projector
from torch import nn
import os, json

import sys
# TODO fix this by change save function
sys.path.append("/data0/workspace/tjm/review_code")

class TableDecoderConfig(PretrainedConfig):
    model_type = "table_decoder"

    def __init__(self, device='cuda', mm_projector_type='linear', encoder_hidden_size=384,
                 decoder_hidden_size=4096, base_model_name_or_path=None, model_path=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.mm_projector_type = mm_projector_type
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.base_model_name_or_path = base_model_name_or_path
        self.model_path = model_path
    @classmethod
    def from_pretrained(cls, model_path = None, config_dict = None, **kwargs):
        '''
        if model_path or config.model_path is provided, it means the config, model, projector are pretrained; else a new model will be created with the provided config_dict
        '''
        if model_path is not None:
            with open(os.path.join(model_path, 'config.json'), 'r') as f:
                config_dict = json.load(f)
        
        elif config_dict is None:
            raise ValueError("The 'config_dict' must be provided if 'model_path' is None.")            
        
        base_model_name_or_path = config_dict.get('base_model_name_or_path')
        if base_model_name_or_path is None:
            raise ValueError("The config file must contain the 'base_model_name_or_path' field.")
        
        all_dict = {
            **config_dict,
            **kwargs,
            'model_path': model_path
        }
        # 5. 返回合并后的配置对象
        return cls(**all_dict)
    
class TableDecoder(nn.Module):
    def __init__(self, config, base_model, projector=None):
        super().__init__()
        self.dtype = config.torch_dtype
        self.config = config
        if projector != None:
            self.projector = projector
        else:
            self.projector = Projector(config) # TODO: load projector from pretrained model
        self.model = base_model
        
        
    # projector_path='/home/lyli/checkpoint-271/projector_t5_ds1.3_v1.pt'
    @classmethod
    def from_pretrained(cls, decoder_path=None, config=None, projector_path=None): 
        '''
        if model_path or config.model_path is provided, it means the config, model, projector are pretrained; else a new model will be created with the provided config_dict
        '''
        dtype = config.torch_dtype
        if config is None:
            assert decoder_path is not None, "Either 'model_path' or 'config' must be provided."
            config = TableDecoderConfig.from_pretrained(decoder_path)
        # 否则直接用传的config
        '''
        if model_path is None:
            model_path = getattr(config, 'model_path', None) # try to load projector from config; if it fails, just initialize a new projector
        if dtype == torch.bfloat16:            
            base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True)#, attn_implementation="flash_attention_2")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True)
        '''

        base_model = AutoModelForCausalLM.from_pretrained(decoder_path, torch_dtype=dtype, low_cpu_mem_usage=True)

        projector = None
        if projector_path is not None:
            # projector_file = os.path.join(projector_path, 'projector.bin')
            if os.path.exists(projector_path):
                projector = torch.load(projector_path).to(dtype=config.torch_dtype)
                print('projector loaded from', projector_path)
        model = cls(config, base_model, projector)
                
        return model
    
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
        table_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.projector.prepare_embeds(
                self.model,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                table_embeds
            )

        return self.model.forward(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # cache_position=cache_position,
            return_dict=return_dict
        )
    
    def generate(self, input_ids, max_new_tokens, table_embeds, **kwargs):
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.projector.prepare_embeds(
            self.model,
            input_ids,
            # position_ids,
            None,
            None,
            None,
            None,
            table_embeds
        )
        return self.model.generate(
            max_new_tokens=max_new_tokens,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs
        )