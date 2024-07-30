import torch
import torch.nn as nn
import re

IGNORE_INDEX = -100

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class Projector(nn.Module):

    def __init__(self, config, delay_load=False, encoder_hidden_size=1024, decoder_hidden_size=3584, **kwargs):
        """
        Build a table projector based on the given configuration.

        Args:
            config (object): mm_projector_type: The type of projector to use. Defaults to 'linear'; hidden_size: ...
            **kwargs: Additional keyword arguments.

        Returns:
            object: The table projector.

        Raises:
            ValueError: If the projector type is unknown.
        """
        super().__init__()
        # print('config:',config)
        # projector_type = config['mm_projector_type']
        # encoder_hidden_size = config['encoder_hidden_size']
        # decoder_hidden_size = config['decoder_hidden_size']
        '''
        projector_type = getattr(config, 'mm_projector_type')
        encoder_hidden_size = getattr(config, 'encoder_hidden_size') # TODO: auto detect
        decoder_hidden_size = getattr(config, 'decoder_hidden_size')
        dtype = config.torch_dtype
        if projector_type == 'linear':
            self.model = nn.Linear(encoder_hidden_size, decoder_hidden_size, dtype = dtype)
            # print(next(self.model.parameters()).dtype)
            return

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(encoder_hidden_size, decoder_hidden_size, dtype = dtype)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(decoder_hidden_size, decoder_hidden_size, dtype = dtype))
            self.model = nn.Sequential(*modules)
            return

        if projector_type == 'identity':
            self.model = IdentityMap()
            return

        self.model.float()
        raise ValueError(f'Unknown projector type: {projector_type}')

    def from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        raise NotImplementedError()
    '''

        mlp_depth = 1
        modules = [nn.Linear(encoder_hidden_size, decoder_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
        self.model = nn.Sequential(*modules)
        return


    def prepare_embeds(
        self, model, input_ids, position_ids, attention_mask, past_key_values, labels, table_imbeds
    ):
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            #print(input_ids)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_embeds = model.model.get_input_embeddings()(cur_input_ids)
            if table_imbeds is not None:
                cur_table_embeds = table_imbeds[batch_idx].clone()
                cur_table_embeds = self.model(cur_table_embeds) # forward through the projector
                new_input_embeds.append(torch.cat([cur_table_embeds, cur_input_embeds], dim=0))
                #new_input_embeds.append(torch.cat([cur_input_embeds, cur_table_embeds], dim=0))
            else:
                new_input_embeds.append(cur_input_embeds)
            # new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_labels = labels[batch_idx]
            if table_imbeds is not None:
                cur_new_labels = torch.cat((torch.full((cur_table_embeds.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype), cur_labels))
                new_labels.append(cur_new_labels)
            else:
                new_labels.append(cur_labels)

        # # Truncate sequences to max length as image embeddings can make the sequence longer
        # tokenizer_model_max_length = getattr(model.config, 'tokenizer_model_max_length', None)
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(model.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            # new_labels = _labels

        if _attention_mask is None:
            pass # keep the newly created attention mask
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels