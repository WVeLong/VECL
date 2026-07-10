from ..models.transformer_decoder import *


class Fusion_Model(nn.Module):
    def __init__(self, cfg = None):
        super().__init__()
        embed_dim = cfg.model.fusion.d_model
        class_num = 3
        decoder_number_layer = cfg.model.fusion.decoder_number_layer

        self.d_model = embed_dim
        decoder_layer = TransformerDecoderWoSelfAttenLayer(self.d_model, 4, 1024, 0.1, 'relu',normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, decoder_number_layer, self.decoder_norm, return_intermediate=False)
        self.dropout_feas = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(  # nn.LayerNorm(768),
                nn.Linear(embed_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, class_num)
            )
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, image_features, text_features, pos=None, use_MLP=True, return_atten=False):
        batch_size = image_features.shape[0]
        image_features = image_features.transpose(0, 1)  # (patch_num,batch_size,dim)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)  # (query_num,batch_size,dim)
        image_features = self.decoder_norm(image_features)
        text_features = self.decoder_norm(text_features)
        features, atten_map = self.decoder(text_features, image_features, memory_key_padding_mask=None, pos=pos, query_pos=None)
        features = self.dropout_feas(features).transpose(0, 1)  # b,embed_dim
        if return_atten:
            if use_MLP == False:
                return features, atten_map
            else:
                out = self.mlp_head(features)
                return out, atten_map
        else:
            if use_MLP == False:
                return features
            out = self.mlp_head(features)
            return out