from collections import OrderedDict
from typing import Any, Callable, List, Tuple, Union, Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
import torch.nn.functional as F
from functools import partial

from bcos_lm.modules.common import DetachableModule
from bcos_lm.common import BcosUtilMixin
from bcos_lm.modules.bcoslinear import BcosLinear
from bcos_lm.modules import norms

from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForSequenceClassification, AutoConfig

__all__ = [
    "bert",
]
# helpers

class BcosOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    def __init__(self, loss=None, logits=None, hidden_states=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states


def exists(x: Any) -> bool:
    return x is not None


def pair(t: Any) -> Tuple[Any, Any]:
    return t if isinstance(t, tuple) else (t, t)


class NormalizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super(NormalizedEmbedding, self).__init__(num_embeddings, embedding_dim, **kwargs)

    def forward(self, input):
        # Retrieve embeddings
        embedding = super(NormalizedEmbedding, self).forward(input)
        # Normalize embeddings along the feature dimension
        normalized_embedding = F.normalize(embedding, p=2, dim=-1)

        return normalized_embedding

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        if config.bcos and config.b!= 1:
            self.linear = partial(BcosLinear, b=config.b) #
            self.norm = norms.NoBias(norms.DetachableLayerNorm)
            self.embedding = NormalizedEmbedding
            self.activation = ACT2FN[config.hidden_act]   
        else:
            self.linear = nn.Linear
            self.norm = nn.LayerNorm
            self.embedding = nn.Embedding
            self.activation = ACT2FN[config.hidden_act]
        self.word_embeddings = self.embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = self.embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = self.embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = self.norm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=self.position_embeddings.weight.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.token_type_embeddings.weight.device)

        
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    
class FeedForward(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        if config.bcos and config.b != 1:
            norm_layer = norms.NoBias(norms.DetachableLayerNorm)
            linear_layer = partial(BcosLinear, b=config.b)
            act_layer = ACT2FN[config.hidden_act]   
        else:
            norm_layer = nn.LayerNorm
            linear_layer = nn.Linear
            act_layer = ACT2FN[config.hidden_act]


        self.net = nn.Sequential(
            OrderedDict(                
                linear1=linear_layer(config.hidden_size, config.intermediate_size),
                act=act_layer,
                linear2=linear_layer(config.intermediate_size, config.hidden_size),
            )
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = norm_layer(config.hidden_size)
        # up-projection -> activation -> down-projection

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        out = self.dropout(out)
        out = self.norm(x + out)
        return out

class Attention(DetachableModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        if config.bcos and config.b != 1:
            norm_layer = norms.NoBias(norms.DetachableLayerNorm)
            linear_layer = partial(BcosLinear, b=config.b)
        else:
            norm_layer = nn.LayerNorm
            linear_layer = nn.Linear

        n_lins = 3
        self.heads = config.num_attention_heads
        dim_head = config.hidden_size // config.num_attention_heads
        self.scale = dim_head**-0.5
        #self.norm = norm_layer(dim)
        self.pos_info = None
        self.attention_biases = None

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(config.hidden_size, config.hidden_size * n_lins, bias=False)
        self.to_out = linear_layer(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob if config.attention_probs_dropout_prob is not None else config.hidden_dropout_prob)
        self.norm = norm_layer(config.hidden_size)

    def forward(self, 
                x: Tensor,
                attention_mask: Optional[torch.FloatTensor] = None) -> Tensor:
        #x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        #q, k, v = self.query(x), self.key(x), self.value(x)

        if self.detach:  # detach dynamic linear weights
            q = q.detach()
            k = k.detach()
            # these are used for dynamic linear w (`attn` below)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None:
            mask = attention_mask.clone().expand(dots.shape)
            dots.masked_fill_(mask==0, float("-inf"))
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        out = self.dropout(out)
        out = self.norm(x + out)
 
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.attn = Attention(config)
        self.ff = FeedForward(config)

    def forward(self, 
                x: Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                ) -> Tensor:
        x = self.attn(x, attention_mask)
        x = self.ff(x)

        return x
    
class Transformer(nn.Sequential):
    def __init__(
        self,
        config,
    ):

        layers_odict = OrderedDict()
        for i in range(config.num_hidden_layers):
            layers_odict[f"encoder_{i}"] = Encoder(config)
        super().__init__(layers_odict)
    
    def forward(self, 
                x: Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                ) -> Tensor:
        for layer in self:
            x = layer(x, attention_mask)
        return x


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.bcos and config.b!= 1:
            self.linear = partial(BcosLinear, b=config.b)
            self.activation = nn.Identity()
        else:
            self.linear = nn.Linear
            self.activation = nn.Tanh()
        self.dense = self.linear(config.hidden_size, config.hidden_size)
        self.activation = self.activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    
#############################################################################################################################

class BertModel(BcosUtilMixin, BertPreTrainedModel):
    
    _no_split_modules = ["BertEmbeddings", "BertLayer"]
    def __init__(self, 
                 config,
                 add_pooling_layer=True):
        super().__init__(config)

        #assert exists(linear_layer), "Provide a linear layer class!"
        #assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        #assert exists(act_layer), "Provide a activation layer class!"



        if config.bcos and config.b != 1:
            self.linear = partial(BcosLinear, b=config.b) #
            self.norm = norms.NoBias(norms.DetachableLayerNorm)
            self.embedding = NormalizedEmbedding
            self.activation = ACT2FN[config.hidden_act]   
        else:
            self.linear = nn.Linear
            self.norm = nn.LayerNorm
            self.embedding = nn.Embedding
            self.activation = ACT2FN[config.hidden_act]

        self.embeddings = BertEmbeddings(config)

        self.encoder = Transformer(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.post_init()

    
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        ):

        # at least one of input_ids or input_embeds should be provided
        assert input_ids is not None or input_embeds is not None, "At least one of input_ids or input_embeds should be provided"
        
        if input_ids is not None:
            input_shape = input_ids.size()
            bsz, seq_len = input_shape
            device = input_ids.device
        else:
            input_shape = input_embeds.shape[:-1]
            bsz, seq_len = input_shape
            device = input_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len), device=device)
        
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

        if input_embeds is not None:
            embedding_output = input_embeds
        else:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )

        """
        padding_mask = torch.eq(input_ids, self.pad_token_id)
        if not padding_mask.any():
            padding_mask = None
        """

        x = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(x) if self.pooler is not None else None
        
        return pooled_output

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def from_pretrained_conventional_model(cls, model_name_or_path):
        pass

    @classmethod
    def from_pretrained_bcos_model(cls, model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls.from_pretrained(model_name_or_path, config=config, **kwargs)


    
#############################################################################################################################

class BertForSequenceClassification(BcosUtilMixin, BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        if config.bcos and config.b != 1:
            self.linear = partial(BcosLinear, b=config.b) #
            self.norm = norms.NoBias(norms.DetachableLayerNorm)
            self.embedding = NormalizedEmbedding
            self.activation = ACT2FN[config.hidden_act]   
        else:
            self.linear = nn.Linear
            self.norm = nn.LayerNorm
            self.embedding = nn.Embedding
            self.activation = ACT2FN[config.hidden_act]

        self.bert = BertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = self.linear(config.hidden_size, config.num_labels)

        self.post_init()
    
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ):

        x = self.bert(input_ids, attention_mask, token_type_ids, position_ids, input_embeds) # after pooling layer
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        sigmoid = nn.Sigmoid()
        if labels is not None:
            loss_fct = nn.BCELoss()
            # convert labels to one-hot encoding, float
            targets = torch.nn.functional.one_hot(labels, num_classes=self.config.num_labels)
            targets = targets.float().to(logits.device)
            targets.requires_grad = False
            loss = loss_fct(sigmoid(logits), targets)
        return BcosOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    @classmethod
    def from_pretrained_conventional_model(cls, model_name_or_path, config):
        print(f"Converting conventional model {model_name_or_path} to Bcos model")
        conventional_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        bcos_model = cls(config)
        conventional_model_params = dict(conventional_model.named_parameters())
        bcos_model_params = dict(bcos_model.named_parameters())

        print("Converting embedding layers")
        bcos_model_params["bert.embeddings.word_embeddings.weight"].data.copy_(conventional_model_params["bert.embeddings.word_embeddings.weight"].data)
        bcos_model_params["bert.embeddings.position_embeddings.weight"].data.copy_(conventional_model_params["bert.embeddings.position_embeddings.weight"].data)
        bcos_model_params["bert.embeddings.token_type_embeddings.weight"].data.copy_(conventional_model_params["bert.embeddings.token_type_embeddings.weight"].data)
        bcos_model_params["bert.embeddings.LayerNorm.weight"].data.copy_(conventional_model_params["bert.embeddings.LayerNorm.weight"].data)
        
        for layer_i in range(config.num_hidden_layers):
            print(f"Converting layer {layer_i}")
            bcos_model_params[f"bert.encoder.encoder_{layer_i}.attn.to_qkv.weight"].data = torch.cat((conventional_model_params[f"bert.encoder.layer.{layer_i}.attention.self.query.weight"].clone(), conventional_model_params[f"bert.encoder.layer.{layer_i}.attention.self.key.weight"].clone(), conventional_model_params[f"bert.encoder.layer.{layer_i}.attention.self.value.weight"].clone()), dim=0)
            bcos_model_params[f"bert.encoder.encoder_{layer_i}.attn.to_out.linear.weight"].data.copy_(conventional_model_params[f"bert.encoder.layer.{layer_i}.attention.output.dense.weight"].data)
            bcos_model_params[f"bert.encoder.encoder_{layer_i}.attn.norm.weight"].data.copy_(conventional_model_params[f"bert.encoder.layer.{layer_i}.attention.output.LayerNorm.weight"].data)
            bcos_model_params[f"bert.encoder.encoder_{layer_i}.ff.net.linear1.linear.weight"].data.copy_(conventional_model_params[f"bert.encoder.layer.{layer_i}.intermediate.dense.weight"].data)
            bcos_model_params[f"bert.encoder.encoder_{layer_i}.ff.net.linear2.linear.weight"].data.copy_(conventional_model_params[f"bert.encoder.layer.{layer_i}.output.dense.weight"].data)
            bcos_model_params[f"bert.encoder.encoder_{layer_i}.ff.norm.weight"].data.copy_(conventional_model_params[f"bert.encoder.layer.{layer_i}.output.LayerNorm.weight"].data)
        print("Converting pooler layer")
        bcos_model_params["bert.pooler.dense.linear.weight"].data.copy_(conventional_model_params["bert.pooler.dense.weight"].data)
        print("Converting classifier layer")
        bcos_model_params["classifier.linear.weight"].data.copy_(conventional_model_params["classifier.weight"].data)

        print("Conversion complete")
        return bcos_model

    
#############################################################################################################################


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.bcos and config.b!= 1:
            self.linear = partial(BcosLinear, b=config.b) #
            self.norm = norms.NoBias(norms.DetachableLayerNorm)
            self.activation = ACT2FN[config.hidden_act]   
        else:
            self.linear = nn.Linear
            self.norm = nn.LayerNorm
            self.activation = ACT2FN[config.hidden_act]
        self.dense = self.linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = self.activation
        self.LayerNorm = self.norm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        if config.bcos and config.b!= 1:
            self.decoder = partial(BcosLinear, b=config.b)
            self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        else:
            self.decoder = nn.Linear
            self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = self.linear(config.hidden_size, config.vocab_size, bias=False)

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.bcos and config.b!= 1:
            self.linear = partial(BcosLinear, b=config.b)
        else:
            self.linear = nn.Linear
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = self.linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
    

class BertForPreTrainingOutput:

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertForPreTraining(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, *optional*, defaults to `{}`):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



