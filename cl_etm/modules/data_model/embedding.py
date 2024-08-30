import torch
import json
import six
import copy
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch.nn as nn
import os

# padding_type, max_length, embed_type
class BertConfig(object):
    def __init__(self, 
                 vocab_size,
                 embed_dim=768, 
                 n_segments=3, 
                 max_len=512, 
                 n_layers=12, 
                 attn_heads=12, 
                 dropout=0.1,
                 padding_type="max_length",
                 embed_type="concat"):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_segments = n_segments
        self.max_len = max_len
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.padding_type = padding_type
        self.embed_type = embed_type

    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
class BioBert:
    def __init__(self, config_file):
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file not found at {config_file}")
        
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        self.model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        self.config = BertConfig.from_json_file(config_file)
        self.attention_aggregator = AttentionAggregator(embed_dim=self.config.embed_dim)

    def embed(self, features_dict):
        last_hidden_states = []
        if self.config.embed_type == "concat":
            text = self.create_text(features_dict)
            inputs = self.tokenizer(text, return_tensors="pt", padding=self.config.padding_type, truncation=True, max_length=self.config.max_len)
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        elif self.config.embed_type == "attention":
            for key, value in features_dict.items():
                inputs = self.tokenizer(f"{key}: {value}", return_tensors="pt", padding=self.config.padding_type, truncation=True, max_length=self.config.max_len)
                outputs = self.model(**inputs)
                last_hidden_states.append(outputs.last_hidden_state)
        
            if len(last_hidden_states) != 0:
                last_hidden_states = torch.cat(last_hidden_states, dim=0)
            else:
                inputs = self.tokenizer("", return_tensors="pt", padding=self.config.padding_type, truncation=True, max_length=self.config.max_len)
                last_hidden_states = self.model(**inputs).last_hidden_state

            # last_hidden_states = last_hidden_states.mean(dim=0, keepdim=True) 
            last_hidden_states = self.attention_aggregator(last_hidden_states)
            
        return last_hidden_states
    
    def create_text(self, features_dict):
        key = list(features_dict.keys())[0]
        features_dict = features_dict[key]
        text = ""
        if key == "inputevents":
            text = (
                f"The patient with a weight of {features_dict['patientweight']} kg received an amount of {features_dict['amount']} units "
                f"under the order category '{features_dict['ordercategoryname']}', totaling {features_dict['totalamount']} units in the end "
                f"at the time of {features_dict['time']}."
            )
        elif key == "proceduresevents":
            text = (
                f"During the procedure categorized as '{features_dict['ordercategoryname']}', the patient with a weight of {features_dict['patientweight']} kg "
                f"received an original amount of {features_dict['originalamount']} units, with a recorded value of {features_dict['value']} "
                f"at the time of {features_dict['time']}."
            )
        elif key == "microbiologyevents":
            text = (
                f"The microbiology test '{features_dict['test_name']}' was conducted, with the following comment recorded: '{features_dict['comment']}' "
                f"at the time of {features_dict['time']}."
            )
        elif key == "labevents":
            text = (
                f"The test result shows a value of {features_dict['value']} valueuom in {features_dict['valueuom']}, with the comment indicating '{features_dict['comments']}' " 
                f"at the time of {features_dict['time']}."
            )
        elif key == "procedure_icd":
            text = (
                f"The procedure was documented using ICD code '{features_dict['icd_code']}', version '{features_dict['icd_version']}' "
                f"at the time of {features_dict['time']}."
            )
        else: 
            text = f"An event occurred at the time of {features_dict['time']}."
        return text
    
class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionAggregator, self).__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=3, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        aggregated_embedding = attn_output.mean(dim=0)
        final_embedding = self.fc(aggregated_embedding)
        return final_embedding.unsqueeze(0)