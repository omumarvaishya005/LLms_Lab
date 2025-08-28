# Creating a Transformer
# Let’s begin by examining what happens when we instantiate an AutoModel:

from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")

# Similar to the tokenizer, the from_pretrained() method will download and cache the model data from the Hugging Face Hub. 
# As mentioned previously, the checkpoint name corresponds to a specific model architecture and weights, in this case a BERT model 
# with a basic architecture (12 layers, 768 hidden size, 12 attention heads) and cased inputs (meaning that the uppercase/lowercase 
# distinction is important). There are many checkpoints available on the Hub — you can explore them here.

# The AutoModel class and its associates are actually simple wrappers designed to fetch the appropriate model architecture for a given checkpoint. 
# It’s an “auto” class meaning it will guess the appropriate model architecture for you and instantiate the correct model class. However, 
# if you know the type of model you want to use, you can use the class that defines its architecture directly:

from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")

# Loading and saving
# Saving a model is as simple as saving a tokenizer. In fact, the models actually have the same save_pretrained() method, which saves the model’s 
# weights and architecture configuration:

model.save_pretrained("directory_on_my_computer")

# This will save two files to your disk:
# ls directory_on_my_computer
# config.json pytorch_model.bin

# If you look inside the config.json file, you’ll see all the necessary attributes needed to build the model architecture. 
# This file also contains some metadata, such as where the checkpoint originated and what 🤗 Transformers version you were using when you last 
# saved the checkpoint.
# The pytorch_model.bin file is known as the state dictionary; it contains all your model’s weights. The two files work together: 
# the configuration file is needed to know about the model architecture, while the model weights are the parameters of the model.
# To reuse a saved model, use the from_pretrained() method again:

from transformers import AutoModel

model = AutoModel.from_pretrained("directory_on_my_computer")

# A wonderful feature of the 🤗 Transformers library is the ability to easily share models and tokenizers with the community. To do this, 
# make sure you have an account on Hugging Face. If you’re using a notebook, you can easily log in with this:

# from huggingface_hub import notebook_login
# notebook_login()

# Otherwise, at your terminal run:
# huggingface-cli login
# Then you can push the model to the Hub with the push_to_hub() method:

model.push_to_hub("my-awesome-model")

# This will upload the model files to the Hub, in a repository under your namespace named my-awesome-model. Then, anyone can load your 
# model with the from_pretrained() method!

from transformers import AutoModel

model = AutoModel.from_pretrained("omvaishya/my-awesome-model")

# You can do a lot more with the Hub API:
#     Push a model from a local repository
#     Update specific files without re-uploading everything
#     Add model cards to document the model’s abilities, limitations, known biases, etc.
# See the documentation for a complete tutorial on this, or check out the advanced Chapter 4.
# Encoding text

# Transformer models handle text by turning the inputs into numbers. Here we will look at exactly what happens when your text is processed by 
# the tokenizer. We’ve already seen in Chapter 1 that tokenizers split the text into tokens and then convert these tokens into numbers. We can see 
# this conversion through a simple tokenizer:

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)

{'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

# We get a dictionary with the following fields:

#     input_ids: numerical representations of your tokens
#     token_type_ids: these tell the model which part of the input is sentence A and which is sentence B (discussed more in the next section)
#     attention_mask: this indicates which tokens should be attended to and which should not (discussed more in a bit)

# We can decode the input IDs to get back the original text:

tokenizer.decode(encoded_input["input_ids"])

# "[CLS] Hello, I'm a single sentence! [SEP]"

# You’ll notice that the tokenizer has added special tokens — [CLS] and [SEP] — required by the model. Not all models need special tokens; 
# they’re utilized when a model was pretrained with them, in which case the tokenizer needs to add them as that model expects these tokens.
# You can encode multiple sentences at once, either by batching them together (we’ll discuss this soon) or by passing a list:

encoded_input = tokenizer("How are you?", "I'm fine, thank you!")
print(encoded_input)

{'input_ids': [[101, 1731, 1132, 1128, 136, 102], [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

# Note that when passing multiple sentences, the tokenizer returns a list for each sentence for each dictionary value. We can also ask the tokenizer 
# to return tensors directly from PyTorch:

encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(encoded_input)

# {'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102],
#          [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]), 
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

# But there’s a problem: the two lists don’t have the same length! Arrays and tensors need to be rectangular, so we can’t simply convert these 
# lists to a PyTorch tensor (or NumPy array). The tokenizer provides an option for that: padding.
# Padding inputs

# If we ask the tokenizer to pad the inputs, it will make all sentences the same length by adding a special padding token to the sentences 
# that are shorter than the longest one:

encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)

# {'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102,     0,     0,     0,     0],
#          [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]), 
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

# Now we have rectangular tensors! Note that the padding tokens have been encoded into input IDs with ID 0, and they have an attention mask value of 0 as well. This is because those padding tokens shouldn’t be analyzed by the model: they’re not part of the actual sentence.
# Truncating inputs

# The tensors might get too big to be processed by the model. For instance, BERT was only pretrained with sequences up to 512 tokens, so it cannot process longer sequences. If you have sequences longer than the model can handle, you’ll need to truncate them with the truncation parameter:

encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])

[101, 1188, 1110, 170, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1179, 5650, 119, 102]

# By combining the padding and truncation arguments, you can make sure your tensors have the exact size you need:

encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)

# {'input_ids': tensor([[  101,  1731,  1132,  1128,   102],
#          [  101,  1045,  1005,  1049,   102]]), 
#  'token_type_ids': tensor([[0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0]]), 
#  'attention_mask': tensor([[1, 1, 1, 1, 1],
#          [1, 1, 1, 1, 1]])}

# Adding special tokens

# Special tokens (or at least the concept of them) is particularly important to BERT and derived models. These tokens are added to better represent the sentence boundaries, such as the beginning of a sentence ([CLS]) or separator between sentences ([SEP]). Let’s look at a simple example:

encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
tokenizer.decode(encoded_input["input_ids"])

# [101, 1731, 1132, 1128, 136, 102]
# '[CLS] How are you? [SEP]'

# These special tokens are automatically added by the tokenizer. Not all models need special tokens; they are primarily used when a model was pretrained with them, in which case the tokenizer will add them since the model expects them.
# Why is all of this necessary?

# Here’s a concrete example. Consider these encoded sequences:

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

# Once tokenized, we have:

encoded_sequences = [
    [
        101,
        1045,
        1005,
        2310,
        2042,
        3403,
        2005,
        1037,
        17662,
        12172,
        2607,
        2026,
        2878,
        2166,
        1012,
        102,
    ],
    [101, 1045, 5223, 2023, 2061, 2172, 999, 102],
]

# This is a list of encoded sequences: a list of lists. Tensors only accept rectangular shapes (think matrices). This “array” is already of rectangular shape, so converting it to a tensor is easy:

import torch

model_inputs = torch.tensor(encoded_sequences)

# Using the tensors as inputs to the model
# Making use of the tensors with the model is extremely simple — we just call the model with the inputs:

output = model(model_inputs)
# While the model accepts a lot of different arguments, only the input IDs are necessary. We’ll explain what the other arguments do and when they are required later, but first we need to take a closer look at the tokenizers that build the inputs that a Transformer model can understand.