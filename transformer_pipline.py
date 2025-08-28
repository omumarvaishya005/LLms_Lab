from transformers import pipeline
#  This code uses the Hugging Face Transformers library to create a sentiment analysis pipeline.
# It initializes the pipeline with a pre-trained model for sentiment analysis and then uses it to analyze
# the sentiment of a given text input. The result is printed to the console.
# Make sure to install the transformers library first using pip:
# pip install transformers
classifier = pipeline("sentiment-analysis")
result=classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)

# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

# You can also analyze multiple texts at once:
result=classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(result)

# By default, this pipeline selects a particular pretrained model that has been fine-tuned for sentiment 
# analysis in English. The model is downloaded and cached when you create the classifier object. 
# If you rerun the command, the cached model will be used instead and there is no need to download the model again.

# There are three main steps involved when you pass some text to a pipeline:

#     The text is preprocessed into a format the model can understand.
#     The preprocessed inputs are passed to the model.
#     The predictions of the model are post-processed, so you can make sense of them.

# Available pipelines for different modalities

# The pipeline() function supports multiple modalities, allowing you to work with text, images, audio, and even multimodal tasks. 
# In this course we’ll focus on text tasks, but it’s useful to understand the transformer architecture’s potential, so we’ll briefly outline it.

# Here’s an overview of what’s available:

# For a full and updated list of pipelines, see the 🤗 Transformers documentation.
# Text pipelines

#     text-generation: Generate text from a prompt
#     text-classification: Classify text into predefined categories
#     summarization: Create a shorter version of a text while preserving key information
#     translation: Translate text from one language to another
#     zero-shot-classification: Classify text without prior training on specific labels
#     feature-extraction: Extract vector representations of text

# Image pipelines

#     image-to-text: Generate text descriptions of images
#     image-classification: Identify objects in an image
#     object-detection: Locate and identify objects in images

# Audio pipelines

#     automatic-speech-recognition: Convert speech to text
#     audio-classification: Classify audio into categories
#     text-to-speech: Convert text to spoken audio

# Multimodal pipelines

#     image-text-to-text: Respond to an image based on a text prompt

# Let’s explore some of these pipelines in more detail!
# Zero-shot classification

# We’ll start by tackling a more challenging task where we need to classify texts that haven’t been labelled. This is a common scenario 
# in real-world projects because annotating text is usually time-consuming and requires domain expertise. For this use case, 
# the zero-shot-classification pipeline is very powerful: it allows you to specify which labels to use for the classification, 
# so you don’t have to rely on the labels of the pretrained model. You’ve already seen how the model can classify a sentence as 
# positive or negative using those two labels — but it can also classify the text using any other set of labels you like.

from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)