# Mask filling

# The next pipeline you’ll try is fill-mask. The idea of this task is to fill in the blanks in a given text:
from transformers import pipeline

unmasker = pipeline("fill-mask")
result =unmasker("This course will teach you all about <mask> models.", top_k=2)
print(result)


# Named entity recognition

# Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to entities such as persons, locations, or organizations. Let’s 
# look at an example:

from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
result=ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)


# The question-answering pipeline answers questions using information from a given context:

from transformers import pipeline

question_answerer = pipeline("question-answering")
result=question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(result)


#  Summarization

# Summarization is the task of reducing a text into a shorter text while keeping all (or most) of the important aspects referenced in the text. Here’s an example:

from transformers import pipeline

summarizer = pipeline("summarization")
result=summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
print(result)

#  Translation

# For translation, you can use a default model if you provide a language pair in the task name (such as "translation_en_to_fr"), but the easiest way is to pick the model you want to use on the Model Hub. Here we’ll try translating from French to English:

from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result=translator("أنا جيد في اللغة العربية")
print(result)


#  Image and audio pipelines

# Beyond text, Transformer models can also work with images and audio. Here are a few examples:
# Image classification

from transformers import pipeline

image_classifier = pipeline(
    task="image-classification", model="google/vit-base-patch16-224"
)
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(result)


#  Automatic speech recognition

from transformers import pipeline

transcriber = pipeline(
    task="automatic-speech-recognition", model="openai/whisper-large-v3"
)
result = transcriber(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
)
print(result)


#  Combining data from multiple sources

# One powerful application of Transformer models is their ability to combine and process data from multiple sources. This is especially useful 
# when you need to:

#     Search across multiple databases or repositories
#     Consolidate information from different formats (text, images, audio)
#     Create a unified view of related information

# For example, you could build a system that:

#     Searches for information across databases in multiple modalities like text and image.
#     Combines results from different sources into a single coherent response. For example, from an audio file and text description.
#     Presents the most relevant information from a database of documents and metadata.

# Conclusion

# The pipelines shown in this chapter are mostly for demonstrative purposes. They were programmed for specific tasks and cannot perform 
# variations of them. In the next chapter, you’ll learn what’s inside a pipeline() function and how to customize its behavior.