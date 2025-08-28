# Text generation

# Now let’s see how to use a pipeline to generate some text. The main idea here is that you provide a prompt and the model will auto-complete
# it by generating the remaining text. This is similar to the predictive text feature that is found on many phones. Text generation involves 
# randomness, so it’s normal if you don’t get the same results as shown below.

from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M") # model passed here or not its gonna work but default models will be used and not sure if we get accurate resluts 
result=generator("In this course, we will teach you how to")
print(result)

# The previous examples used the default model for the task at hand, but you can also choose a particular model from the Hub to use in a 
# pipeline for a specific task — say, text generation. Go to the Model Hub and click on the corresponding tag on the left to display only 
# the supported models for that task. You should get to a page like this one.

# Let’s try the HuggingFaceTB/SmolLM2-360M model! Here’s how to load it in the same pipeline as before:

from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
result=generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(result)
