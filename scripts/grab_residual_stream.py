import nnsight
import torch
from nnsight import LanguageModel

templates = {
        "sentiment.1": "That movie was {filler}",
        "sentiment.2": "The film felt {filler}",
        "sentiment.3": "I thought the movie was {filler}",
        "sentiment.4": "The movie turned out {filler}",
        "sentiment.5": "Overall, the film was {filler}",
        "sentiment.6": "Watching it was {filler}",
        "sentiment.7": "The movie came across as {filler}",
        "sentiment.8": "The film ended up being {filler}",
        "sentiment.9": "The movie was truly {filler}",
        "sentiment.10": "In the end, the film was {filler}",
}

sentiment_words = {
        "sentiment.1": ["abysmal", "atrocious", "awful"],
        "sentiment.2": ["terrible", "horrible", "dreadful"],
        "sentiment.3": ["bad", "poor", "weak"],
        "sentiment.4": ["mediocre", "meh", "uninspired"],
        "sentiment.5": ["okay", "fine", "average"],
        "sentiment.6": ["decent", "solid", "pleasant"],
        "sentiment.7": ["good", "enjoyable", "satisfying"],
        "sentiment.8": ["great", "impressive", "strong"],
        "sentiment.9": ["excellent", "fantastic", "outstanding"],
        "sentiment.10": ["masterpiece", "phenomenal", "spectacular"],
}

MODEL_NAME = "gpt2"
LLM = LanguageModel(MODEL_NAME, device_map = "auto")

print(f"LLM: {MODEL_NAME} model tree:\n{LLM}")

def get_sentiment_sentence(
        template_id:str,
        sentiment_level_id:str,
        sentiment_word_id:str
        ):

    # Assertions
    template_num = int(template_id)
    assert 1 <= template_num <= len(templates)
    complete_template_id = f"sentiment.{template_num}"
    assert complete_template_id in templates

    level_num = int(sentiment_level_id)
    assert 1 <= level_num <= len(sentiment_words)
    complete_sentiment_level_id = f"sentiment.{level_num}"
    assert complete_sentiment_level_id in sentiment_words

    word_num = int(sentiment_word_id)
    assert 1 <= word_num <= 3
    complete_word_num = word_num - 1 # For 0 based indexing

    # Get the template that we want to fill
    template =  templates[complete_template_id]

    # Get the sentiment word that we wanna add to it
    word = sentiment_words[complete_sentiment_level_id][complete_word_num]

    # Fill the template
    filled_template = template.format(filler = word)

    return filled_template

def get_last_residual_stream(prompt):

    n = 3
    model = LLM
    tokenizer = model.tokenizer
    tokenized_input = tokenizer(prompt,
                                return_tensors = "pt",
                                add_special_tokens = False)
    input_ids = tokenized_input["input_ids"][0]

    last_n_ids = input_ids[-n:].tolist()
    last_untokenized_tokens = [tokenizer.decode(id) for id in last_n_ids]
    
    print(f"Last token ids: {last_n_ids}")
    print(f"Last decoded tokens: {last_untokenized_tokens}")

    with model.trace(tokenized_input):

        resid_last = model.transformer.h[-1].output[0].save()

    return resid_last

def main():

    prompt = get_sentiment_sentence(1, 1, 1) 
    res_stream = get_last_residual_stream(prompt)

    print(res_stream)

    print("Done.")

if __name__ == "__main__":
    main()
