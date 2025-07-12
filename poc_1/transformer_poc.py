from transformers import pipeline

# 1. Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis")
sentiment = sentiment_pipeline("I love Hugging Face! It's so easy to use.")
print("\n🧠 Sentiment Analysis:", sentiment)

# 2. Text Generation
text_gen_pipeline = pipeline("text-generation", model="gpt2")
generated_text = text_gen_pipeline("Once upon a time in a world ruled by AI,", max_length=30, num_return_sequences=1)
print("\n📝 Text Generation:", generated_text[0]['generated_text'])

# 3. summarizer
summarizer_pipeline = pipeline("summarization")
long_text = (
    "Hugging Face is an AI company that has created the Transformers library. "
    "This library provides thousands of pre-trained models for natural language processing tasks, "
    "such as sentiment analysis, translation, text generation, summarization, and more."
)
summary = summarizer_pipeline(long_text, max_length=40, min_length=10, do_sample=False)
print("\n📚 Summarization:", summary[0]['summary_text'])

