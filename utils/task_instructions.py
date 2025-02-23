#%%writefile task_instructions.py
#@title Task Instructions
def get_sentiment_instruction(language):
    # Define language-specific sentiment labels
    labels = {
        "English": {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral"
        },
        "swahili": {
            "positive": "Chanya",
            "negative": "Hasi",
            "neutral": "Wastani"
        },
        "hausa": {
            "positive": "Kyakkyawa",
            "negative": "Korau",
            "neutral": "Tsaka-tsaki"
        }
    }

    sentiment_instruction = f"""Analyze the sentiment of the following text. Your task is to classify the text into one of three sentiment categories based on the following criteria:

{labels[language]["positive"]}:
- The text conveys positive emotions such as happiness, approval, or satisfaction.
- It contains praise, appreciation, or optimistic language.
- The overall tone is upbeat, enthusiastic, or celebratory.
- Positive adjectives or affirmations are used.

{labels[language]["negative"]}:
- The text expresses negative emotions such as anger, dissatisfaction, or disapproval.
- It contains complaints, criticism, or unfavorable language.
- The overall tone is critical, pessimistic, or hostile.
- Negative adjectives or accusations are used.

{labels[language]["neutral"]}:
- The text is impartial, factual, and does not show strong emotions.
- It provides objective information, descriptions, or events without judgment.
- The tone is balanced, matter-of-fact, or descriptive.
- No emotional language or opinions are present.

Your response should be **exactly one label**: {labels[language]["positive"]}, {labels[language]["negative"]}, or {labels[language]["neutral"]}.
"""

    return sentiment_instruction


def get_translation_instruction(language):
  translation_instruction = """Translate the following English text to {language}. Important guidelines:
    - Maintain the original meaning and tone
    - Keep proper names unchanged
    - Preserve any formatting or punctuation
    - For idiomatic expressions, use culturally appropriate equivalents
    - For formal text, use formal language in the target language
    - For informal text, use casual language in the target language

    Provide only the translation in {language}, without explanations or notes."""

  return translation_instruction.format(language=language)

def get_xnli_instruction(language):
    xnli_instruction = """Given a premise and hypothesis in {language}, determine if the hypothesis can be logically inferred from the premise.

    Even though the text is in {language}, always return exactly one label in English:
    True - if the hypothesis logically and definitely follows from the premise
    False - if the hypothesis contradicts or is incompatible with the premise
    Neither - if the truth of the hypothesis cannot be determined from the premise alone

    Guidelines:
    - Consider only the information provided in the premise
    - Avoid making assumptions beyond what's explicitly stated
    - Cultural and contextual meanings in {language} should be considered
    - Focus on logical relationship, not just topic similarity

    Answer with only one English word: True/False/Neither"""

    return xnli_instruction.format(language=language.title())
