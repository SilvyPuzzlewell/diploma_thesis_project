from transformers import pipeline

zh_en_translator = pipeline("translation_zh_to_en", model='Helsinki-NLP/opus-mt-zh-en')

def helsinky_translation(zh_txt):
    return zh_en_translator(zh_txt)[0]['translation_text']

"""
def bart_translation():
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

    article_zh = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
    article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


    tokenizer.src_lang = "zh_CN"
    encoded_zh = tokenizer(test_txt, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_zh,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    # => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."
"""
