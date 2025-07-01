# from vinorm import TTSnorm

import re
import unicodedata
from underthesea import sent_tokenize

def normalize_vietnamese_text(text):
    # Basic unicode normalization
    text = (
        unicodedata.normalize("NFC", text)
        .replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("AI", "ây ai")
        .lower()
    )
    return text

def calculate_keep_len(text, lang):
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = (
        text.count(".")
        + text.count("!")
        + text.count("?")
        + text.count(",")
    )

    if word_count < 5:
        return 8000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 8000 * word_count + 2000 * num_punct
    return -1

if __name__ == "__main__": 
    text = "Xin chào, tôi đang thử hệ thống chuyển văn bản thành giọng nói AI."
    # text = "Hàm này được phát triển từ 8/2019. Có phải tháng 12/2020 đã có vaccine phòng ngừa Covid-19 xmz ?"

    new_text = normalize_vietnamese_text(text)
    print(new_text)
    from underthesea import sent_tokenize

    token = sent_tokenize(new_text)

    print(token)
 