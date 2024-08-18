import numpy as np
import torch
import torchaudio

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷", }


class SenceVoice():
    def __init__(self, model, fs=16000, language='auto'):
        self.fs = fs
        self.language = self.set_language(language)
        self.model = model

    def set_fs(self,fs):
        self.fs = fs

    def set_language(self, language):
        language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
                         "nospeech": "nospeech"}

        # task = "Speech Recognition" if task is None else task
        language = "auto" if len(language) < 1 else language
        selected_language = language_abbr[language]
        return selected_language

    def model_inference(self, input_wav):
        language = self.language

        text = self.model.generate(
            input=input_wav,
            cache={},
            language=language,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            # merge_vad=True,
            # merge_length_s=15,
        )

        text = text[0]["text"]
        text = self.format_str_v3(text)

        return text

    def format_str(self, s):
        for sptk in emoji_dict:
            s = s.replace(sptk, emoji_dict[sptk])
        return s

    def format_str_v2(self, s):
        sptk_dict = {}
        for sptk in emoji_dict:
            sptk_dict[sptk] = s.count(sptk)
            s = s.replace(sptk, "")
        emo = "<|NEUTRAL|>"
        for e in emo_dict:
            if sptk_dict[e] > sptk_dict[emo]:
                emo = e
        for e in event_dict:
            if sptk_dict[e] > 0:
                s = event_dict[e] + s
        s = s + emo_dict[emo]

        for emoji in emo_set.union(event_set):
            s = s.replace(" " + emoji, emoji)
            s = s.replace(emoji + " ", emoji)
        return s.strip()

    def format_str_v3(self, s):
        def get_emo(s):
            return s[-1] if s[-1] in emo_set else None

        def get_event(s):
            return s[0] if s[0] in event_set else None

        s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
        for lang in lang_dict:
            s = s.replace(lang, "<|lang|>")
        s_list = [self.format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
        new_s = " " + s_list[0]
        cur_ent_event = get_event(new_s)
        for i in range(1, len(s_list)):
            if len(s_list[i]) == 0:
                continue
            if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
                s_list[i] = s_list[i][1:]
            # else:
            cur_ent_event = get_event(s_list[i])
            if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
                new_s = new_s[:-1]
            new_s += s_list[i].strip().lstrip()
        new_s = new_s.replace("The.", " ")
        return new_s.strip()
