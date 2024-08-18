import streamlit as st
from funasr import AutoModel
from modelscope.pipelines import pipeline
from sencevoice import SenceVoice
from speaker_diarization import SpeakerDiarization
import pandas as pd
from openai import OpenAI

# 设置页面布局为宽屏模式
st.set_page_config(layout="wide")
client = OpenAI(api_key="sk-264bcea8497949f1b6d82365732f6173", base_url="https://api.deepseek.com")


@st.cache_resource
def load_models():
    model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
    sv_pipeline = pipeline(
        task='speaker-verification'
        # , model="damo/speech_eres2net_sv_zh-cn_16k-common"
        , model="damo/speech_eres2netv2_sv_zh-cn_16k-common"
        , disable_update=True)
    change_locator_pipeline = pipeline(
        task='speaker-diarization',
        model='damo/speech_xvector_transformer_scl_zh-cn_16k-common', disable_update=True)
    sv_model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=False,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
        disable_update=True
    )

    sencevioce = SenceVoice(sv_model)

    sd = SpeakerDiarization(vad_model=model,
                            sv_pipeline=sv_pipeline,
                            change_locator_pipeline=change_locator_pipeline,
                            sv_model=sencevioce
                            )
    return sd


sd = load_models()

# 初始化会话状态
if 'audio_df' not in st.session_state:
    st.session_state.audio_df = None
if 'speaker_mapping' not in st.session_state:
    st.session_state.speaker_mapping = {}

# Streamlit 应用
st.title("音频解析应用")


def format_time(time):
    if time < 60:
        time_str = f"{int(time)} s"
    else:
        minutes = int(time // 60)
        seconds = int(time % 60)
        time_str = f"{minutes} m {seconds} s"
    return time_str


with st.sidebar:
    st.subheader("音频文件解析")
    uploaded_file = st.file_uploader("上传文件", type=["mp3", "wav", "flac", "mp4"])
    language = st.selectbox("选择语言", options=['en', 'zh', 'auto'], index=1)
    speaker_num = st.number_input("对话人数", min_value=1, max_value=10, value=1)
    interval_num = st.number_input("按时间间隔合并短音频，为0不合并", min_value=0, max_value=10, value=2)
    if st.button("解析"):
        if uploaded_file is not None:
            st.session_state.audio_df = sd.discern(uploaded_file, language, speaker_num=speaker_num,
                                                   merge_chunk=interval_num)
            st.session_state.speaker_mapping = {}  # 重置说话人映射
            st.success("文件解析完成！")
        else:
            st.error("请先上传文件！")

    # 对话内容:
    st.subheader("对音频内容进行提问")

    if question := st.chat_input():
        # question = st.input("关于音频的问题")

        if st.session_state.audio_df is not None and len(question) > 1:
            conversation = "\n".join([
                                         f"{format_time(row['开始时间'])} ~ {format_time(row['结束时间'])} | {st.session_state.speaker_mapping.get(row['说话人'], row['说话人'])}: {row['音频文本内容']}"
                                         for index, row in st.session_state.audio_df.iterrows()])
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": "下面是一段音频转文本后的内容，你需要根据这段内容回答用户提出来的问题\n====\n" + conversation},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                stream=False
            )

            background = response.choices[0].message.content
            st.write("对话结果：")
            st.write(background)
        else:
            st.error("请先解析音频文件！")

if st.session_state.audio_df is not None and len(st.session_state.audio_df) > 0:
    # 显示说话人映射修改框
    st.subheader("修改说话人映射")
    for speaker in st.session_state.audio_df['说话人'].unique():
        if speaker not in st.session_state.speaker_mapping:
            st.session_state.speaker_mapping[speaker] = speaker
        new_name = st.text_input(f"修改说话人 {speaker} 的名字", value=st.session_state.speaker_mapping[speaker])
        st.session_state.speaker_mapping[speaker] = new_name

    # 显示解析结果
    st.subheader("解析结果")
    for index, row in st.session_state.audio_df.iterrows():
        st.audio(row['文件路径'], format='audio/wav')

        start_time = row['开始时间']
        end_time = row['结束时间']
        start_time_str = format_time(start_time)
        end_time_str = format_time(start_time)

        speaker = row['说话人']
        speaker_name = st.session_state.speaker_mapping[speaker]

        st.write(f"{start_time_str} ~ {end_time_str} | 说话人 {speaker_name} : {row['音频文本内容']}")
else:
    st.info("请先上传并解析文件。")

