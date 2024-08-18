from typing import Union

from modelscope.fileio import File
import soundfile as sf
import io
import torch
import torchaudio
import numpy as np
import ast

from cluster_backend import ClusterBackend
from loguru import logger

import os
import uuid

import pandas as pd
from pydub import AudioSegment
import hashlib


class SpeakerDiarization:
    def __init__(self
                 , vad_model
                 , sv_pipeline
                 , change_locator_pipeline
                 , sv_model
                 ):
        self.fs = 16000
        self.seg_dur = 1.5
        self.seg_shift = 0.75
        self.vad_model = vad_model
        self.sv_pipeline = sv_pipeline
        self.change_locator_pipeline = change_locator_pipeline
        self.sv_model = sv_model
        self.cluster_config = {
            "merge_thr": 0.68
        }
        self.interval = 2
        self.cluster_bakcend = ClusterBackend(self.cluster_config)

    def discern(self, wav_file, language='auto',speaker_num=1,merge_chunk=0):
        logger.info("0.正在处理wav文件")
        hex_dig,wav_path,audio = self.wav2audio(wav_file)

        # 1.vad模型识别对话端点
        logger.info("1.vad模型识别对话端点")
        vad_time = self.vad_model.generate(input=wav_path, fs=self.fs, is_final=True)[0]['value']
        vad_segments = self.get_segements(audio, vad_time)

        # 2.audio切分
        logger.info("2.audio切分")
        segments = self.chunk(vad_segments)

        # 3.audio片段转语音
        logger.info("3.audio片段转语音")
        embeddings = self.get_embedding(segments)

        # 4.聚类取标签
        logger.info("4.聚类取标签")
        labels = self.cluster_bakcend.forward(embeddings, oracle_num=speaker_num)

        # 5.结果处理
        logger.info("5.结果处理")
        output = self.postprocess(segments, vad_segments, labels, embeddings)

        # 6.短音频合并
        if merge_chunk > 0:
            logger.info("6.短音频合并")
            output = self.merge_segments(output, merge_chunk)
        else:
            pass

        # 7.切分音频成片段文件
        logger.info("7.切分音频成片段文件")
        segment_paths, sample_rate = self.cut_audio_to_files(wav_path, output, r'./output')

        # 8.将音频文件转文本
        logger.info("8.将音频文件转文本")
        self.sv_model.set_fs(sample_rate)
        self.sv_model.set_language(language)

        # 遍历output和segment_paths，填充DataFrame
        rows = []
        for i, (start_time, end_time, speaker) in enumerate(output):
            if i < len(segment_paths):
                segment_path = segment_paths[i]
                audio_text = self.sv_model.model_inference(segment_path)
                row = {
                    '开始时间': start_time,
                    '结束时间': end_time,
                    '说话人': speaker,
                    '文件路径': segment_path,
                    '音频文本内容': audio_text
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def cut_audio_to_files(self, wav_file, cut_timeline, output_dir):
        # Create a unique directory to store the audio segments
        unique_id = str(uuid.uuid4())
        segment_dir = os.path.join(output_dir, unique_id)
        os.makedirs(segment_dir, exist_ok=True)

        # Load the audio file
        waveform, sample_rate = torchaudio.load(wav_file)

        segment_paths = []
        for i, (start, end, speaker) in enumerate(cut_timeline):
            # Convert seconds to sample indices
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)

            # Cut the audio segment
            audio_segment = waveform[:, start_sample:end_sample]

            # Save the audio segment to a file
            segment_path = os.path.join(segment_dir, f"segment_{i}_{speaker}.wav")
            torchaudio.save(segment_path, audio_segment, sample_rate)
            segment_paths.append(segment_path)

        return segment_paths, sample_rate

    def wav2audio(self, uploaded_file):
        # 使用 pydub 将文件转换为 WAV 格式
        audio_segment = AudioSegment.from_file(uploaded_file)

        # 创建一个 BytesIO 对象来保存 WAV 数据
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)  # 重置指针到开始位置

        # 计算哈希值
        wav_bytes = wav_io.getvalue()
        hash_object = hashlib.sha256(wav_bytes)
        hex_dig = hash_object.hexdigest()

        # 保存转换后的 WAV 文件到 './data' 目录
        if not os.path.exists('./temp'):
            os.makedirs('./temp')

        wav_path = os.path.join('./temp', f'{hex_dig}.wav')

        # 检查文件是否已经存在
        if not os.path.exists(wav_path):
            with open(wav_path, 'wb') as f:
                f.write(wav_bytes)
        else:
            logger.info(f"文件 {wav_path} 已经存在，跳过保存。")

        # 读取上传的音频文件
        wav_io.seek(0)  # 重置指针到开始位置
        audio_segment = AudioSegment.from_file(wav_io, format="wav")

        # 获取采样率
        fs = audio_segment.frame_rate

        # 将音频数据转换为 numpy 数组
        audio = np.array(audio_segment.get_array_of_samples())

        # 如果音频是立体声（两个通道），则只取第一个通道
        if audio_segment.channels == 2:
            audio = audio.reshape((-1, 2))
            audio = audio[:, 0]

        # 将采样率转换为 16000 Hz
        sample_rate = 16000
        audio, fs = torchaudio.sox_effects.apply_effects_tensor(
            torch.from_numpy(audio).unsqueeze(0),
            fs,
            effects=[['rate', str(sample_rate)]])

        # 去掉多余的维度，并确保音频数据类型为 float32
        audio = audio.squeeze(0).numpy()
        if audio.dtype in ['int16', 'int32', 'int64']:
            audio = (audio / (1 << 15)).astype('float32')
        else:
            audio = audio.astype('float32')

        return hex_dig,wav_path, audio

    def get_segements(self, audio, vad_time):
        if isinstance(vad_time, str):
            vad_time_list = ast.literal_eval(vad_time)
        elif isinstance(vad_time, list):
            vad_time_list = vad_time
        vad_segments = []
        for t in vad_time_list:
            st = int(t[0]) / 1000
            ed = int(t[1]) / 1000
            vad_segments.append(
                [st, ed, audio[int(st * self.fs):int(ed * self.fs)]])
        return vad_segments

    def chunk(self, vad_segments: list) -> list:
        def seg_chunk(seg_data):
            seg_st = seg_data[0]
            data = seg_data[2]
            chunk_len = int(self.seg_dur * self.fs)
            chunk_shift = int(self.seg_shift * self.fs)
            last_chunk_ed = 0
            seg_res = []
            for chunk_st in range(0, data.shape[0], chunk_shift):
                chunk_ed = min(chunk_st + chunk_len, data.shape[0])
                if chunk_ed <= last_chunk_ed:
                    break
                last_chunk_ed = chunk_ed
                chunk_st = max(0, chunk_ed - chunk_len)
                chunk_data = data[chunk_st:chunk_ed]
                if chunk_data.shape[0] < chunk_len:
                    chunk_data = np.pad(chunk_data,
                                        (0, chunk_len - chunk_data.shape[0]),
                                        'constant')
                seg_res.append([
                    chunk_st / self.fs + seg_st, chunk_ed / self.fs + seg_st,
                    chunk_data
                ])
            return seg_res

        segs = []
        for i, s in enumerate(vad_segments):
            segs.extend(seg_chunk(s))

        return segs

    def get_embedding(self, segments):
        embeddings = []
        for s in segments:
            save_dict = self.sv_pipeline([s[2]], output_emb=True)
            if save_dict['embs'].shape == (1, 192):
                embeddings.append(save_dict['embs'])
        embeddings = np.concatenate(embeddings)
        return embeddings

    def merge_segments(self, segments, interval):
        if not segments:
            return []

        merged_segments = []
        current_segment = segments[0]

        for i in range(1, len(segments)):
            next_segment = segments[i]

            # 如果当前片段和下一个片段之间的间隔小于设定值，并且标签相同，则合并
            if next_segment[0] - current_segment[1] <= interval and current_segment[2] == next_segment[2]:
                current_segment[1] = next_segment[1]  # 更新结束时间
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment

        # 添加最后一个片段
        merged_segments.append(current_segment)

        return merged_segments

    def postprocess(self, segments: list, vad_segments: list,
                    labels: np.ndarray, embeddings: np.ndarray) -> list:
        assert len(segments) == len(labels)
        labels = self.correct_labels(labels)
        distribute_res = []
        for i in range(len(segments)):
            distribute_res.append([segments[i][0], segments[i][1], labels[i]])
        # merge the same speakers chronologically
        distribute_res = self.merge_seque(distribute_res)

        # accquire speaker center
        spk_embs = []
        for i in range(labels.max() + 1):
            spk_emb = embeddings[labels == i].mean(0)
            spk_embs.append(spk_emb)
        spk_embs = np.stack(spk_embs)

        def is_overlapped(t1, t2):
            if t1 > t2 + 1e-4:
                return True
            return False

        # distribute the overlap region
        for i in range(1, len(distribute_res)):
            if is_overlapped(distribute_res[i - 1][1], distribute_res[i][0]):
                p = (distribute_res[i][0] + distribute_res[i - 1][1]) / 2
                short_utt_st = max(p - 1.5, distribute_res[i - 1][0])
                short_utt_ed = min(p + 1.5, distribute_res[i][1])
                if short_utt_ed - short_utt_st > 1:
                    audio_data = self.cut_audio(short_utt_st, short_utt_ed,
                                                vad_segments)
                    spk1 = distribute_res[i - 1][2]
                    spk2 = distribute_res[i][2]
                    _, ct = self.change_locator_pipeline(
                        audio_data, [spk_embs[spk1], spk_embs[spk2]],
                        output_res=True)
                    if ct is not None:
                        p = short_utt_st + ct
                distribute_res[i][0] = p
                distribute_res[i - 1][1] = p

        # smooth the result
        distribute_res = self.smooth(distribute_res)

        return distribute_res

    def cut_audio(self, cut_st: float, cut_ed: float,
                  audio: Union[np.ndarray, list]) -> np.ndarray:
        # collect audio data given the start and end time.
        if isinstance(audio, np.ndarray):
            return audio[int(cut_st * self.fs):int(cut_ed * self.fs)]
        elif isinstance(audio, list):
            for i in range(len(audio)):
                if i == 0:
                    if cut_st < audio[i][1]:
                        st_i = i
                else:
                    if cut_st >= audio[i - 1][1] and cut_st < audio[i][1]:
                        st_i = i

                if i == len(audio) - 1:
                    if cut_ed > audio[i][0]:
                        ed_i = i
                else:
                    if cut_ed > audio[i][0] and cut_ed <= audio[i + 1][0]:
                        ed_i = i
            audio_segs = audio[st_i:ed_i + 1]
            cut_data = []
            for i in range(len(audio_segs)):
                s_st, s_ed, data = audio_segs[i]
                cut_data.append(
                    data[int((max(cut_st, s_st) - s_st)
                             * self.fs):int((min(cut_ed, s_ed) - s_st)
                                            * self.fs)])
            cut_data = np.concatenate(cut_data)
            return cut_data
        else:
            raise ValueError('modelscope error: Wrong audio format.')

    def correct_labels(self, labels):
        labels_id = 0
        id2id = {}
        new_labels = []
        for i in labels:
            if i not in id2id:
                id2id[i] = labels_id
                labels_id += 1
            new_labels.append(id2id[i])
        return np.array(new_labels)

    def merge_seque(self, distribute_res):
        res = [distribute_res[0]]
        for i in range(1, len(distribute_res)):
            if distribute_res[i][2] != res[-1][2] or distribute_res[i][
                0] > res[-1][1]:
                res.append(distribute_res[i])
            else:
                res[-1][1] = distribute_res[i][1]
        return res

    def smooth(self, res, mindur=1):
        # short segments are assigned to nearest speakers.
        for i in range(len(res)):
            res[i][0] = round(res[i][0], 2)
            res[i][1] = round(res[i][1], 2)
            if res[i][1] - res[i][0] < mindur:
                if i == 0:
                    res[i][2] = res[i + 1][2]
                elif i == len(res) - 1:
                    res[i][2] = res[i - 1][2]
                elif res[i][0] - res[i - 1][1] <= res[i + 1][0] - res[i][1]:
                    res[i][2] = res[i - 1][2]
                else:
                    res[i][2] = res[i + 1][2]
        # merge the speakers
        res = self.merge_seque(res)

        return res
