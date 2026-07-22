"""
GPT-SoVITS Core Inference Pipeline

Mirrors the official inference_webui.py get_tts_wav() flow:
  1. Reference audio → SSL features → prompt_semantic + refer_spec
  2. Text preprocessing → phones + BERT features
  3. GPT inference → pred_semantic
  4. SoVITS decode → waveform

Supports v2/v2Pro/v2ProPlus.
"""

import os
import time
import traceback

import numpy as np
import torch
import torchaudio
import librosa

from engines.gpt_sovits.runtime import configure_gpt_sovits_source


def get_tts_wav(
    model_loader,
    ref_wav_path: str,
    prompt_text: str,
    prompt_language: str,
    text: str,
    text_language: str,
    how_to_cut: str = "凑四句一切",
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    speed: float = 1.0,
    pause_second: float = 0.3,
):
    """Generate TTS audio using loaded GPT-SoVITS models.

    Args:
        model_loader: GPTSovitsModelLoader instance with loaded models
        ref_wav_path: Path to reference audio (3-10s wav)
        prompt_text: Transcript of reference audio
        prompt_language: Language code for prompt (zh/en/ja)
        text: Target text to synthesize
        text_language: Language code for target text
        how_to_cut: Text splitting method
        top_k: GPT top-k sampling
        top_p: GPT top-p sampling
        temperature: GPT temperature
        speed: Speech speed factor
        pause_second: Silence duration between sentences

    Returns:
        (sample_rate, waveform_tensor) where waveform is [1, samples]
    """
    loader = model_loader
    device = loader.device
    is_half = loader.is_half
    hps = loader.hps
    dtype = torch.float16 if is_half else torch.float32

    configure_gpt_sovits_source()
    from GPT_SoVITS.inference_webui import (
        dict_language_v1,
        dict_language_v2,
        cut1, cut2, cut3, cut4, cut5,
        get_phones_and_bert,
        get_spepc,
        merge_short_text_in_array,
        process_text,
    )

    dict_language = dict_language_v1 if loader.version == "v1" else dict_language_v2
    splits = {"。", ".", "？", "?", "！", "!", "~"}

    t0 = time.perf_counter()

    # Validate inputs
    if not ref_wav_path or not os.path.isfile(ref_wav_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_wav_path}")
    if not text:
        raise ValueError("Text is empty")

    # Map language display names to codes
    prompt_language = dict_language.get(prompt_language, "zh")
    text_language = dict_language.get(text_language, "zh")

    # Clean prompt text
    ref_free = False
    if prompt_text is None or len(prompt_text.strip()) == 0:
        ref_free = True
        prompt_text = ""

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text and prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."

    text = text.strip("\n")

    # === Phase 1: Reference Audio Processing ===
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half else np.float32,
    )

    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise ValueError("Reference audio must be 3-10 seconds")

            wav16k = torch.from_numpy(wav16k).to(device)
            if is_half:
                wav16k = wav16k.half()

            zero_wav_torch = torch.from_numpy(zero_wav).to(device)
            if is_half:
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = loader.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(1, 2)
            codes = loader.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = time.perf_counter()

    # === Phase 2: Text Splitting ===
    cut_map = {
        "凑四句一切": cut1,
        "凑50字一切": cut2,
        "按中文句号。切": cut3,
        "按英文句号.切": cut4,
        "按标点符号切": cut5,
        "不切": lambda x: x,
    }
    cut_fn = cut_map.get(how_to_cut, cut1)
    text = cut_fn(text)
    text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)

    # === Phase 3 & 4: GPT Inference + SoVITS Decode ===
    audio_opt = []
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, loader.version)

    for i_text, seg_text in enumerate(texts):
        if len(seg_text.strip()) == 0:
            continue
        if seg_text[-1] not in splits:
            seg_text += "。" if text_language != "en" else "."

        phones2, bert2, norm_text2 = get_phones_and_bert(seg_text, text_language, loader.version)

        # Build input tensors
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = time.perf_counter()

        # GPT inference
        with torch.no_grad():
            pred_semantic, idx = loader.t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=loader.hz * loader.max_sec,
            )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        t3 = time.perf_counter()

        # SoVITS decode
        # Get reference speaker characteristics
        refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, loader.is_v2pro)
        refers = [refers]

        sv_emb = None
        if loader.is_v2pro and loader.sv_model is not None:
            sv_emb = [loader.sv_model.compute_embedding3(audio_tensor)]

        with torch.no_grad():
            if loader.is_v2pro and sv_emb is not None:
                audio = loader.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(device).unsqueeze(0),
                    refers,
                    speed=speed,
                    sv_emb=sv_emb,
                )[0][0]
            else:
                audio = loader.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(device).unsqueeze(0),
                    refers,
                    speed=speed,
                )[0][0]

        audio = audio.detach().cpu().numpy()
        audio_opt.append(audio)
        audio_opt.append(zero_wav)

        t4 = time.perf_counter()

    # === Final: Concatenate and convert ===
    if not audio_opt:
        raise RuntimeError("No audio generated")

    audio_final = np.concatenate(audio_opt, axis=0)
    waveform = torch.from_numpy(audio_final).float().unsqueeze(0).unsqueeze(0)
    sample_rate = hps.data.sampling_rate

    total_time = time.perf_counter() - t0
    print(f"   TTS done: {total_time:.2f}s, sr={sample_rate}, duration={len(audio_final)/sample_rate:.2f}s")

    return sample_rate, waveform
