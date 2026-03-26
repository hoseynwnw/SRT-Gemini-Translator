# -*- coding: utf-8 -*-
import re
import google.generativeai as genai
from tqdm import tqdm
import os
import configparser
import json
import chardet
import argparse
import time

# --- 1. 配置加载 ---
def get_config():
    with open('settings.cfg', 'rb') as f:
        raw = f.read()
        enc = chardet.detect(raw)['encoding']
    config = configparser.ConfigParser()
    with open('settings.cfg', encoding=enc) as f:
        config.read_string(f.read())
    return config.get('option', 'gemini-apikey'), config.get('option', 'target-language')

gemini_key, target_lang = get_config()
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="SRT file path")
args = parser.parse_args()

base_name = os.path.splitext(args.filename)[0]
source_json = f"{base_name}_source.json"
trans_json = f"{base_name}_translated.json"

# --- 2. 提取 SRT 数据 ---
def prepare_data(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 兼容不同系统的换行符，并按空行拆分块
    blocks = re.split(r'\n\s*\n', content.strip())
    source_dict = {}
    time_dict = {}
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            idx = lines[0].strip()
            time_dict[idx] = lines[1].strip()
            source_dict[idx] = " ".join(lines[2:]).strip()
    
    with open(source_json, 'w', encoding='utf-8') as f:
        json.dump({"source": source_dict, "time": time_dict}, f, ensure_ascii=False, indent=4)
    return source_dict, time_dict

# --- 3. 翻译核心函数 ---
def translate_core(batch_dict):
    if not batch_dict: return {}, 0
    
    # 构造：序号,原文
    input_text = "\n".join([f"{k},{v}" for k, v in batch_dict.items()])
    batch_tokens = len(input_text) // 4

    # 终极优化版 Prompt
    prompt = (
        f"You are an expert subtitle localizer. Translate the following list into {target_lang}.\n"
        f"RULES:\n"
        f"1. Output format: 'number,translated_text'.\n"
        f"2. ONE item per line. DO NOT use markdown code blocks.\n"
        f"3. Maintain the original emotional tone (natural spoken language).\n"
        f"4. NO introductory text or explanations.\n\n"
        f"{input_text}"
    )
    
    for retry in range(3):
        try:
            response = model.generate_content(prompt)
            # 暴力清理 Markdown 标签
            clean_text = re.sub(r'```.*?(\n|$)', '', response.text).strip()
            lines = clean_text.split('\n')
            res = {}
            for line in lines:
                if ',' in line:
                    p = line.split(',', 1)
                    idx_key = p[0].strip()
                    trans_val = p[1].strip()
                    if idx_key and trans_val:
                        res[idx_key] = trans_val
            return res, batch_tokens
        except Exception as e:
            if retry < 2:
                time.sleep(5)
                continue
            print(f"\n[API Error] {e}")
            return {}, 0

# --- 4. 执行逻辑 ---
src_dict, time_dict = prepare_data(args.filename)
trans_dict = {}

if os.path.exists(trans_json):
    with open(trans_json, 'r', encoding='utf-8') as f:
        trans_dict = json.load(f)

indices = list(src_dict.keys())
batch_size = 100
accumulated_tokens = 0
submit_count = 0

# A. 第一轮：主翻译进度条
pbar = tqdm(range(0, len(indices), batch_size), desc="翻译中", unit="batch")
for i in pbar:
    batch_keys = indices[i : i + batch_size]
    to_do = {k: src_dict[k] for k in batch_keys if k not in trans_dict}
    
    if to_do:
        submit_count += 1
        result, tokens = translate_core(to_do)
        trans_dict.update(result)
        accumulated_tokens += tokens
        pbar.set_description(f"提交:{submit_count}次 | Token≈{accumulated_tokens}")
        with open(trans_json, 'w', encoding='utf-8') as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=4)
pbar.close()

# B. 第二轮：自动检查并补翻译 (解决缺失问题)
missing_keys = [k for k in indices if k not in trans_dict or len(trans_dict[k]) < 1]
if missing_keys:
    print(f"\n[补查模式] 发现 {len(missing_keys)} 条缺失，正在追补...")
    to_do_missing = {k: src_dict[k] for k in missing_keys}
    # 补翻译时减小 batch 防止再次出错
    result, tokens = translate_core(to_do_missing)
    trans_dict.update(result)
    accumulated_tokens += tokens
    with open(trans_json, 'w', encoding='utf-8') as f:
        json.dump(trans_dict, f, ensure_ascii=False, indent=4)

# --- 5. 导出文件 ---
print("\n正在生成最终字幕...")
pure_res = []
bilingual_res = []

for idx in indices:
    t_range = time_dict[idx]
    orig_text = src_dict[idx]
    # 如果补翻译依然失败，则保留原文以免字幕消失
    trans_text = trans_dict.get(idx) or f"(Missing) {orig_text}"

    # 纯译文版
    pure_res.append(f"{idx}\n{t_range}\n{trans_text}\n")
    # 标准双语版 (序号 -> 时间 -> 原文 -> 译文)
    bilingual_res.append(f"{idx}\n{t_range}\n{orig_text}\n{trans_text}\n")

# 写入
with open(f"{base_name}_pure.srt", 'w', encoding='utf-8') as f:
    f.write("\n".join(pure_res))
with open(f"{base_name}_bilingual.srt", 'w', encoding='utf-8') as f:
    f.write("\n".join(bilingual_res))

# 清理缓存
if os.path.exists(source_json): os.remove(source_json)
if os.path.exists(trans_json): os.remove(trans_json)

print(f"\n✨ 翻译圆满完成！")
print(f"📊 统计：提交 {submit_count} 次，预估消耗 {accumulated_tokens} Tokens。")
print(f"📁 已生成：\n1. {base_name}_pure.srt (纯译文)\n2. {base_name}_bilingual.srt (标准双语)")