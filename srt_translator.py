# -*- coding: utf-8 -*-
import re
from google import genai
from tqdm import tqdm
import os
import configparser
import json
import chardet
import argparse
import time

# --- 1. 配置与初始化 ---
def get_config():
    if not os.path.exists('settings.cfg'):
        with open('settings.cfg', 'w', encoding='utf-8') as f:
            f.write("[option]\ngemini-apikey = YOUR_API_KEY\ngemini-apikey-srt1 = YOUR_API_KEY_1\ngemini-apikey-srt2 = YOUR_API_KEY_2\ngemini-apikey-srt3 = YOUR_API_KEY_3\ntarget-language = Chinese")
        print("已生成 settings.cfg，请填写 API Key 后运行。")
        exit()
        
    with open('settings.cfg', 'rb') as f:
        raw = f.read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
    
    config = configparser.ConfigParser()
    config.read('settings.cfg', encoding=enc)
    
    # 动态获取 settings.cfg 中定义的多个 Key
    key_fields = ['gemini-apikey', 'gemini-apikey-srt1', 'gemini-apikey-srt2', 'gemini-apikey-srt3']
    keys = []
    for field in key_fields:
        val = config.get('option', field, fallback=None)
        if val and "YOUR_API_KEY" not in val and not val.startswith('#'):
            keys.append(val)
    
    if not keys:
        print("未在 settings.cfg 中找到任何有效的 API Key。")
        exit()
        
    return keys, config.get('option', 'target-language')

# 获取 Key 列表并初始化
api_keys, target_lang = get_config()
current_key_index = 0
client = genai.Client(api_key=api_keys[current_key_index])
MODEL_ID = "gemini-3.1-flash-lite-preview" # 建议使用稳定版本

def switch_api_key():
    """检测到限制时，自动切换到下一个 Key"""
    global client, current_key_index
    if len(api_keys) <= 1:
        return False
    
    current_key_index = (current_key_index + 1) % len(api_keys)
    new_key = api_keys[current_key_index]
    client = genai.Client(api_key=new_key)
    print(f"\n🔄 当前 Key 额度耗尽或受限，已切换至 Key #{current_key_index + 1}")
    return True

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="SRT 字幕文件路径")
args = parser.parse_args()

clean_path = args.filename.strip("'").strip('"')
base_name = os.path.splitext(clean_path)[0]
source_json = f"{base_name}_source.json"      
trans_json = f"{base_name}_translated.json"  

# --- 2. 提取 SRT 数据 ---
def prepare_data(srt_path):
    if not os.path.exists(srt_path):
        print(f"找不到文件: {srt_path}")
        exit()
    with open(srt_path, 'rb') as f:
        raw = f.read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
    with open(srt_path, 'r', encoding=enc) as f:
        content = f.read()
    
    blocks = re.split(r'\n\s*\n', content.strip())
    src_dict, full_blocks = {}, {} 
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            idx = lines[0].strip()
            text = " ".join(lines[2:]).strip()
            src_dict[idx] = text
            full_blocks[idx] = lines
    
    with open(source_json, 'w', encoding='utf-8') as f:
        json.dump(src_dict, f, ensure_ascii=False, indent=4)
    return src_dict, full_blocks, list(src_dict.keys())

# --- 3. 翻译核心逻辑 ---
LAST_REQUEST_TIME = 0
RPM_INTERVAL = 1.0 

def translate_batch(batch_dict, is_retry=False):
    global LAST_REQUEST_TIME, client
    if not batch_dict: return {}
    
    input_text = "\n".join([f"ID_{k}: {v}" for k, v in batch_dict.items()])
    prompt = (
         f"You are a professional translator. Translate these subtitle lines into {target_lang}.\n"
        f"CONTEXT: Academic history/education lecture.\n\n"
        f"RULES:\n"
        f"1. FORMAT: 'ID ||| Translation'. Example: '1060 ||| 这里的翻译内容'.\n"
        f"2. ONE-TO-ONE: You must provide a translation for EVERY single ID. Do not skip or merge.\n"
        f"3. LITERAL: If the text is short (e.g. 'Right?'), translate it literally. Do not leave it empty.\n"
        f"4. OUTPUT ONLY the list. No explanations.\n\n"
        f"SOURCE:\n"
        f"{input_text}"
    )
    
    temp = 0.4 if is_retry else 0.1
    # 最大尝试次数设为 Key 数量的两倍，确保每个 Key 都有机会尝试
    max_total_attempts = len(api_keys) * 2 
    
    for attempt in range(max_total_attempts):
        try:
            elapsed = time.time() - LAST_REQUEST_TIME
            if elapsed < RPM_INTERVAL:
                time.sleep(RPM_INTERVAL - elapsed)

            LAST_REQUEST_TIME = time.time()
            response = client.models.generate_content(
                model=MODEL_ID, 
                contents=prompt,
                config={'temperature': temp, 'max_output_tokens': 2048}
            )
            
            res = {}
            raw_text = response.text.replace('```', '').strip()
            lines = raw_text.split('\n')
            
            for line in lines:
                if "|||" in line:
                    parts = line.split("|||", 1)
                    idx_match = re.search(r'\d+', parts[0])
                    if idx_match:
                        idx_str = idx_match.group()
                        if idx_str in batch_dict:
                            res[idx_str] = parts[1].strip()

            if len(res) >= len(batch_dict):
                return res
            else:
                print(f"警告：翻译缺失，重试中 ({attempt+1}/{max_total_attempts})...")
                time.sleep(1)
                
        except Exception as e:
            err_str = str(e).upper()
            # 捕获 429 (频率限制) 或 QUOTA_EXCEEDED (配额用尽)
            if "429" in err_str or "QUOTA" in err_str or "LIMIT" in err_str:
                if switch_api_key():
                    time.sleep(1) 
                    continue
                else:
                    print("所有 API Key 均已达到限制，休眠 30 秒...")
                    time.sleep(30)
            else:
                print(f"API 异常: {e}")
                time.sleep(2)
                
    return {}

# --- 4. 执行翻译流程 ---
src_dict, full_blocks, all_indices = prepare_data(clean_path)
trans_dict = {}

if os.path.exists(trans_json):
    with open(trans_json, 'r', encoding='utf-8') as f:
        trans_dict = json.load(f)

undone = [k for k in all_indices if k not in trans_dict or not trans_dict[k] or "[FIXME]" in str(trans_dict[k])]

if undone:
    batch_size = 50 
    pbar = tqdm(range(0, len(undone), batch_size), desc="🚀 正在翻译")
    for i in pbar:
        batch_keys = undone[i : i + batch_size]
        result = translate_batch({k: src_dict[k] for k in batch_keys})
        trans_dict.update(result)
        with open(trans_json, 'w', encoding='utf-8') as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=4)
    pbar.close()

final_missing = [k for k in all_indices if k not in trans_dict or not trans_dict[k] or "[FIXME]" in str(trans_dict[k])]
if final_missing:
    print(f"\n🔍 正在补译 {len(final_missing)} 条...")
    for k in tqdm(final_missing):
        result = translate_batch({k: src_dict[k]}, is_retry=True)
        if result and k in result:
            trans_dict[k] = result[k]
        else:
            trans_dict[k] = f"[FIXME] {src_dict[k]}"
        with open(trans_json, 'w', encoding='utf-8') as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=4)

# --- 5. 导出结果文件 ---
bilingual_res, chinese_res = [], []
for idx in all_indices:
    t_range = full_blocks[idx][1]
    orig = src_dict[idx]
    trans = trans_dict.get(idx) or f"[FIXME] {orig}"
    bilingual_res.append(f"{idx}\n{t_range}\n{orig}\n{trans}\n")
    chinese_res.append(f"{idx}\n{t_range}\n{trans}\n")

with open(f"{base_name}_dual.srt", 'w', encoding='utf-8') as f:
    f.write("\n".join(bilingual_res))
with open(f"{base_name}_chinese.srt", 'w', encoding='utf-8') as f:
    f.write("\n".join(chinese_res))

print(f"\n✅ 字幕生成成功！")
if os.path.exists(source_json): os.remove(source_json)
if os.path.exists(trans_json): os.remove(trans_json)
print(f"✨ 处理完毕。")
