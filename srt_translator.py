# Version: 0.1.5
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
import concurrent.futures

# --- 1. 配置与初始化 ---
def get_config():
    if not os.path.exists('settings.cfg'):
        with open('settings.cfg', 'w', encoding='utf-8') as f:
            f.write("[option]\ngemini-apikey = YOUR_API_KEY\ntarget-language = Chinese\nmodel = gemini-1.5-flash-lite")
        print("已生成 settings.cfg，请填写 API Key 后运行。")
        exit()
        
    with open('settings.cfg', 'rb') as f:
        raw = f.read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
    
    config = configparser.ConfigParser()
    config.read('settings.cfg', encoding=enc)
    
    keys = []
    if config.has_section('option'):
        for option in config.options('option'):
            if option.startswith('gemini-apikey'):
                val = config.get('option', option)
                if val and "YOUR_API_KEY" not in val and not val.strip().startswith('#'):
                    keys.append(val.strip())
    
    if not keys:
        print("未在 settings.cfg 中找到任何有效的 API Key。")
        exit()
    
    target_model = config.get('option', 'model', fallback='gemini-1.5-flash-lite').strip()
    target_lang = config.get('option', 'target-language', fallback='Chinese').strip()
    
    print(f"成功加载 {len(keys)} 个 API Key。")
    print(f"当前使用模型: {target_model}")
    return keys, target_lang, target_model

api_keys, target_lang, MODEL_ID = get_config()
current_key_index = 0
client = genai.Client(api_key=api_keys[current_key_index])

def switch_api_key(reason=""):
    global client, current_key_index
    if len(api_keys) <= 1:
        return False
    
    old_index = current_key_index + 1
    current_key_index = (current_key_index + 1) % len(api_keys)
    new_key = api_keys[current_key_index]
    client = genai.Client(api_key=new_key)
    
    print(f"\n🔄 Key #{old_index} 受限或异常，已切换至 Key #{current_key_index + 1}")
    if reason:
        print(f"原因: {reason}")
    return True

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="SRT 字幕文件路径")
args = parser.parse_args()

clean_path = args.filename.strip("'").strip('"')
base_name = os.path.splitext(clean_path)[0]
source_json = f"{base_name}_source.json"      
trans_json = f"{base_name}_translated.json"  

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
RPM_INTERVAL = 0.8 

def _raw_api_call(prompt, is_repair_mode):
    """纯粹的 API 调用，在子线程中运行"""
    return client.models.generate_content(
        model=MODEL_ID, 
        contents=prompt,
        config={
            'temperature': 0.2 if is_repair_mode else 0.1, 
            'max_output_tokens': 2048
        }
    )

def translate_batch(batch_dict, is_repair_mode=False):
    global LAST_REQUEST_TIME, client
    if not batch_dict: return {}
    
    input_text = "\n".join([f"ID_{k}: {v}" for k, v in batch_dict.items()])
    prompt = (
        f"You are a professional translator. Translate these subtitle lines into {target_lang}.\n"
        f"RULES:\n1. FORMAT: 'ID ||| Translation'.\n2. ONE-TO-ONE: Translate every ID.\n"
        f"3. OUTPUT ONLY the list.\n\n"
        f"SOURCE:\n{input_text}"
    )
    
    max_attempts = 3 if is_repair_mode else 2
    
    for attempt in range(max_attempts):
        try:
            elapsed = time.time() - LAST_REQUEST_TIME
            if elapsed < RPM_INTERVAL:
                time.sleep(RPM_INTERVAL - elapsed)

            LAST_REQUEST_TIME = time.time()
            
            # Windows 兼容的超时监控逻辑
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_raw_api_call, prompt, is_repair_mode)
                try:
                    # 如果 40 秒没拿回结果，就认为网络死掉了
                    response = future.result(timeout=40) 
                except concurrent.futures.TimeoutError:
                    raise Exception("WIN_TIMEOUT: 网络响应太慢，已强制断开重试")

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

            if res:
                if len(res) < len(batch_dict) and not is_repair_mode:
                    print(f"\n📢 提示：收到 {len(res)}/{len(batch_dict)} 条，缺失项将在补译环节处理。")
                return res
            else:
                if is_repair_mode:
                    print(f"警告：AI 返回为空，正在进行第 {attempt+1} 次重试...")

        except Exception as e:
            err_str = str(e).upper()
            # 针对 Windows 网络波动和 API 限制的综合判断
            if any(x in err_str for x in ["TIMEOUT", "DEADLINE", "SSL", "429", "QUOTA", "EOF", "DISCONNECTED"]):
                if switch_api_key(reason=str(e)):
                    continue
            print(f"\nAPI 异常: {e}")
            time.sleep(2)
                
    return {}

# --- 4. 执行流程 ---
src_dict, full_blocks, all_indices = prepare_data(clean_path)
trans_dict = {}

if os.path.exists(trans_json):
    with open(trans_json, 'r', encoding='utf-8') as f:
        trans_dict = json.load(f)

undone = [k for k in all_indices if k not in trans_dict or not trans_dict[k] or "[FIXME]" in str(trans_dict[k])]

batch_size = 40 

if undone:
    pbar = tqdm(range(0, len(undone), batch_size), desc="🚀 正在初翻")
    for i in pbar:
        batch_keys = undone[i : i + batch_size]
        result = translate_batch({k: src_dict[k] for k in batch_keys}, is_repair_mode=False)
        trans_dict.update(result)
        with open(trans_json, 'w', encoding='utf-8') as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=4)
    pbar.close()

# --- 5. 补译环节 ---
final_missing = [k for k in all_indices if k not in trans_dict or not trans_dict[k] or "[FIXME]" in str(trans_dict[k])]
if final_missing:
    repair_batch_size = max(1, int(batch_size * 0.5))
    print(f"\n🔍 发现 {len(final_missing)} 条缺失，启动最终补译 (Batch: {repair_batch_size})...")
    
    repair_pbar = tqdm(range(0, len(final_missing), repair_batch_size), desc="🛠️ 补救中")
    for i in repair_pbar:
        batch_keys = final_missing[i : i + repair_batch_size]
        result = translate_batch({k: src_dict[k] for k in batch_keys}, is_repair_mode=True)
        
        for k in batch_keys:
            if k in result:
                trans_dict[k] = result[k]
            else:
                if k not in trans_dict or not trans_dict[k]:
                    trans_dict[k] = f"[FIXME] {src_dict[k]}"
                
        with open(trans_json, 'w', encoding='utf-8') as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=4)
    repair_pbar.close()

# --- 6. 导出文件 ---
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

print(f"\n✅ 字幕处理完成！")
if os.path.exists(source_json): os.remove(source_json)
if os.path.exists(trans_json): os.remove(trans_json)
