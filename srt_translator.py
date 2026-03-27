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
            f.write("[option]\ngemini-apikey = YOUR_API_KEY\ntarget-language = Chinese")
        print("已生成 settings.cfg，请填写 API Key 后运行。")
        exit()
        
    with open('settings.cfg', 'rb') as f:
        raw = f.read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
    
    config = configparser.ConfigParser()
    config.read('settings.cfg', encoding=enc)
    return config.get('option', 'gemini-apikey-srt1'), config.get('option', 'target-language')

gemini_key, target_lang = get_config()
client = genai.Client(api_key=gemini_key)
# 建议使用稳定版本 gemini-1.5-flash 以获得更好的格式遵循能力
MODEL_ID = "gemini-3.1-flash-lite-preview" 

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="SRT 字幕文件路径")
args = parser.parse_args()

# 清理路径中的引号
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
    
    # 使用正则表达式分割字幕块，处理不同系统的换行符
    blocks = re.split(r'\n\s*\n', content.strip())
    src_dict, full_blocks = {}, {} 
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            idx = lines[0].strip()
            # 合并可能存在的多行文本
            text = " ".join(lines[2:]).strip()
            src_dict[idx] = text
            full_blocks[idx] = lines
    
    with open(source_json, 'w', encoding='utf-8') as f:
        json.dump(src_dict, f, ensure_ascii=False, indent=4)
    return src_dict, full_blocks, list(src_dict.keys())

# --- 3. 翻译核心逻辑 ---
LAST_REQUEST_TIME = 0
RPM_INTERVAL = 1.5 # 对应约 40 RPM，免费版建议设为 3-4

def translate_batch(batch_dict, is_retry=False):
    global LAST_REQUEST_TIME
    if not batch_dict: return {}
    
    # 频率限制
    elapsed = time.time() - LAST_REQUEST_TIME
    if elapsed < RPM_INTERVAL:
        time.sleep(RPM_INTERVAL - elapsed)

    # 构造更清晰的输入格式
    input_text = "\n".join([f"ID_{k}: {v}" for k, v in batch_dict.items()])

    # 强化 Prompt：明确要求即便句子简短也必须翻译，严禁合并
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
    
    # 如果是补译阶段，增加温度以获取不同结果
    temp = 0.4 if is_retry else 0.1
    
    for attempt in range(3):
        try:
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
                    # 稳健提取数字 ID
                    idx_match = re.search(r'\d+', parts[0])
                    if idx_match:
                        idx_str = idx_match.group()
                        if idx_str in batch_dict:
                            res[idx_str] = parts[1].strip()

            # 检查是否全部翻译完成
            if len(res) >= len(batch_dict):
                return res
            else:
                print(f"警告：批次缺失 {len(batch_dict)-len(res)} 行，重试中 (尝试 {attempt+1}/3)...")
                time.sleep(2 * (attempt + 1))
                
        except Exception as e:
            if "429" in str(e):
                print("达到频率限制，休眠 15 秒...")
                time.sleep(15)
            else:
                print(f"API 异常: {e}")
                time.sleep(2)
                
    return res

# --- 4. 执行翻译流程 ---
src_dict, full_blocks, all_indices = prepare_data(clean_path)
trans_dict = {}

# 加载已有的翻译（如果有）
if os.path.exists(trans_json):
    with open(trans_json, 'r', encoding='utf-8') as f:
        trans_dict = json.load(f)

# 第一轮：翻译未完成或被标记为 FIXME 的行
undone = [k for k in all_indices if k not in trans_dict or not trans_dict[k] or "[FIXME]" in str(trans_dict[k])]

if undone:
    batch_size = 50 # 较小的批次更稳健
    pbar = tqdm(range(0, len(undone), batch_size), desc="🚀 正在翻译")
    for i in pbar:
        batch_keys = undone[i : i + batch_size]
        result = translate_batch({k: src_dict[k] for k in batch_keys})
        trans_dict.update(result)
        # 实时保存，防止崩溃丢失进度
        with open(trans_json, 'w', encoding='utf-8') as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=4)
    pbar.close()

# 第二轮：深度补译（针对顽固的 FIXME 或者是漏译行）
final_missing = [k for k in all_indices if k not in trans_dict or not trans_dict[k] or "[FIXME]" in str(trans_dict[k])]
if final_missing:
    print(f"\n🔍 正在进行深度补译 (处理 {len(final_missing)} 条顽固项)...")
    # 补译采用单条处理模式，确保 100% 成功率
    for k in tqdm(final_missing):
        result = translate_batch({k: src_dict[k]}, is_retry=True)
        if result and k in result:
            trans_dict[k] = result[k]
        else:
            # 如果单条都失败，记录原文以便后续手动检查
            trans_dict[k] = f"[FIXME] {src_dict[k]}"
        
        with open(trans_json, 'w', encoding='utf-8') as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=4)

# --- 5. 导出结果文件 ---
bilingual_res, chinese_res = [], []
for idx in all_indices:
    t_range = full_blocks[idx][1]
    orig = src_dict[idx]
    # 最终检查，如果没有翻译则显示原文
    trans = trans_dict.get(idx) or f"[FIXME] {orig}"

    bilingual_res.append(f"{idx}\n{t_range}\n{orig}\n{trans}\n")
    chinese_res.append(f"{idx}\n{t_range}\n{trans}\n")

# ... 之前的导出逻辑 ...

with open(f"{base_name}_dual.srt", 'w', encoding='utf-8') as f:
    f.write("\n".join(bilingual_res))
with open(f"{base_name}_chinese.srt", 'w', encoding='utf-8') as f:
    f.write("\n".join(chinese_res))

print(f"\n✅ 字幕生成成功！")
print(f"双语字幕: {base_name}_dual.srt")
print(f"中文字幕: {base_name}_chinese.srt")

# --- 新增：清理临时 JSON ---
print(f"\n正在清理临时缓存...")
for temp_file in [source_json, trans_json]:
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"已清理: {temp_file}")

print(f"✨ 全部流程处理完毕。")
