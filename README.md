SRT-Gemini-Translator: 稳健的双语字幕翻译器
简介
这是一个基于 Google Gemini API 的 SRT 字幕翻译工具。与传统的整块翻译不同，本项目采用**“结构化索引匹配”**逻辑，彻底解决了翻译后时间轴错位、字幕丢失或格式崩坏的问题。

核心特性
🎯 零错位翻译：通过将 SRT 解析为 序号:内容 的 JSON 结构，只翻译文本，回填时根据序号锁定时间轴，确保时间轴 100% 准确。

恢复力强：支持断点续传。如果翻译中断，重新运行会读取本地 JSON 缓存，直接从失败处继续，不浪费 Token。

二次自动追补：翻译完成后会自动扫描全文，若发现 AI 漏译或格式错误导致的空缺，会自动发起二次追补请求。

快速开始
1. 安装依赖
Bash
pip install -U google-generativeai tqdm chardet
2. 配置 API Key
在程序同级目录下创建或修改 settings.cfg 文件：
settings.cfg.json 改为settings.cfg 更给里面的api为自己的api


Ini, TOML
[option]
gemini-apikey = 你的_GEMINI_API_KEY
target-language = Chinese

3. 运行
Bash
python srt_translation_gemini.py your_subtitle_file.srt
