import argparse
from pathlib import Path
import re
import time
import openai
from typing import List, Tuple
from cacheout import Cache

class zimufanyi:
    def __init__(self, api_key: str, api_url: str = "https://api.openai.com", moxing: str = "gpt-3.5-turbo", buyouhua: bool = True):
        self.moxing = moxing
        self.buyouhua = buyouhua
        openai.api_base = f"{api_url}/v1"
        openai.api_key = api_key
        
    def translate_to_zh(self, text: str, max_retries: int = 3) -> Tuple[bool, str]:
        """openai发力"""
        retries = 0
        while retries < max_retries:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.moxing,
                    messages=[
                        {"role": "system", "content": "You are a translation engine that can only translate text and cannot interpret it."},
                        {"role": "user", "content": f"translate to zh-CN:\n\n{text}"}
                    ],
                    temperature=0,
                    top_p=1
                )
                result = completion.choices[0].message.content.strip()
                return True, result
            except Exception as e:
                retries += 1
                print(f"翻译错误，重试中 (次数 {retries}): {str(e)}")
                if retries < max_retries:
                    time.sleep(5)  # Wait before retrying
                else:
                    print("超过重试次数，跳过")
                    return False, str(e)

    def merge_subtitles(self, subtitles: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
        """处理字幕"""
        if not subtitles:
            return []
        
        merged = []
        current_group = list(subtitles[0])
        
        def should_merge(text1: str, text2: str) -> bool:
            text1, text2 = text1.strip(), text2.strip()
            
            if re.search(r'[.!?。！？]\s*$', text1):
                return False
                
            if re.match(r'^[a-z,;]', text2):
                return True
                
            # 处理短语
            incomplete_patterns = [
                r'[,，]\s*$',  # 以逗号结束
                r'\b(the|a|an)\s*$',  # 以冠词结束
                r'\b(and|or)\s*$',  # 以连词结束
                r'\b(to|of|for|in)\s*$',  # 以介词结束
                r'\bI\s+(will|am)\s*$',  # 未完成的动词短语
                r'\b(will|can|should)\s*$',  # 情态动词
                r'\b(is|are|was|were)\s*$'  # be动词
            ]
            return any(re.search(pattern, text1) for pattern in incomplete_patterns)

        for subtitle in subtitles[1:]:
            if should_merge(current_group[2], subtitle[2]):
                # 合并
                current_group[2] = f"{current_group[2]} {subtitle[2]}"
                # 更新時間戳
                timestamp_parts = subtitle[1].split(' --> ')
                current_timestamp_parts = current_group[1].split(' --> ')
                current_group[1] = f"{current_timestamp_parts[0]} --> {timestamp_parts[1]}"
            else:
                merged.append(tuple(current_group))
                current_group = list(subtitle)
        
        merged.append(tuple(current_group))
        
        # 重编字幕
        return [(str(i), timestamp, text.strip(), f"{i}\n{timestamp}\n{text.strip()}")
                for i, (_, timestamp, text, _) in enumerate(merged, 1)]

    def process_srt_file(self, file_path: Path):
        """处理srt文件，合并字幕、翻译、保存"""
        print(f"正在翻译: {file_path}")
        start_time = time.time()
        
        # 开启文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分块，剔除无意义字幕
        blocks = re.split(r'\n\n+', content.strip())
        subtitles = []
        for block in blocks:
            if not block.strip():
                continue
            parts = block.strip().split('\n', 2)
            if len(parts) >= 3 and not re.match(r'^\s*(♪+|\[.*?\]|\(.*?\)|<.*?>|#|\*+)\s*$', parts[2]):
                subtitles.append((parts[0], parts[1], parts[2], block))
        
        # 合并字幕
        if self.buyouhua:
            merged_subtitles = self.merge_subtitles(subtitles)
        else:
            merged_subtitles = [(sub[0], sub[1], sub[2], sub[3]) for sub in subtitles]
        
        # 准备扔ai
        batch_size = 5000
        batches = []
        current_batch = []
        current_chars = 0
        
        for subtitle in merged_subtitles:
            if current_chars + len(subtitle[2]) > batch_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(subtitle)
            current_chars += len(subtitle[2])
        
        if current_batch:
            batches.append(current_batch)
        
        # 翻译
        translated_subtitles = []
        for batch_idx, batch in enumerate(batches, 1):
            print(f"翻译字幕中 {batch_idx}/{len(batches)}...")
            
            # 翻译
            texts = [text for _, _, text, _ in batch]
            context = "请将以下字幕翻译为中文，保持原始的语气和表达方式。每行翻译请用换行符分隔：\n\n"
            success, translated = self.translate_to_zh(context + "\n".join(texts))
            
            if success:
                translated_lines = translated.strip().split('\n')
                if len(translated_lines) == len(batch):
                    translated_subtitles.extend([
                        (idx, timestamp, trans.strip())
                        for (idx, timestamp, _, _), trans in zip(batch, translated_lines)
                    ])
                else:
                    # 重试
                    for idx, timestamp, text, _ in batch:
                        success, trans = self.translate_to_zh(text)
                        translated_subtitles.append((idx, timestamp, trans.strip() if success else text))
            
            time.sleep(2)  # 重试delay时间
        
        # 保存文件
        for suffix, content in [
            ('.autofix', '\n\n'.join(f"{idx}\n{timestamp}\n{next(sub[2] for sub in merged_subtitles if sub[0] == idx)}"
                                    for idx, timestamp, _ in translated_subtitles)),
            ('.zh', '\n\n'.join(f"{idx}\n{timestamp}\n{text}"
                               for idx, timestamp, text in translated_subtitles)),
            ('.zhen', '\n\n'.join(f"{idx}\n{timestamp}\n{text}\n{next(sub[2] for sub in merged_subtitles if sub[0] == idx)}"
                                 for idx, timestamp, text in translated_subtitles))
        ]:
            output_path = file_path.with_stem(file_path.stem + suffix)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"保存至: {output_path}")
        
        print(f"翻译完成，耗时 {time.time() - start_time:.2f} 秒")

def main():
    parser = argparse.ArgumentParser(description="简单处理字幕后用openai字幕翻译")
    parser.add_argument("--apikey", required=True, help="OpenAI API key")
    parser.add_argument("--apiurl", default="https://api.openai.com", help="OpenAI API URL")
    parser.add_argument("--mulu", default="C:\\fanyi", help="选择字幕文件目录，预设 C:\fanyi\ ")
    parser.add_argument("--buyouhua", action="store_true", help="不进行字幕优化")
    parser.add_argument("--moxing", default="gpt-3.5-turbo", help="输入openai模型")
    args = parser.parse_args()

    translator = zimufanyi(
        api_key=args.apikey,
        api_url=args.apiurl,
        moxing=args.moxing,
        buyouhua=not args.buyouhua
    )

    mulu = Path(args.mulu)
    for file_path in mulu.glob("*.srt"):
        if not file_path.stem.endswith(('.zh', '.zhen', '.autofix')):
            translator.process_srt_file(file_path)

if __name__ == "__main__":
    main()
