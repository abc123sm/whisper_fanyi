import os
import argparse
import subprocess
import json
import tempfile
import requests
from pathlib import Path
from faster_whisper import WhisperModel
from zimufanyi import zimufanyi
import time
from groq import Groq
# 使用intel显卡，仅linux可用
#import intel_extension_for_pytorch as ipex


class AudioHandler:
    @staticmethod
    def get_audio_codec(file_path):
        """获取音频编码格式"""
        try:
            command = ['ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
                      '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
                      str(file_path)]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.stdout.decode('utf-8').strip() if result.returncode == 0 else None
        except Exception as e:
            print(f"获取音频编码失败: {e}")
            return None

    @staticmethod
    def prepare_audio(video_path, temp_dir=None):
        """
        处理音频：
        1. 如果是音频文件，直接返回路径
        2. 如果是视频文件：
           - 先尝试提取原始音频流（AAC/FLAC）
           - 如果提取失败，则转码为Opus格式ogg封装
        """
        # 音频文件直接返回
        ## audio_formats = {'.aac', '.flac', '.ogg', '.m4a', '.mp3'}
        ## if video_path.suffix.lower() in audio_formats:
        ##     return str(video_path)
        
        # 获取音频编码格式
        ## codec = AudioHandler.get_audio_codec(video_path)
        ## if not codec:
        ##     print(f"无法获取音频编码信息")
        ##     return None

        # 创建临时文件
        output_ext = '.ogg'
        temp_audio = tempfile.NamedTemporaryFile(
            suffix=output_ext,
            dir=temp_dir,
            delete=False
        )
        ## 
        ## # 尝试直接提取音频流
        ## if codec in ['aac', 'flac']:
        ##     command = [
        ##         'ffmpeg',
        ##         '-hide_banner',
        ##         '-loglevel', 'warning', '-y',
        ##         '-i', str(video_path),
        ##         '-vn',
        ##         '-acodec', 'copy',
        ##         temp_audio.name
        ##     ]
        ##     try:
        ##         subprocess.run(command, check=True)
        ##         print(f"成功提取原始{codec.upper()}音频流")
        ##         return temp_audio.name
        ##     except subprocess.CalledProcessError as e:
        ##         print(f"提取音频流失败: {e}")

        # 转换为Opus格式ogg封装
        print("转换音频为Opus/OGG格式")
        temp_audio = tempfile.NamedTemporaryFile(
            suffix='.ogg',
            dir=temp_dir,
            delete=False
        )
        command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning', '-y',
            '-i', str(video_path),
            '-vn',
            '-acodec', 'libopus',
            '-ar', '16000',
            '-ac', '1',
            '-b:a', '32k',
            '-f', 'ogg',
            temp_audio.name
        ]
        
        try:
            subprocess.run(command, check=True)
            return temp_audio.name
        except subprocess.CalledProcessError as e:
            print(f"音频转换失败: {e}")
            return None

class WhisperTranscriber:
    def __init__(self, mode, model_name, api_url=None, api_key=None):
        self.mode = mode
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        
        if mode == "bendi":
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            #self.model = WhisperModel(model_name, device="xpu")
            
        if mode == "groq":
            self.client = Groq(api_key=api_key)

    def transcribe(self, audio_path, language):
        if self.mode == "bendi":
            return self._local_transcribe(audio_path, language)
        elif self.mode == "api":
            return self._api_transcribe(audio_path, language)
        elif self.mode == "cf_api":
            return self._cf_api_transcribe(audio_path, language)
        elif self.mode == "cf_worker":
            return self._cf_worker_transcribe(audio_path, language)
        elif self.mode == "groq":
            return self._groq_transcribe(audio_path, language)

    def _groq_transcribe(self, audio_path, language):
        """使用Groq API进行转录"""
        print(f"\n[Groq调试] 开始处理音频文件: {audio_path}")
        
        try:
            with open(audio_path, "rb") as file:
                start_time = time.time()
                transcription = self.client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    language=language if language != 'auto' else None,
                    temperature=0.0
                )
                latency = time.time() - start_time
                print(f"[Groq调试] 请求耗时: {latency:.2f}s")
                
                # 将响应转换为字典
                response = json.loads(transcription.model_dump_json())
                print(f"[Groq调试] 原始响应:\n{json.dumps(response, indent=2, ensure_ascii=False)}")
                
                return self._parse_groq_response(response)
        except Exception as e:
            print(f"[Groq异常] API调用失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _parse_groq_response(self, response):
        """解析Groq的响应格式"""
        segments = []
        try:
            for segment in response.get('segments', []):
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip().replace('\n', ' ')
                segments.append((start, end, text))
            
            print(f"[Groq解析] 成功解析 {len(segments)} 个时间段")
            return segments
        except KeyError as e:
            print(f"[Groq解析错误] 缺少必要字段: {str(e)}")
            return []


    def _local_transcribe(self, audio_path, language):
        segments, _ = self.model.transcribe(
            audio_path,
            language=language if language != 'auto' else None
        )
        return [(seg.start, seg.end, seg.text) for seg in segments]

    def _api_transcribe(self, audio_path, language):
        """标准OpenAI Whisper API调用"""
        print(f"\n[OpenAI调试] 开始处理音频文件: {audio_path}")
        
        # 构造请求参数
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # 不需要手动设置Content-Type，requests会自动处理
        }
        
        files = {
            'file': (os.path.basename(audio_path), open(audio_path, 'rb'), 'audio/ogg')
        }
        
        data = {
            "model": "whisper-1",
            "response_format": "verbose_json",  # 获取带时间戳的详细响应
            "language": language if language != 'auto' else None
        }
        
        # 清理None值参数
        data = {k: v for k, v in data.items() if v is not None}
        
        print(f"[OpenAI调试] 请求URL: {self.api_url}/v1/audio/transcriptions")
        print(f"[OpenAI调试] 请求头: Authorization: Bearer {self.api_key[:8]}******")
        print(f"[OpenAI调试] 请求参数: {json.dumps(data, indent=2)}")
        
        try:
            # 发送请求
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data
            )
            latency = time.time() - start_time
            print(f"[OpenAI调试] 请求耗时: {latency:.2f}s")
            print(f"[OpenAI调试] 响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                print(f"[OpenAI错误] 请求失败: {error_msg}")
                return []

            # 解析响应
            result = response.json()
            print(f"[OpenAI调试] 原始响应:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
            
            return self._parse_openai_response(result)
            
        except Exception as e:
            print(f"[OpenAI异常] API调用失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _parse_openai_response(self, response):
        """解析OpenAI详细JSON响应"""
        print("[OpenAI解析] 开始解析响应...")
        
        segments = []
        try:
            # 检查是否存在带时间戳的响应
            if 'segments' not in response:
                print("[OpenAI解析] 未找到时间戳数据，使用完整文本")
                return [(0, 0, response.get('text', ''))]
            
            for segment in response['segments']:
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
                
                # 合并相同时间段的换行
                text = text.replace('\n', ' ')
                
                segments.append((start, end, text))
            
            print(f"[OpenAI解析] 成功解析 {len(segments)} 个时间段")
            return segments
            
        except KeyError as e:
            print(f"[OpenAI解析错误] 缺少必要字段: {str(e)}")
            return []

    def _cf_api_transcribe(self, audio_path, language):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.api_url}/ai/run/@cf/openai/whisper"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/octet-stream"
        }
        
        try:
            print(f"\n[CF调试] 开始处理音频文件: {audio_path}")
            print(f"[CF调试] 使用账户ID: {self.api_url[:4]}**** (完整ID隐藏)")
            print(f"[CF调试] 请求地址: {url}")

            # 读取音频文件
            with open(audio_path, 'rb') as f:
                file_data = f.read()
                print(f"[CF调试] 读取到 {len(file_data)/1024:.2f} KB 音频数据")
                
                # 验证文件头
                print(f"[CF调试] 文件头(HEX): {' '.join(f'{b:02x}' for b in file_data[:4])}")
                
            # 发送请求
            print(f"[CF调试] 正在发送请求...")
            start_time = time.time()
            response = requests.post(
                url,
                headers=headers,
                data=file_data  # 关键修改：使用data参数直接发送二进制
            )
            latency = time.time() - start_time
            print(f"[CF调试] 请求耗时: {latency:.2f}s")

            # 打印响应状态
            print(f"[CF调试] 响应状态码: {response.status_code}")
            print(f"[CF调试] 响应头: {dict(response.headers)}")
            
            # 处理非200响应
            if response.status_code != 200:
                print(f"[CF错误] 请求失败！状态码：{response.status_code}")
                print(f"[CF错误] 错误响应内容: {response.text[:200]}...")
                return []

            # 解析响应
            print(f"[CF调试] 尝试解析响应JSON...")
            try:
                json_response = response.json()
                print(f"[CF调试] 原始响应JSON:\n{json.dumps(json_response, indent=2, ensure_ascii=False)}")
            except:
                print(f"[CF错误] 响应不是有效JSON！原始响应：{response.text[:200]}...")
                return []

            return self._parse_cf_response(json_response)
            
        except Exception as e:
            print(f"[CF异常] 调用失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _parse_api_response(self, response):
        # 根据实际API响应格式调整
        return [(0, 0, response.get('text', ''))]

    def _cf_worker_transcribe(self, audio_path, language):
        """调用自定义Cloudflare Worker进行转录"""
        print(f"\n[CF Worker调试] 开始处理音频文件: {audio_path}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # 重要修改：直接发送二进制数据
            "Content-Type": "application/octet-stream"  
        }
        
        try:
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
                print(f"[CF Worker调试] 读取到 {len(audio_data)/1024:.2f}KB 音频数据")
    
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=headers,
                data=audio_data  # 直接发送二进制数据
            )
            
            latency = time.time() - start_time
            print(f"[CF Worker调试] 请求耗时: {latency:.2f}s")
            print(f"[CF Worker调试] 响应状态码: {response.status_code}")
    
            if response.status_code != 200:
                print(f"[CF Worker错误] 请求失败: {response.text[:200]}...")
                return []
    
            json_response = response.json()
            print(f"[CF Worker调试] 原始响应:\n{json.dumps(json_response, indent=2, ensure_ascii=False)}")
            
            return self._parse_cf_response(json_response)
            
        except Exception as e:
            print(f"[CF Worker异常] 调用失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _parse_cf_response(self, response):
        try:
            # 检查是否存在vtt字段
            if 'vtt' in response.get('result', {}):
                vtt_content = response['result']['vtt']
                print(f"[CF解析] 检测到VTT格式字幕，长度: {len(vtt_content)}字符")
                return self._parse_vtt_content(vtt_content)
            
            # 兼容旧版逻辑
            if 'text' in response.get('result', {}):
                print("[CF解析] 使用完整文本（无时间戳）")
                return [(0, 0, response['result']['text'])]
            
            print("[CF解析警告] 未找到有效字幕内容")
            return []
        except Exception as e:
            print(f"解析Cloudflare响应失败: {str(e)}")
            return []


    def _parse_vtt_content(self, vtt_content):
        """解析VTT格式内容为时间段"""
        segments = []
        blocks = vtt_content.strip().split('\n\n')
        
        for block in blocks[1:]:  # 跳过开头的WEBVTT行
            lines = block.split('\n')
            if len(lines) < 2:
                continue
                
            # 解析时间轴 例如: "00.580 --> 01.460"
            time_line = lines[0].strip()
            if '-->' not in time_line:
                continue
                
            start_str, end_str = time_line.split('-->')
            start = self._vtt_time_to_seconds(start_str.strip())
            end = self._vtt_time_to_seconds(end_str.strip())
            
            # 合并文本行并清理换行符
            text = ' '.join(line.strip() for line in lines[1:])
            text = text.replace('\n', ' ')
            
            segments.append((start, end, text))
        
        print(f"[VTT解析] 共解析出 {len(segments)} 个字幕片段")
        return segments

    def _vtt_time_to_seconds(self, time_str):
        """将VTT时间格式转换为秒数"""
        try:
            # 处理类似 "00.580" 或 "12.700" 的格式
            if '.' in time_str:
                seconds_part, millis_part = time_str.split('.')
                seconds = float(seconds_part)
                millis = float(millis_part) / 1000.0
                return seconds + millis
            else:
                return float(time_str)
        except Exception as e:
            print(f"时间格式转换失败: {time_str} - {str(e)}")
            return 0
           

def process_file(file_path, translator, bufanyi, whimoxing, yuyan, whi_moshi, whi_url, whi_key):
    print(f"处理文件: {file_path}")
    total_start_time = time.time()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = AudioHandler.prepare_audio(file_path, temp_dir)
        if not audio_path:
            print(f"处理文件失败: {file_path}")
            return
        
        # 初始化转录器
        transcriber = WhisperTranscriber(
            mode=whi_moshi,
            model_name=whimoxing,
            api_url=whi_url,
            api_key=whi_key
        )
        
        try:
            segments = transcriber.transcribe(audio_path, yuyan)
        except Exception as e:
            print(f"语音识别失败: {e}")
            return

        # 生成字幕
        srt_content = ""
        for i, (start, end, text) in enumerate(segments, start=1):
            start_time = f"{int(start // 3600):02d}:{int(start % 3600 // 60):02d}:{int(start % 60):02d},{int(start * 1000 % 1000):03d}"
            end_time = f"{int(end // 3600):02d}:{int(end % 3600 // 60):02d}:{int(end % 60):02d},{int(end * 1000 % 1000):03d}"
            srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

        en_srt_path = file_path.with_suffix('.en.srt')
        with open(en_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        print(f"字幕保存至 {en_srt_path}")

        total_time = time.time() - total_start_time
        print(f"字幕处理耗时 {total_time:.2f} 秒")

        if bufanyi:
            return

        translator.process_srt_file(en_srt_path)

def main():
    parser = argparse.ArgumentParser(description="用whisper提取字幕后自动扔去给ai翻译")
    parser.add_argument("--apikey", default="123", help="openai apikey")
    parser.add_argument("--apiurl", default="https://api.openai.com", help="openai apiurl")
    parser.add_argument("--mulu", default="C:\\fanyi", help="需要处理的目录")
    parser.add_argument("--bufanyi", action="store_true", help="只提取音频，不翻译")
    parser.add_argument("--buyouhua", action="store_true", help="不进行字幕优化")
    parser.add_argument("--whimoxing", default="base.en", help="whisper模型名称")
    parser.add_argument("--moxing", default="gpt-3.5-turbo", help="翻译模型名称")
    parser.add_argument("--yuyan", default="en", help="音频原始语言，可输入zh ja en等，参考此页 https://github.com/openai/whisper/blob/main/whisper/tokenizer.py ")
    parser.add_argument("--whi_moshi", choices=["bendi", "api", "cf_api", "cf_worker", "groq"], default="bendi",
                       help="识别模式: bendi api cf_api cf_worker groq，对应 本地，api，Cloudflare API，Cloudflare Worker，groq")
    parser.add_argument("--whi_url", help="whisper的URL或CF账户ID（Groq模式不需要）")
    parser.add_argument("--whi_key", help="whisper的KEY")
    
    args = parser.parse_args()

    # 参数校验
    if args.whi_moshi in ["api", "cf_api"] and not (args.whi_url and args.whi_key):
        raise ValueError("API模式需要提供whi_url和whi_key参数")
    if args.whi_moshi == "groq" and not args.whi_key:
        raise ValueError("Groq模式需要提供whi_key参数（Groq API Key）")
    if args.whi_moshi == "cf_worker" and not (args.whi_url and args.whi_key):
        raise ValueError("cf_worker模式需要提供whi_url（Worker地址）和whi_key（认证密钥）")

    overall_start_time = time.perf_counter()

    translator = zimufanyi(
        api_key=args.apikey,
        api_url=args.apiurl,
        moxing=args.moxing,
        buyouhua=not args.buyouhua
    )

    mulu = Path(args.mulu)
    for file_path in mulu.glob("*"):
        if file_path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            process_file(
                file_path,
                translator,
                args.bufanyi,
                args.whimoxing,
                args.yuyan,
                args.whi_moshi,
                args.whi_url,
                args.whi_key
            )

    overall_total_time = time.perf_counter() - overall_start_time
    print(f"任务总耗时: {overall_total_time:.2f} 秒")

if __name__ == "__main__":
    main()