# whisper_fanyi
用whisper提取字幕后自动扔去给ai翻译，可批处理整个资料夹，并且会做简单的字幕合并处理

## 项目介绍

该项目由两个主要组件组成：

1. **whisper.py**：负责从音视频文件中提取字幕。它支持多种语音识别模式，包括本地处理、API调用和Cloudflare服务。
2. **zimufanyi.py**：负责将提取的英文字幕翻译成中文。可以使用OpenAI格式的API进行翻译，并提供了字幕优化选项。

主要功能包括：

- 从音视频文件中提取字幕
- 将字幕翻译成中文
- 字幕优化（可选，合并短句，目前只能处理英文）
- 支持批量处理文件

## 使用方式

1. 克隆项目到本地：

   ```bash
   git clone https://github.com/abc123sm/whisper_fanyi.git
   cd whisper_fanyi
   ```

2. 安装依赖：

   ```bash
   pip install groq faster-whisper openai
   ```

3. 运行脚本处理文件：

   ```bash
   python whisper.py --mulu /path/to/your/media/files
   ```

   或者，如果只想提取字幕而不翻译：

   ```bash
   python whisper.py --mulu /path/to/your/media/files --bufanyi
   ```

## 示例命令

- 处理目录中的所有音视频文件：

  ```bash
  python whisper.py --mulu C:\fanyi
  ```

- 只提取字幕，不进行翻译：

  ```bash
  python whisper.py --mulu C:\fanyi --bufanyi
  ```

- 使用特定的Whisper模型进行本地处理：

  ```bash
  python whisper.py --mulu C:\fanyi --whimoxing large --whi_moshi bendi
  ```

- 使用特定的模型进行翻译处理：

  ```bash
  python whisper.py --mulu C:\fanyi --moxing grok-2-1212 --apiurl "https://api.x.ai" --apikey xai-123456789 --whi_moshi bendi --whimoxing large
  ```

- 使用OpenAI API进行字幕翻译：

  ```bash
  python whisper.py --mulu C:\fanyi --apikey your_openai_api_key --apiurl https://api.openai.com
  ```

- 使用Groq API进行语音识别：

  ```bash
  python whisper.py --mulu C:\fanyi --whi_moshi groq --whi_key groq_api_key
  ```

- 禁用字幕优化：

  ```bash
  python whisper.py --mulu C:\fanyi --buyouhua
  ```

- 使用cf worker，并使用老马的grok2翻译:

  ```bash
  python whisper.py --whi_moshi cf_worker --whi_url "https://XXX.workers.dev/" --whi_key XXXX --moxing grok-2-1212 --apiurl "https://api.x.ai" --apikey xai-123456789 --mulu "C:\fanyi"

## 配置选项

- `--apikey`：OpenAI格式的API密钥（用于字幕翻译）
- `--apiurl`：OpenAI格式的APIURL（用于字幕翻译，默认`https://api.openai.com`）
- `--mulu`：要处理的目录路径（默认为`C:\fanyi`）
- `--bufanyi`：只提取字幕，不进行翻译
- `--buyouhua`：不进行字幕优化
- `--whimoxing`：Whisper模型名称（默认为`base.en`）
- `--moxing`：翻译模型名称（默认为`gpt-3.5-turbo`）[参考https://github.com/openai/whisper](https://github.com/openai/whisper)
- `--yuyan`：音频原始语言（默认为`en`）[参考tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)  
- `--whi_moshi`：识别模式（`bendi`、`api`、`cf_api`、`cf_worker`、`groq`）
- `--whi_url`：Whisper API或Cloudflare Worker的URL
- `--whi_key`：Whisper API或Cloudflare Worker的密钥

## CF worker

  ```js
// worker.js
export default {
  async fetch(request, env) {
    // 允许CORS
    const headers = new Headers({
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization"
    });

    // 处理预检请求
    if (request.method === "OPTIONS") {
      return new Response(null, { headers });
    }

    // 只处理POST请求
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405, headers });
    }

    try {
      // 从请求体获取二进制音频数据
      const audioBlob = await request.arrayBuffer();
      
      // 调用AI模型
      const result = await env.AI.run(
        '@cf/openai/whisper',
        { audio: [...new Uint8Array(audioBlob)] }
      );

      return Response.json({ result }, { headers });
    } catch (error) {
      return Response.json(
        { error: error.message },
        { status: 500, headers }
      );
    }
  }
};
```

## 注意事项

- 确保您有足够的API配额，特别是在处理大量文件时。
- 对于本地模式，请确保您的设备有足够的计算资源。
- 翻译结果可能会因API模型的不同而有所变化。
- 觉得写一堆麻烦你就自己改改，直接把api啥的都写进代码里
