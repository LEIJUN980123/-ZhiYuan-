# qwen_client.py —— 原始风格 + 关键参数修复
import os
import time
import logging
from typing import Optional, Generator
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError, AuthenticationError

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QwenClient")

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    timeout=30
)

def call_qwen(prompt: str, model: str = "qwen-turbo", max_retries: int = 3, retry_delay: float = 1.0) -> str:
    if not client.api_key:
        return "❌ 错误：未配置 DASHSCOPE_API_KEY，请检查 .env 文件"
    if not isinstance(prompt, str):
        logger.error(f"❌ prompt 类型错误：期望 str，实际为 {type(prompt)}，值: {repr(prompt)[:100]}")
        return "❌ 内部错误：prompt 必须是字符串"

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"🚀 调用 Qwen (尝试 {attempt + 1}/{max_retries + 1})")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500  # ← 关键修复
            )
            result = response.choices[0].message.content.strip()
            logger.info("✅ Qwen 调用成功")
            return result

        except AuthenticationError:
            return "❌ 认证失败 (401): API Key 无效或缺失。请检查 .env 中的 DASHSCOPE_API_KEY"
        except RateLimitError:
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"⚠️ 触发限流 (429)，{wait_time:.1f} 秒后重试...")
            if attempt < max_retries:
                time.sleep(wait_time)
            else:
                return "❌ 限流错误 (429): 已达最大重试次数。建议降低调用频率。"
        except APIStatusError as e:
            if e.status_code == 403:
                return f"❌ 权限错误 (403): 可能未开通 {model} 模型或余额不足。"
            elif e.status_code >= 500:
                logger.error(f"💥 服务端错误 ({e.status_code}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    return f"❌ 服务端错误 ({e.status_code}): 请稍后再试。"
            else:
                return f"❌ 请求错误 ({e.status_code}): {e.message}"
        except (APITimeoutError, APIConnectionError) as e:
            logger.error(f"🌐 网络错误: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return "❌ 网络超时：请检查网络连接或重试。"
        except Exception as e:
            logger.exception("🔥 未知错误")
            return f"❌ 未知错误: {str(e)}"
    return "❌ 调用失败：达到最大重试次数"


def stream_call_qwen(prompt: str, model: str = "qwen-turbo") -> Generator[str, None, None]:
    if not client.api_key:
        yield "❌ 错误：未配置 DASHSCOPE_API_KEY，请检查 .env 文件"
        return
    if not isinstance(prompt, str):
        yield "❌ 内部错误：prompt 必须是字符串"
        return

    try:
        logger.info("🚀 流式调用 Qwen")
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,      # ← 防止截断
            stream=True
        )
        for chunk in stream:
            content = getattr(chunk.choices[0].delta, 'content', None)
            if content:
                yield content
        logger.info("✅ 流式调用完成")
    except Exception as e:
        logger.exception("🔥 流式调用异常")
        yield f"\n\n❌ 流式生成出错: {str(e)}"