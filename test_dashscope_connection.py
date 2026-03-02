# test_dashscope_connection.py
import os
import time
import ssl
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("❌ DASHSCOPE_API_KEY 未设置，请检查 .env 文件")

print("✅ API Key 已加载")

# === 测试 1: 简单 HTTPS 连通性（不调用 API）===
print("\n📡 测试能否访问 dashscope.aliyuncs.com ...")
try:
    import urllib.request
    start = time.time()
    with urllib.request.urlopen("https://dashscope.aliyuncs.com", timeout=10) as response:
        print(f"✅ 能访问主页，状态码: {response.getcode()}，耗时: {time.time() - start:.2f}s")
except Exception as e:
    print(f"❌ 无法访问 DashScope 主站: {e}")
    print("→ 可能原因：网络不通 / 防火墙 / 代理 / DNS 污染")

# === 测试 2: 调用 Embedding API ===
print("\n🧠 测试 Qwen Embedding API 调用...")
try:
    from dashscope import TextEmbedding
    start = time.time()
    response = TextEmbedding.call(
        model="text-embedding-v2",
        input=["你好，世界"],
        api_key=api_key,
        timeout=30  # 显式设置超时
    )
    emb = response.output["embeddings"][0]["embedding"]
    print(f"✅ Embedding 成功！维度: {len(emb)}，耗时: {time.time() - start:.2f}s")
except KeyboardInterrupt:
    print("\n⚠️ 用户中断（Ctrl+C），说明请求卡住了（通常是网络/SSL 问题）")
except Exception as e:
    print(f"❌ API 调用失败: {type(e).__name__}: {e}")
    if "InvalidApiKey" in str(e):
        print("→ API Key 无效，请重新获取")
    elif "SSLError" in str(e) or "certificate" in str(e).lower():
        print("→ SSL 证书验证失败（常见于企业网络）")
    elif "timeout" in str(e).lower():
        print("→ 请求超时，网络不通或被阻断")
    else:
        print("→ 其他错误，请检查网络和权限")