import praw
import requests
import os
from PIL import Image
from io import BytesIO
import hashlib
import time
from requests.exceptions import RequestException

# 初始化 Reddit 客户端
reddit = praw.Reddit(
    user_agent="BulkOrCut Image Downloader",
    client_id="K6XHR17E0SPZEtWSSyiktg",
    client_secret="ryrSu7NJfQZ8ew0UMCvpIoCW9_Qvmg",
    username="Ok_Confidence_4332",
    password="zyt20030503"
)

# 设置超时时间
reddit._core._requestor._http.timeout = 60

# 指定子版块名称
subreddit = reddit.subreddit("SkinnyWithAbs")



# 创建用于保存图片的文件夹
output_dir = "images_from_SkinnyWithAbs"
os.makedirs(output_dir, exist_ok=True)

# 支持的图片扩展名
valid_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".ico", ".heic")

# 初始化计数器和已下载的文件名集合
def initialize_counter(directory):
    return len([f for f in os.listdir(directory) if f.endswith(".png")])

def get_existing_urls(directory):
    existing_files = set()
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            existing_files.add(filename)
    return existing_files

image_counter = initialize_counter(output_dir)
downloaded_urls = get_existing_urls(output_dir)
print(f"已下载的图片数量: {image_counter}")

# 图片下载函数
def download_image_with_fallback(image_url, fallback_url=None, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            if response.status_code == 200:
                return response
        except RequestException as e:
            print(f"尝试 {attempt + 1}/{retries} 失败: {e}")
        time.sleep(2)
    if fallback_url:
        print(f"主链接失败，尝试备用链接: {fallback_url}")
        try:
            response = requests.get(fallback_url, stream=True, timeout=10)
            if response.status_code == 200:
                return response
        except RequestException as e:
            print(f"备用链接失败: {e}")
    return None

# 开始下载图片
print("开始提取图片并转换格式...")

try:
    for submission in subreddit.new(limit=None):
        # 检查 submission 是否包含 media_metadata，并确保它是字典类型
        if hasattr(submission, "media_metadata") and isinstance(submission.media_metadata, dict):
            print(f"相册帖子: {submission.url}")
            for media_id, media_data in submission.media_metadata.items():
                image_url = None
                fallback_url = None
                
                # 检查 media_data 是否包含 's' 键并且 'u' 键存在
                if "s" in media_data and "u" in media_data["s"]:
                    image_url = media_data["s"]["u"].replace("&amp;", "&")
                    if "i.redd.it" in image_url:
                        fallback_url = image_url
                
                # 如果 's' 不存在，检查 'p' 键是否存在
                elif "p" in media_data and isinstance(media_data["p"], list) and media_data["p"]:
                    image_url = media_data["p"][-1]["u"].replace("&amp;", "&")
                    if "i.redd.it" in image_url:
                        fallback_url = image_url

                if not image_url:
                    continue

                if image_url not in downloaded_urls:
                    output_file = os.path.join(output_dir, f"{image_counter}.png")
                    downloaded_urls.add(image_url)
                    response = download_image_with_fallback(image_url, fallback_url)
                    if response:
                        try:
                            image = Image.open(BytesIO(response.content))
                            image.convert("RGB").save(output_file, "PNG")
                            print(f"图片已保存并转换为 PNG: {output_file}")
                            image_counter += 1
                        except Exception as e:
                            print(f"图片处理错误: {e}")
        # 处理直接包含图片 URL 的帖子
        elif submission.url.lower().endswith(valid_extensions):
            if submission.url not in downloaded_urls:
                output_file = os.path.join(output_dir, f"{image_counter}.png")
                downloaded_urls.add(submission.url)
                response = download_image_with_fallback(submission.url)
                if response:
                    try:
                        image = Image.open(BytesIO(response.content))
                        image.convert("RGB").save(output_file, "PNG")
                        print(f"图片已保存并转换为 PNG: {output_file}")
                        image_counter += 1
                    except Exception as e:
                        print(f"图片处理错误: {e}")
except Exception as e:
    print(f"网络请求或 Reddit API 错误: {e}")

print(f"图片提取与格式转换完成，共下载了 {image_counter} 张图片！")
