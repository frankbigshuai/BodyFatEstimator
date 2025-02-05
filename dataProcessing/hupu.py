import requests
from bs4 import BeautifulSoup
import os
import time
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def setup_session():
    """设置请求会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def save_checkpoint(page, cursor):
    """保存断点信息"""
    with open("checkpoint.txt", "w") as f:
        f.write(f"{page},{cursor}")

def load_checkpoint():
    """加载断点信息"""
    if os.path.exists("checkpoint.txt"):
        with open("checkpoint.txt", "r") as f:
            page, cursor = f.read().strip().split(",")
            return int(page), cursor
    return 1, None

def log_error(message):
    """记录错误日志"""
    with open("error_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def download_images_from_topic(session, url, output_folder):
    """从单个帖子下载图片"""
    image_counter = 0
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all(attrs={'data-origin': True})

        print(f"在帖子中找到 {len(img_tags)} 个图片")

        for img_tag in img_tags:
            img_url = img_tag['data-origin']
            if img_url.startswith("http"):
                try:
                    print(f"下载图片: {img_url}")
                    img_response = session.get(img_url)

                    if img_response.status_code == 200:
                        existing_images = len([name for name in os.listdir(output_folder) if name.endswith('.jpg')])
                        img_path = os.path.join(output_folder, f'{existing_images + 1}.jpg')

                        with open(img_path, 'wb') as f:
                            f.write(img_response.content)
                        print(f"✓ 保存图片: {img_path}")
                        image_counter += 1
                    else:
                        print(f"× 图片下载失败: {img_url}")

                except Exception as e:
                    print(f"× 处理图片时出错: {e}")

                time.sleep(random.uniform(0.5, 1))

    except Exception as e:
        print(f"× 处理帖子失败: {e}")
        log_error(f"Thread URL {url}: {e}")

    return image_counter

def search_and_download(session, keyword="体脂", topic_id=23, output_folder="images"):
    """边搜索边下载"""
    print("\n=== 开始搜索并下载 ===")
    os.makedirs(output_folder, exist_ok=True)
    print(f"创建/确认图片保存文件夹: {output_folder}")
    
    base_url = "https://m.hupu.com/api/v2/bbs/topicThreads"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    
    total_posts = 0
    total_images = 0
    page, cursor = load_checkpoint()
    
    while page <= 100000:  # 限制最大页数为 500
        try:
            print(f"\n正在搜索第 {page} 页...")
            params = {"topicId": topic_id, "page": page}
            if cursor:
                params["cursor"] = cursor
            
            response = session.get(base_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # 检查是否有数据返回
            if not data or "data" not in data or not data["data"]:
                print(f"第 {page} 页无有效数据，跳过...")
                break
            
            threads = data["data"].get("topicThreads", [])
            next_cursor = data["data"].get("nextCursor")
            
            # 检查 threads 是否为空
            if threads is None:
                print(f"第 {page} 页的帖子数据为空，跳过...")
                break
            
            for thread in threads:
                if keyword in thread["title"]:
                    print(f"\n找到相关帖子: {thread['title']}")
                    total_posts += 1
                    images_downloaded = download_images_from_topic(session, thread["url"], output_folder)
                    total_images += images_downloaded
                    print(f"从该帖子下载了 {images_downloaded} 张图片")
            
            if not next_cursor or not threads:
                print("\n没有更多页面，搜索完成")
                break
                
            cursor = next_cursor
            page += 1
            save_checkpoint(page, cursor)
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"搜索过程出错: {e}")
            log_error(f"Page {page}: {e}")
            break
    
    return total_posts, total_images

def main():
    print("开始执行虎扑图片下载程序...")

    session = setup_session()
    output_folder = "images1"

    total_posts, total_images = search_and_download(session, output_folder=output_folder)

    print("\n=== 执行完成 ===")
    print(f"共处理了 {total_posts} 个帖子")
    print(f"下载了 {total_images} 张图片")
    print(f"图片保存在: {os.path.abspath(output_folder)} 文件夹中")

if __name__ == "__main__":
    main()
