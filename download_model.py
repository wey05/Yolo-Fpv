import urllib.request
import sys

def download_model():
    urls = [
        "https://github.com.cnpmjs.org/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "https://hub.fastgit.org/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    ]
    
    print("正在尝试下载YOLOv8n模型文件...")
    
    for url in urls:
        try:
            print(f"\n尝试从: {url}")
            urllib.request.urlretrieve(url, "yolov8n.pt")
            print("✓ 下载成功！")
            return True
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            continue
    
    print("\n所有镜像均下载失败，请手动下载：")
    print("1. 访问: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
    print("2. 下载后保存到当前目录")
    return False

if __name__ == "__main__":
    download_model()
    input("\n按回车键退出...")
