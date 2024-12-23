import psutil
import GPUtil
import time

def monitor_resources():
    while True:
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # GPU usage (if available)
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load if gpus else "N/A"
        
        print(f"CPU: {cpu_percent}% | Memory: {memory.percent}% | GPU: {gpu_usage}")
        time.sleep(5)

if __name__ == "__main__":
    monitor_resources() 