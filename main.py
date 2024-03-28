import sys
from iopaint import entry_point

if __name__ == "__main__":
    # 设置你想要传递的参数
    sys.argv = ['main.py', 'start', '--enable-interactive-seg', '--interactive-seg-device=cuda', '--port=8081']

    # 调用主函数
    entry_point()
