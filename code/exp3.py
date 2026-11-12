"""实验三运行入口；计算模块保存在共用程序。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from pipeline import cli

if __name__=='__main__':cli('3')
