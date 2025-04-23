# File: main.py
import os

# Disable efficient attention mechanism
os.environ['USE_FLASH_ATTENTION'] = '0'

from examples.example_usage import example_usage

if __name__ == '__main__':
    example_usage()
