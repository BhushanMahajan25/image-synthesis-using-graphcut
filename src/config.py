import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
INPUT_DIR = os.path.join(ROOT_DIR, "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
INTERMEDIATE_DIR = os.path.join(ROOT_DIR, "intermediate")
OUT_DIM = (512,512)