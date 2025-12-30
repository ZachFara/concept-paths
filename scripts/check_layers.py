import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capture import GPT2


def main():
    gpt = GPT2()
    num_layers = len(gpt.LLM.transformer.h)
    print(f"num_layers: {num_layers}")
    print(f"indices: {list(range(num_layers))}")


if __name__ == "__main__":
    main()
