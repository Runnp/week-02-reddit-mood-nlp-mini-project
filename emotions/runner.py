import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emotion_tool.analyzer import analyze
from emotion_tool.charts import full_dashboard

def run_cli():
    print("\n=== Emotion Illustration Tool ===")
    print("Type or paste your text below.")
    print("Press Enter twice when done.\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    text = " ".join(lines).strip()
    if not text:
        print("No text entered. Exiting.")
        return

    print("\nAnalysing...")
    result = analyze(text)

    print(f"\nOverall mood  : {result['overall_mood'].upper()}")
    print(f"Dominant emotion: {result['dominant'].upper()}")
    print(f"Word count    : {result['word_count']}")
    print(f"Sentences     : {len(result['sentences'])}")
    print(f"\nVADER scores  : {result['vader']}")
    print("\nEmotion scores:")
    for emotion, score in result["emotions"].items():
        bar = "█" * int(score * 200)
        print(f"  {emotion:12} {score:.4f}  {bar}")

    print("\nGenerating charts...")
    os.makedirs("outputs", exist_ok=True)
    fig = full_dashboard(result, save_path="outputs/emotion_dashboard.png")
    plt.show()
    print("\nDone.")

if __name__ == "__main__":
    run_cli()