"""
plot_results.py
───────────────
$ python plot_results.py path/to/log.csv
"""

import argparse, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="results log produced by train_sft.py")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax2 = ax1.twinx()

    for phase, color in [("train","tab:blue"), ("val","tab:orange")]:
        sub = df[df.phase==phase]
        ax1.plot(sub.step, sub.loss, label=f"{phase}-loss", color=color, ls="-")
        ax2.plot(sub.step, sub.ppl, label=f"{phase}-ppl", color=color, ls="--")

    ax1.set_xlabel("update step"); ax1.set_ylabel("loss")
    ax2.set_ylabel("perplexity")
    ax1.legend(loc="upper left");  ax2.legend(loc="upper right")

    out_png = Path(args.csv).with_suffix(".png")
    plt.tight_layout(); plt.savefig(out_png, dpi=150)
    print(f"plot saved to {out_png}")

if __name__ == "__main__":
    main()
