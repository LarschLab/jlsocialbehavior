from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "setupAgents.md"
TARGET = ROOT / "coding.md"
START = "<!-- coding.md:start -->"
END = "<!-- coding.md:end -->"


def main():
    text = SOURCE.read_text(encoding="utf-8")
    try:
        body = text.split(START, 1)[1].split(END, 1)[0].strip()
    except IndexError as exc:
        raise SystemExit("coding.md markers not found in setupAgents.md") from exc
    TARGET.write_text(body + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
