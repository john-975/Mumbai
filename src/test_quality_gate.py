import json
import sys
from pathlib import Path

from quality_gate import check_image_file


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python src/test_quality_gate.py "
            "dataset/raw/drone dataset/IMG20261101111216.jpg"
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])

    result = check_image_file(image_path)

    print(json.dumps(result, indent=4))

    if result["status"] == "READY":
        print("\nImage passed the preliminary quality checks.")
    else:
        print("\nImage cannot yet be used for reliable inspection.")

        for reason in result["reasons"]:
            print(f"- {reason}")


if __name__ == "__main__":
    main()