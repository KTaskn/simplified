import json
import argparse

# compare the json files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two JSON files and find differences.")
    parser.add_argument("original", type=str, help="Path to the first JSON file.")
    parser.add_argument("simplified", type=str, help="Path to the second JSON file.")
    args = parser.parse_args()

    with open(args.original, "r", encoding="utf-8") as f1:
        original = json.load(f1)

    with open(args.simplified, "r", encoding="utf-8") as f2:
        simplified = json.load(f2)

    differences = []
    dict_original ={
        row["id"]: row["caption"]
        for row in original["annotations"]
    }
    print(dict_original)
    dict_simplified = {
        row["id"]: row["source_caption"]
        for row in simplified["annotations"]
    }
    print(dict_simplified)
    for key in dict_original:
        if key in dict_simplified:
            if dict_original[key] != dict_simplified[key]:
                differences.append((key, dict_original[key], dict_simplified[key]))
        else:
            differences.append((key, dict_original[key], None))
    print(f"Found {len(differences)} differences:")