import argparse
from se_unlearning.data.make_splits import prepare

def main():
    p = argparse.ArgumentParser(description="Validate JSONL and write split files.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(prepare(args.input, args.output))

if __name__ == "__main__":
    main()
