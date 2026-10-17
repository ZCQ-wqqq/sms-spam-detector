import argparse
import csv
import re

import pandas as pd

RAW_PATH = "data/raw/SMSSpamCollection"
OUT_PATH = "data/sms.csv"


def load_raw(path=RAW_PATH):
    # each line is "label<TAB>message", some messages contain quotes so quoting is turned off
    return pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_v1(df):
    # v1 = the raw data as downloaded
    return df.copy()


def make_v2(df):
    # v2 = normalized text, empty messages and duplicates removed
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="text")
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    args = parser.parse_args()

    df = load_raw()
    if args.version == "v1":
        data = make_v1(df)
    else:
        data = make_v2(df)

    data.to_csv(OUT_PATH, index=False)
    print(f"{args.version}: saved {len(data)} rows to {OUT_PATH}")
    print(data["label"].value_counts())


if __name__ == "__main__":
    main()
