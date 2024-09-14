import pandas as pd
import argparse

def check_output_format(output_file):
    df = pd.read_csv(output_file)
    if 'index' not in df.columns or 'prediction' not in df.columns:
        print("Output file must contain 'index' and 'prediction' columns.")
        return False

    for index, prediction in zip(df['index'], df['prediction']):
        if not isinstance(index, int) or (not isinstance(prediction, str) and prediction != ''):
            print(f"Invalid entry at index {index}: {prediction}")
            return False

    print(f"Parsing successful for file: {output_file}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_filename', required=True, help='Path to the test CSV file')
    parser.add_argument('--output_filename', required=True, help='Path to the output CSV file')
    args = parser.parse_args()

    check_output_format(args.output_filename)