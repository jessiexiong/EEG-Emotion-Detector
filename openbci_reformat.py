import numpy as np
import pandas as pd
import os

def main():
    for x in os.listdir("D:/EEGDecoder/bcidata0210/test"):
        df = pd.read_csv("D:/EEGDecoder/bcidata0210/test/" + x, names=["Header"])
        df = df['Header'].str.split('\t', expand=True)
        df = df.iloc[:, 0:5]
        df.columns = ["Timestamp", "AF8", "AF7", "TP9", "TP10"]

        save_filepath = 'D:/EEGDecoder/bcidata0210/preprocessed/'
        os.makedirs(save_filepath, exist_ok=True)
        df.to_csv(save_filepath+x, index = False)


if __name__ == '__main__':
    main()