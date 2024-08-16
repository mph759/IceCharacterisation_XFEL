import numpy as np
import pandas as pd
from pathlib import Path
import pathlib

def read_reflections(filename: str | Path):
    df = pd.read_csv(filename, sep='\s+')
    df['I'] = df['I'] / 100
    df.drop(columns=['F(real)', 'F(imag)', '|F|', 'ID(λ)', 'M', 'Phase'], inplace=True)
    df.rename(columns={'I': 'Rel. I', 'd(Å)': 'd (A)', '2θ': '2theta'}, inplace=True)
    return df


if __name__ == '__main__':
    print(Path().absolute())
    path_root = Path().absolute()/ r'Pure ice phases'
    ice_ih = read_reflections(path_root / r'Ice Ih' / 'Ice Ih (1538173).txt')
    ice_ic = read_reflections(path_root / r'Ice Ic' / 'Ice Ic (1541503).txt')

    print(f'Ice Ih')
    print(ice_ih.iloc[0:5])
    print(f'Ice Ic')
    print(ice_ic.iloc[0:2])
