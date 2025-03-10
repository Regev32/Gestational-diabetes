import pandas as pd


def xlsx_to_csv(xlsx_file: str, csv_file: str) -> None:
    df = pd.read_excel(xlsx_file)
    df.to_csv(csv_file, index=False)

xlsx_to_csv('../data/GDM_master.xlsx', '../data/filled_avg/gdm_master.csv')
