#!/usr/bin/env python

"""
main.py

This script runs all the models located under the "models/" folder. It assumes you have:
  - models/GDM.py
  - models/PEPH.py
  - models/GDM_drugs.py
  - models/GenericOutcome.py

Each module must have a main(...) function that accepts:
  - all_data (bool): whether to include CRL-related columns.
  - data_path (str): path to the CSV file.
  - For GenericOutcome.py, an additional parameter target_col is required.

The script runs each model with both all_data=False and all_data=True.
Results (ROC plots and model pickle files) are saved under the appropriate folders.
"""

# Import model modules from the models folder.
from models import GDM, PEPH, GDM_drugs, other, GDM_GA
from superplot import generate_superplot

def run_all_models(data_master, data_4000):
    for all_data in [True, False]:
        # Run GDM model
        print(f"Running GDM model with data before {'and at the beginning of' if all_data else 'the'} pregnancy...")
        GDM.main(all_data=all_data, data_path=data_master)


        # Run PE/PH model
        print(f"Running PE/PH model with data before {'and at the beginning of' if all_data else 'the'} pregnancy...")
        PEPH.main(all_data=all_data, data_path=data_master)


        # Run GDM drugs model
        print(f"Running GDM drugs model with data before {'and at the beginning of' if all_data else 'the'} pregnancy...")
        GDM_drugs.main(all_data=all_data, data_path=data_4000)

        # Run GDM GA model
        print(f"Running GDM GA model with data before {'and at the beginning of' if all_data else 'the'} pregnancy...")
        GDM_GA.main(all_data=all_data, data_path=data_4000)


        # Run generic outcome model for multiple targets:
        for outcome in ["BW", "BW %", "out ga"]:
            print(f"Running {outcome} model with data before {'and at the beginning of' if all_data else 'the'} pregnancy...")
            other.main(all_data=all_data, data_path=data_master, target_col=outcome)

if __name__ == "__main__":
    data_path = "data/gdm_master.csv"
    other_path = "data/gdm_4000.csv"
    run_all_models(data_path, other_path)
