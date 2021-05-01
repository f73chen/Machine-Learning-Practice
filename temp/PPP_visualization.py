import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Imports csv and selects the columns of interest
# Note: data are already of type float
df = pd.read_csv('PPP_Cares_Act_Loan_Totals_to_New_Jersey_Businesses.csv')
df = df[["NAICS Code", "Jobs Retained"]]

# Remove rows with empty data
df.dropna(inplace = True)

# Make sorted list of unique NAICS codes
codes = sorted(list(set(np.array(df["NAICS Code"]))))
jobs = []

# Sum jobs retained for each job code
for code in codes:
    total = df.loc[df["NAICS Code"] == code, "Jobs Retained"].sum()
    jobs.append(total)

# Make ticks evenly spaced despite their values
x_pos = np.arange(len(codes))

# Add chart labels
plt.bar(x_pos, jobs)
plt.title("Jobs Retained Per NAICS Code")
plt.xlabel("NAICS Code")
plt.ylabel("Jobs Retained")

plt.xticks(x_pos, codes)
plt.show()