#%%
import pandas as pd
import os
import matplotlib.pyplot as plt

# Define output path
output_dir = os.path.join("..", "..", "output", "figures")
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# Define figure name
fig_name_energy = "car_energy_shares.png"
fig_path_energy = os.path.join(output_dir, fig_name_energy)
fig_name_age = "car_age_shares.png"
fig_path_age = os.path.join(output_dir, fig_name_age)


#%%

# 1. Load the data
pop = pd.read_csv("../../data/raw/Population_EU.csv")
age_car = pd.read_csv("../../data/raw/Age_Car_EU.csv")
energy_car = pd.read_csv("../../data/raw/Energy_Car_EU.csv")

# 2. Drop unnecessary columns
cols_to_keep = ['geo', 'TIME_PERIOD', 'OBS_VALUE']
pop = pop[cols_to_keep].rename(columns={'OBS_VALUE': 'Population'}).rename(columns={'TIME_PERIOD': 'Year'})

age_car = age_car[['geo', 'TIME_PERIOD', 'age_car', 'OBS_VALUE']].rename(columns={'TIME_PERIOD': 'Year'})
energy_car = (energy_car[['geo', 'TIME_PERIOD', 'mot_nrg', 'OBS_VALUE']]).rename(columns={'TIME_PERIOD': 'Year'})

# 3. Remove "TOTAL" category
age_car = age_car[age_car['age_car'] != 'TOTAL']
energy_car = energy_car[energy_car['mot_nrg'] != 'TOTAL']

#Group the categories
category_group_map_age = {
    'Y_LT2': '0-5 years',
    'Y2-5': '0-5 years',
    'Y5-10': '5-20 years',
    'Y10-20': '5-20 years',
    'Y_GT20': '20+ years',
    #'TOTAL': 'TOTAL'
}
category_group_map_energy = {
    'ALT': 'throw',              # Alternative energy
    'BIFUEL': 'Brown',           # Bi-fuel
    'BIODIE': 'Brown',           # Biodiesel
    'BIOETH': 'Brown',           # Bioethanol
    'DIE': 'Brown',              # Diesel
    'DIE_X_HYB': 'throw',        # Diesel (excluding hybrids)
    'ELC': 'Green',              # Electricity
    'ELC_DIE_HYB': 'throw',      # Hybrid diesel-electric
    'ELC_DIE_PI': 'throw',       # Plug-in hybrid diesel-electric
    'ELC_PET_HYB': 'throw',      # Hybrid petrol-electric
    'ELC_PET_PI': 'throw',       # Plug-in hybrid petrol-electric
    'GAS': 'Brown',              # Natural gas
    'HYD_FCELL': 'Green',        # Hydrogen and fuel cells
    'LPG': 'Brown',              # Liquefied petroleum gas
    'OTH': 'Brown',              # Other
    'PET': 'Brown',              # Petrol (excluding hybrids)
    'PET_X_HYB': 'throw',        # Hybrid electric-petrol
    #'TOTAL': 'TOTAL'
}

# Step 2: Map to grouped categories
age_car['age_car'] = age_car['age_car'].map(category_group_map_age)
# yy = age_car.groupby(['geo', 'Year', 'age_car'], as_index=False)['OBS_VALUE'].sum()
# yy[(yy['geo'] == 'FR') & (yy['Year'] == 2022)]
age_car = age_car[age_car['age_car'] != 'throw']


energy_car['mot_nrg'] = energy_car['mot_nrg'].map(category_group_map_energy)
# xx = energy_car.groupby(['geo', 'Year', 'mot_nrg'], as_index=False)['OBS_VALUE'].sum()
# xx[(xx['geo'] == 'FR') & (xx['Year'] == 2022)]
energy_car = energy_car[energy_car['mot_nrg'] != 'throw']



# xx_pivot = xx.pivot_table(index=['geo', 'Year'],
#                           columns='mot_nrg',
#                           values='OBS_VALUE',
#                           fill_value=0).reset_index()
# xx_pivot['Brown_Green'] = xx_pivot['Brown'] + xx_pivot['Green']
# xx_pivot['Difference'] = xx_pivot['TOTAL'] - xx_pivot['Brown_Green']
# xx_pivot[(xx_pivot['geo'] == 'UK') & (xx_pivot['Year'] == 2022)]


# 4. Pivot to get car categories as columns
age_car_pivot = age_car.pivot_table(index=['geo', 'Year'],
                                    columns='age_car',
                                    values='OBS_VALUE',
                                    aggfunc='sum').reset_index().fillna(0)

energy_car_pivot = energy_car.pivot_table(index=['geo', 'Year'],
                                          columns='mot_nrg',
                                          values='OBS_VALUE',
                                          aggfunc='sum').reset_index().fillna(0)

# 5. Compute total number of cars per country/year
age_car_pivot['total_cars'] = age_car_pivot.drop(columns=['geo', 'Year']).sum(axis=1)
energy_car_pivot['total_cars'] = energy_car_pivot.drop(columns=['geo', 'Year']).sum(axis=1)

# 6. Compute proportion of cars by category
age_cols = age_car_pivot.columns.difference(['geo', 'Year', 'total_cars'])
for col in age_cols:
    age_car_pivot[col] = age_car_pivot[col] / age_car_pivot['total_cars']

energy_cols = energy_car_pivot.columns.difference(['geo', 'Year', 'total_cars'])
for col in energy_cols:
    energy_car_pivot[col] = energy_car_pivot[col] / energy_car_pivot['total_cars']

# 7. Merge with population
age_merged = pd.merge(age_car_pivot, pop, on=['geo', 'Year'], how='left')
age_merged = age_merged[(age_merged['Year'] >= 2014) & (age_merged['Year'] <= 2023) & (age_merged['geo'] != 'EU27_2020')]
energy_merged = pd.merge(energy_car_pivot, pop, on=['geo', 'Year'], how='left')
energy_merged = energy_merged[(energy_merged['Year'] >= 2014) & (energy_merged['Year'] <= 2023) & (energy_merged['geo'] != 'EU27_2020')]


# 8. Compute weighted and unweighted averages
def compute_weighted_unweighted(df, value_cols):
    # Unweighted average (mean across countries)
    unweighted = df.groupby('Year')[value_cols].mean().add_suffix('_unweighted')

    # Weighted average (population-weighted)
    weighted = df.groupby('Year').apply(
        lambda x: (x[value_cols].multiply(x['Population'], axis=0)).sum() / x['Population'].sum()
    ).add_suffix('_weighted')

    return pd.concat([unweighted, weighted], axis=1).reset_index()
    #return pd.concat([unweighted], axis=1).reset_index()

# Apply to age categories
age_avg = compute_weighted_unweighted(age_merged, list(age_cols))

# Apply to motor categories
energy_avg = compute_weighted_unweighted(energy_merged, list(energy_cols))


# %%

proportion_cols = [col for col in age_avg.columns if col != 'Year' and col.endswith('_weighted')]  # or '_unweighted' if you prefer
plt.figure(figsize=(6, 3))
plt.stackplot(age_avg['Year'],
              [age_avg[col] for col in proportion_cols],
              labels=[col.replace('_weighted', '') for col in proportion_cols])


plt.title("Evolution of Car Age Category Shares (Population-weighted)")
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(fname = fig_path_age, dpi=300)
plt.show()

# %%
proportion_cols = [col for col in age_avg.columns if col != 'Year' and col.endswith('_unweighted')]  # or '_unweighted' if you prefer
plt.figure(figsize=(6, 3))
plt.stackplot(age_avg['Year'],
              [age_avg[col] for col in proportion_cols],
              labels=[col.replace('_unweighted', '') for col in proportion_cols])

plt.title("Evolution of Car Age Category Shares (unweighted)")
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()



# %%
proportion_cols = [col for col in energy_avg.columns if col != 'Year' and col.endswith('_weighted')]  # or '_unweighted' if you prefer
plt.figure(figsize=(6, 3))
plt.stackplot(energy_avg['Year'],
              [energy_avg[col] for col in proportion_cols],
              labels=[col.replace('_weighted', '') for col in proportion_cols])

plt.title("Evolution of Car Energy Category Shares (Population-weighted)")
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(fname = fig_path_energy, dpi=300)
plt.show()

# %%
proportion_cols = [col for col in energy_avg.columns if col != 'Year' and col.endswith('_unweighted')]  # or '_unweighted' if you prefer
plt.figure(figsize=(6, 3))
plt.stackplot(energy_avg['Year'],
              [energy_avg[col] for col in proportion_cols],
              labels=[col.replace('_unweighted', '') for col in proportion_cols])

plt.title("Evolution of Car Energy Category Shares (unweighted)")
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Export the calibration tables.

# Extract and round 2023 data
energy_2023 = energy_avg[energy_avg['Year'] == 2023].round(4)
age_2023 = age_avg[age_avg['Year'] == 2023].round(4)

# Reorder age columns as requested
# Make sure these columns exist exactly with these names, otherwise adjust!
desired_order = ['Year', '0-5 years_weighted', '5-20 years_weighted', '20+ years_weighted', '0-5 years_unweighted', '5-20 years_unweighted', '20+ years_unweighted']
age_2023 = age_2023[desired_order]
desired_order = ['Year','Green_weighted', 'Brown_weighted', 'Green_unweighted', 'Brown_unweighted']
energy_2023 = energy_2023[desired_order]

# Prepare output folder
output_dir = os.path.join("..", "..", "output", "tables")
os.makedirs(output_dir, exist_ok=True)

# File paths
energy_path = os.path.join(output_dir, "calibration_energy_2023.csv")
age_path = os.path.join(output_dir, "calibration_age_2023.csv")

# Export separately
energy_2023.to_csv(energy_path, index=False)
age_2023.to_csv(age_path, index=False)

print(f"Energy calibration table saved to: {os.path.abspath(energy_path)}")
print(f"Age calibration table saved to: {os.path.abspath(age_path)}")
# %% Calibration SS shares
energy = energy_2023[['Year'] + [c for c in energy_2023 if '_unweighted' in c]]
energy.columns = ['Year'] + [c.replace('_unweighted', '') for c in energy.columns[1:]]

age = age_2023[['Year'] + [c for c in age_2023 if '_unweighted' in c]]
age.columns = ['Year'] + [c.replace('_unweighted', '') for c in age.columns[1:]]

calibration = pd.DataFrame({
    'Year': energy['Year'],
    **{f'{ec} x {ac}': energy[ec] * age[ac]
       for ec in energy.columns[1:] for ac in age.columns[1:]}
}).round(4)

#Add the none fraction.
Fraction_none = 0.28 #Belgium: https://www.eea.europa.eu/publications/ENVISSUENo12/page018.html?utm_source=chatgpt.com
calibration.insert(1, 'None', Fraction_none)
calibration.iloc[:, 2:] = calibration.iloc[:, 2:].div(calibration.iloc[:, 2:].sum(axis=1), axis=0) * (1 - Fraction_none)


# File paths
calibration_path = os.path.join(output_dir, "calibration_2023.csv")

# Export separately
calibration.drop(columns=['Year']).to_csv(calibration_path, index=False)

# %%
