import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = "dataset2/WDICSV.csv"
pivoted_data = 'dataset2/PivotedDataset.csv'

def getDataset(data):
    dataset = pd.read_csv(data)
    return dataset

INDICATOR_CODES = ["AG.LND.ARBL.ZS", "FB.BNK.CAPA.ZS", "GC.DOD.TOTL.GD.ZS", "SE.TER.CUAT.BA.ZS", "SL.EMP.TOTL.SP.ZS",
                   "GC.XPN.TOTL.GD.ZS", "BX.KLT.DINV.WD.GD.ZS", "AG.LND.FRST.ZS", "NY.GDP.PCAP.KD.ZG", "NY.GNS.ICTR.ZS",
                   "NY.GNS.ICTR.GN.ZS", "IT.NET.USER.ZS", "ST.INT.ARVL", "SL.TLF.TOTL.IN", "SP.DYN.LE00.IN",
                   "SE.ADT.LITR.ZS", "SP.POP.GROW", "SP.POP.TOTL.FE.IN", "SP.POP.TOTL.MA.IN", "AG.LND.TOTL.RU.K2",
                   "SP.RUR.TOTL.ZS", "NE.TRD.GNFS.ZS", "BG.GSR.NFSV.GD.ZS", "SL.UEM.TOTL.NE.ZS"]

featureMap = {
    "AG.LND.ARBL.ZS": "Arable land",
    "FB.BNK.CAPA.ZS": "Bank capital to assets ratio",
    "GC.DOD.TOTL.GD.ZS": "Central government debt",
    "SE.TER.CUAT.BA.ZS": "Educational attainment",
    "SL.EMP.TOTL.SP.ZS": "Employment to population ratio",
    "GC.XPN.TOTL.GD.ZS": "Expense (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment (% of GDP)",
    "AG.LND.FRST.ZS": "Forest area (% of land area)",
    "NY.GDP.PCAP.KD.ZG": "GDP per capita growth (annual %)",
    "NY.GNS.ICTR.ZS": "Gross savings (% of GDP)",
    "NY.GNS.ICTR.GN.ZS": "Gross savings (% of GNI)",
    "IT.NET.USER.ZS": "Individuals using the Internet",
    "ST.INT.ARVL": "International tourism, number of arrivals",
    "SL.TLF.TOTL.IN": "Labor force, total",
    "SP.DYN.LE00.IN": "Life expectancy ",
    "SE.ADT.LITR.ZS": "Literacy rate",
    "SP.POP.GROW": "Population growth",
    "SP.POP.TOTL.FE.IN": "Population, female",
    "SP.POP.TOTL.MA.IN": "Population, male",
    "AG.LND.TOTL.RU.K2": "Rural land area (sq. km)",
    "SP.RUR.TOTL.ZS": "Rural population",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "BG.GSR.NFSV.GD.ZS": "Trade in services (% of GDP)",
    "SL.UEM.TOTL.NE.ZS": "Unemployment)"
}

countryMap = {
    "Africa Western and Central": "AFW",
    "Australia": "AUS",
    "Canada": "CAN",
    "Germany": "DEU",
    "India": "IND",
    "Italy": "ITA",
    "Russian Federation": "RUS",
    "Singapore": "SGP",
    "Thailand": "THA",
    "United Kingdom": "GBR",
    "United States": "USA"
}

# Melting the dataset
melted_df = getDataset(data).melt(
    id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
    var_name='Year', value_name='Value'
)

# Pivot the table to have Indicator Names as columns
pivoted_df = melted_df.pivot_table(
    index=['Country Name', 'Country Code', 'Year'],
    columns='Indicator Code', values='Value'
).reset_index()

pivoted_df.to_csv('PivotedDataset.csv')

filtered_columns = [col for col in pivoted_df.columns if col in INDICATOR_CODES]
df_filtered = pivoted_df[['Country Name', 'Country Code', 'Year'] + filtered_columns]
# Assuming df_filtered is your DataFrame



# Cleaning the transformed dataset
# Fill missing values with the mean of the column
df_filtered = df_filtered.fillna(df_filtered.mean(numeric_only=True))
df_filtered.to_csv('CleanedDataset1.csv',index=False)
# Display summary statistics
print("Summary Statistics:")
print(df_filtered.describe())
# Assuming df_filtered is your DataFrame
iqr_values = df_filtered.quantile(0.75, numeric_only=True) - df_filtered.quantile(0.25, numeric_only=True)
print(iqr_values)
import seaborn as sns
import matplotlib.pyplot as plt

selected_indices = ["AG.LND.ARBL.ZS", "NY.GDP.PCAP.KD.ZG", "SL.TLF.TOTL.IN", "NE.TRD.GNFS.ZS","SP.DYN.LE00.IN","BX.KLT.DINV.WD.GD.ZS"]

# Selecting only the columns corresponding to the selected indices
df_selected = df_filtered[['Country Name', 'Country Code', 'Year'] + selected_indices]

# Filter rows for Germany
germany_code = countryMap["India"]
df_germany = df_selected[df_selected['Country Code'] == germany_code]

# Calculate and print the correlation matrix for Germany
correlation_matrix_germany = df_germany.corr(numeric_only=True)

# Replace column names with indicator names
correlation_matrix_germany = correlation_matrix_germany.rename(columns=featureMap)

# Replace row names with indicator names
correlation_matrix_germany = correlation_matrix_germany.rename(index=featureMap)

print("\nCorrelation Matrix for Germany:")
print(correlation_matrix_germany)

# Create a heatmap for better visualization
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_germany, annot=True, cmap='plasma', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap for Germany")
plt.show()