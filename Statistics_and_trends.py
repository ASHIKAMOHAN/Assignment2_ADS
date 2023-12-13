import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# paths to datasets
Original_dataUri = "C:\\Users\\ashik\\Desktop\\WDICSV.csv"
pivoted_data_uri = 'PivotedDataset.csv'
cleaned_data_uri = 'CleanedDataset1.csv'

INDICATOR_CODES = ["SE.TER.CUAT.BA.FE.ZS", "SE.TER.CUAT.BA.MA.ZS", "SE.TER.CUAT.BA.ZS", "TX.VAL.OTHR.ZS.WT", "IT.NET.USER.ZS",
                   "EG.CFT.ACCS.ZS", "EG.ELC.ACCS.ZS", "SH.XPD.GHED.GD.ZS", "SP.DYN.LE00.IN", "AG.LND.ARBL.ZS",
                   "GC.XPN.TOTL.GD.ZS", "AG.LND.FRST.ZS", "NY.GDP.PCAP.KD.ZG", "NY.GNS.ICTR.ZS",
                   "NY.GNS.ICTR.GN.ZS", "ST.INT.ARVL", "SL.TLF.TOTL.IN", "SP.DYN.LE00.IN", "SP.POP.GROW",
                   "SP.POP.TOTL.FE.IN", "SP.POP.TOTL.MA.IN", "SP.RUR.TOTL.ZS", "SP.URB.TOTL.IN.ZS", "GB.XPD.RSDV.GD.ZS"]
feature_map = {
    "SE.TER.CUAT.BA.FE.ZS": "Educational attainment,population 25+, female (%)",
    "SE.TER.CUAT.BA.MA.ZS": "Educational attainment,population 25+, male (%)",
    "SE.TER.CUAT.BA.ZS": "Educational attainment,population 25+, total (%) (cumulative)",
    "TX.VAL.OTHR.ZS.WT": "Computer, communications and other services",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "EG.CFT.ACCS.ZS": "Access to clean fuels and technologies for cooking (% of population)",
    "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
    "SH.XPD.GHED.GD.ZS": "Domestic general government health expenditure (% of GDP)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    "AG.LND.ARBL.ZS": "Arable land",
    "GC.XPN.TOTL.GD.ZS": "Expense (% of GDP)",
    "AG.LND.FRST.ZS": "Forest area (% of land area)",
    "NY.GDP.PCAP.KD.ZG": "GDP per capita growth (annual %)",
    "NY.GNS.ICTR.ZS": "Gross savings (% of GDP)",
    "NY.GNS.ICTR.GN.ZS": "Gross savings (% of GNI)",
    "ST.INT.ARVL": "International tourism, number of arrivals",
    "SL.TLF.TOTL.IN": "Labor force, total",
    "SP.DYN.LE00.IN": "Life expectancy",
    "SP.POP.GROW": "Population growth",
    "SP.POP.TOTL.FE.IN": "Population, female",
    "SP.POP.TOTL.MA.IN": "Population, male",
    "SP.RUR.TOTL.ZS": "Rural population (% of total population)", "SP.URB.TOTL.IN.ZS": "Urban population (% of total population)",
    "GB.XPD.RSDV.GD.ZS": "Research and development expenditure (% of GDP)"}

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

def getDataset(dataUrl):
    dataset = pd.read_csv(dataUrl)
    melted_df = dataset.melt(id_vars=[
        'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Value')
    Transposed_df = melted_df.pivot_table(
        index=['Country Name', 'Country Code', 'Year'], columns='Indicator Code', values='Value').reset_index()
    Transposed_df.to_csv('PivotedDataset.csv')
    return Transposed_df

def cleanDataSet(dataset):
    filtered_columns = [
        col for col in dataset.columns if col in INDICATOR_CODES]
    df_filtered = getDataset(Original_dataUri)[filtered_columns]
    df_cleaned = df_filtered.fillna(df_filtered.mean())
    df_cleaned.to_csv('CleanedDataset1.csv')
    return cleaned_df

df_filtered = pd.read_csv('CleanedDataset1.csv')

# Display summary statistics
print("Summary Statistics:")
print(df_filtered.describe())

def plot_correlation_heatmap(df, country_name, selected_indices, feature_map, country_map, figure_number):

    # Selecting only the columns corresponding to the selected indices
    df_selected = df_filtered[['Country Name',
                               'Country Code', 'Year'] + selected_indices]

    # Filter rows for the specified country
    country_code = country_map[country_name]
    df_country = df_selected[df_selected['Country Code'] == country_code]

    # Calculate the correlation matrix
    correlation_matrix_country = df_country.corr(numeric_only=True)

    # Replace column names with indicator names
    correlation_matrix_country = correlation_matrix_country.rename(
        columns=feature_map)

    # Replace row names with indicator names
    correlation_matrix_country = correlation_matrix_country.rename(
        index=feature_map)

    # Print the correlation matrix
    print(f"\nCorrelation Matrix for {country_name}:")
    print(correlation_matrix_country)

    # Create a heatmap for better visualization with different color maps for each chart
    plt.figure(figsize=(12, 10))

    # Define custom color maps
    if figure_number == 1:
        cmap = sns.color_palette("viridis", as_cmap=True)
    elif figure_number == 2:
        cmap = sns.color_palette("plasma", as_cmap=True)
    else:
        cmap = sns.color_palette("cividis", as_cmap=True)

    sns.heatmap(correlation_matrix_country, annot=True, cmap=cmap,
                fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap for {country_name}")
    # Save the figure with a specific filename
    plt.savefig(f"heatmap_{figure_number}.png")
    plt.show()

# Define selected indices, feature map, and country map
selected_indices = ["SE.TER.CUAT.BA.ZS", "IT.NET.USER.ZS",
                    "AG.LND.ARBL.ZS", "SH.XPD.GHED.GD.ZS", "SP.DYN.LE00.IN",
                    "AG.LND.FRST.ZS", "NY.GDP.PCAP.KD.ZG", "SP.POP.GROW", "GB.XPD.RSDV.GD.ZS"]
feature_map = {
    "SE.TER.CUAT.BA.ZS": "Educational attainment,population 25+, total (%) (cumulative)",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "AG.LND.ARBL.ZS": "Arable land",
    "SH.XPD.GHED.GD.ZS": "Domestic general government health expenditure (% of GDP)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    "AG.LND.FRST.ZS": "Forest area (% of land area)",
    "NY.GDP.PCAP.KD.ZG": "GDP per capita growth (annual %)",
    "SP.POP.GROW": "Population growth",
    "GB.XPD.RSDV.GD.ZS": "Research and development expenditure (% of GDP)"
}
country_map = {"Australia": "AUS", "Singapore": "SGP", "India": "IND"}

# Plot correlation matrix and heatmap for Australia
plot_correlation_heatmap(df_filtered, "Australia",
                         selected_indices, feature_map, country_map, 1)

# Plot correlation matrix and heatmap for Singapore
plot_correlation_heatmap(df_filtered, "Singapore",
                         selected_indices, feature_map, country_map, 2)

# Plot correlation matrix and heatmap for India
plot_correlation_heatmap(df_filtered, "India",
                         selected_indices, feature_map, country_map, 3)


def plot_grouped_bar_chart(df, countries, years, indicator_column, title, xlabel, ylabel, palette, figure_number):
    filtered_data = df[
        (df['Country Name'].isin(countries)) &
        (df['Year'].isin(years))
    ]

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Country Name', y=indicator_column,
                hue='Year', data=filtered_data, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Year')

    # Save the figure with a specific filename
    plt.savefig(f"grouped_bar_chart_{figure_number}.png")

    plt.show()

# Read the dataset
df_population_by_year = pd.read_csv('CleanedDataset1.csv')

# Define countries, years, and palette
countries_of_interest = ['India', 'Australia', 'Germany', 'Canada']
years_of_interest = [1960, 1980, 2000, 2022]
palette_population_growth = 'viridis'
palette_gdp_growth = 'pastel'

# Plot Population Growth
plot_grouped_bar_chart(
    df_population_by_year,
    countries_of_interest,
    years_of_interest,
    'SP.POP.GROW',
    'Population Growth in 1960, 1980, 2000, and 2022',
    'Country',
    'Population Growth (in %)',
    palette_population_growth,
    1
)

# Plot GDP per capita growth
plot_grouped_bar_chart(
    df_population_by_year,
    countries_of_interest,
    years_of_interest,
    'NY.GDP.PCAP.KD.ZG',
    'GDP per Capita Growth in 1960, 1980, 2000, and 2022',
    'Country',
    'GDP per Capita Growth (annual %)',
    palette_gdp_growth,
    2
)

def plot_gdp_scatter(df, country_name, indicator_column):
    # Filter data for the specified country
    country_data = df[df['Country Name'] == country_name]

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(country_data['Year'], country_data[indicator_column],
                color='blue', marker='o', label='GDP')

    # Set labels and title
    plt.title(f'Scatter Plot of GDP over Different Years in UNITED KINGDOM')
    plt.xlabel('Year')
    plt.ylabel(f'{indicator_column} (in %)')

    # Add legend
    plt.legend()

    # Show the plot with a custom x-axis ticks interval (e.g., every 5 years)
    plt.xticks(country_data['Year'][::5])

    # Save the figure
    plt.savefig(f'scatter_plot_{country_name.lower().replace(" ", "_")}.png')

    # Show the plot
    plt.show()

# Example usage
indicator_column = 'NY.GDP.PCAP.KD.ZG'
country_name = 'United Kingdom'

# Call the function
plot_gdp_scatter(df_filtered, country_name, indicator_column)


def plot_indicators_for_country(ax, df, country_name, selected_indicators):
    # Selecting the data for a specific country
    country_data = df[df['Country Name'] == country_name]

    # Plotting the line graph
    for indicator in selected_indicators:
        ax.plot(country_data['Year'], country_data[indicator], label=indicator)

    ax.set_title(f"{country_name}")
    ax.set_xlabel('Year')
    ax.set_ylabel('Indicator Values')
    ax.legend()

    # Setting x-axis ticks to display every 5 years
    ax.set_xticks(country_data['Year'][::5])

# Creating subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Example usage for India
selected_indicators_india = ["GC.XPN.TOTL.GD.ZS", "NY.GNS.ICTR.ZS"]
plot_indicators_for_country(
    axs[0], df_filtered, 'India', selected_indicators_india)

# Example usage for Australia
selected_indicators_australia = ["GC.XPN.TOTL.GD.ZS", "NY.GNS.ICTR.ZS"]
plot_indicators_for_country(
    axs[1], df_filtered, 'Australia', selected_indicators_australia)

# Example usage for United States
selected_indicators_us = ["GC.XPN.TOTL.GD.ZS", "NY.GNS.ICTR.ZS"]
plot_indicators_for_country(
    axs[2], df_filtered, 'United States', selected_indicators_us)

# Adjust layout
plt.tight_layout()

# Save the figure with a specific filename
plt.savefig("indicators_subplots.png")

# Show the subplots
plt.show()
