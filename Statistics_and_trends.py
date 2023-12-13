import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.utils import resample


# paths to datasets
# Link for original world bank dataset https://datacatalog.worldbank.org/search/dataset/0037712
Original_dataUri = 'WDICSV.csv'
pivoted_data_uri = 'PivotedDataset.csv'
cleaned_data_uri = 'CleanedDataset1.csv'

#Creating a list that specifies the column names to be analysed in the cleaned dataset
INDICATOR_CODES =["SE.TER.CUAT.BA.FE.ZS", "SE.TER.CUAT.BA.MA.ZS","SE.TER.CUAT.BA.ZS", "TX.VAL.OTHR.ZS.WT","IT.NET.USER.ZS",
                   "EG.CFT.ACCS.ZS", "EG.ELC.ACCS.ZS", "SH.XPD.GHED.GD.ZS", "SP.DYN.LE00.IN", "AG.LND.ARBL.ZS",
                   "GC.XPN.TOTL.GD.ZS", "AG.LND.FRST.ZS", "NY.GDP.PCAP.KD.ZG", "NY.GNS.ICTR.ZS",
                   "NY.GNS.ICTR.GN.ZS", "ST.INT.ARVL", "SL.TLF.TOTL.IN", "SP.DYN.LE00.IN", "SP.POP.GROW",
                   "SP.POP.TOTL.FE.IN", "SP.POP.TOTL.MA.IN","SP.RUR.TOTL.ZS","SP.URB.TOTL.IN.ZS","GB.XPD.RSDV.GD.ZS"]

# Mapping of indicator codes to indicator names
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
    "SP.RUR.TOTL.ZS":"Rural population (% of total population)",
    "SP.URB.TOTL.IN.ZS":"Urban population (% of total population)",
    "GB.XPD.RSDV.GD.ZS":"Research and development expenditure (% of GDP)"
    }

#Mapping of country code to country name
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

# Reading data set using pandas.
# use of melt and pivot methods in dataframe.
def getDataset(dataUrl):
    """ Function to  read a dataset and Transpose using pandas
    Args:
        dataUrl (uri): This argument takes a path to the dataset 
     Returns:
        Returns a transposed dataframe after melt and pivot operations on the dataUrl that was passed.
    """

    dataset = pd.read_csv(dataUrl)
    melted_df = dataset.melt(id_vars=[
        'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Value')
    Transposed_df = melted_df.pivot_table(
        index=['Country Name', 'Country Code', 'Year'], columns='Indicator Code', values='Value').reset_index()
    Transposed_df.to_csv('PivotedDataset.csv')
    return Transposed_df


def cleanDataSet(dataset):
    """ Function to clean a dataset
    Args:
        dataset: Dataset which needs to be cleaned is passed as an argument.

    Returns:
        _Returns a cleaned dataset after removing null values and indicators which are not useful.
    """

    filtered_columns = [col for col in dataset.columns if col in INDICATOR_CODES]
    df_filtered = getDataset(Original_dataUri)[filtered_columns]
    df_cleaned = df_filtered.fillna(df_filtered.mean())
    df_cleaned.to_csv('CleanedDataset1.csv')
    return cleaned_df


df_filtered = pd.read_csv('CleanedDataset1.csv')

#Using describe method to find statistical properties
print(df_filtered.describe())

# Select a few countries for analysis
selected_countries = ['India', 'Australia', 'Germany', 'Canada']

# Select indicators of interest
indicators_of_interest = ['NY.GDP.PCAP.KD.ZG', 'SP.POP.GROW']

# Filter data for selected countries
df_selected = df_filtered[df_filtered['Country Name'].isin(selected_countries)]


# Function to calculate skewness and kurtosis for a given indicator
def calculate_skewness_kurtosis(data):
    """
    Calculate skewness and kurtosis for a given dataset.

    Parameters:
    - data (array-like): The input data for which skewness and kurtosis will be calculated.

    Returns:
    - tuple: A tuple containing two values - skewness and kurtosis.
    """

    skewness = skew(data)
    kurt = kurtosis(data)
    return skewness, kurt

# Function to perform bootstrapping and confidence intervals
def bootstrap(data, num_iterations=1000, confidence_level=0.95):
    """
    Perform bootstrapping on a given dataset to estimate confidence intervals for skewness and kurtosis.

    Parameters:
    - data (array-like): The input data for bootstrapping.
    - num_iterations (int, optional): The number of bootstrap iterations. Default is 1000.
    - confidence_level (float, optional): The confidence level for calculating intervals. Default is 0.95.

    Returns:
    - numpy.ndarray: A 2D array containing the confidence intervals for skewness and kurtosis.
    """
    results = []
    for _ in range(num_iterations):
        # Generate a bootstrap sample
        sample = resample(data)
        skewness, kurt = calculate_skewness_kurtosis(sample)
        results.append((skewness, kurt))

    # Calculate confidence intervals
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = 100 - lower_percentile
    confidence_intervals = np.percentile(results, [lower_percentile, upper_percentile], axis=0)

    return confidence_intervals


# Analyze and compare indicators
for indicator in indicators_of_interest:
    print(f"\nStatistical properties for indicator: {indicator}\n")

    for country in selected_countries:
        # Extract data for the specific country and indicator
        country_data = df_selected[df_selected['Country Name'] == country][indicator]

        # Calculate skewness and kurtosis
        skewness, kurt = calculate_skewness_kurtosis(country_data)

        # Perform bootstrapping and calculate confidence intervals
        confidence_intervals = bootstrap(country_data)

        # Print results
        print(f"Country: {country}")
        print(f"Skewness: {skewness:.4f}")
        print(f"Kurtosis: {kurt:.4f}")
        print(f"95% Confidence Intervals for Skewness: [{confidence_intervals[0][0]:.4f}, {confidence_intervals[1][0]:.4f}]")
        print(f"95% Confidence Intervals for Kurtosis: [{confidence_intervals[0][1]:.4f}, {confidence_intervals[1][1]:.4f}]\n")
    
def plot_correlation_heatmap(df, country_name, selected_indices, feature_map, country_map, figure_number):
    
    # Selecting only the columns corresponding to the selected indices
    df_selected = df_filtered[['Country Name', 'Country Code', 'Year'] + selected_indices]


    # Filter rows for the specified country
    country_code = country_map[country_name]
    df_country = df_selected[df_selected['Country Code'] == country_code]

    # Calculate the correlation matrix
    correlation_matrix_country = df_country.corr(numeric_only=True)

    # Replace column names with indicator names
    correlation_matrix_country = correlation_matrix_country.rename(columns=feature_map)

    # Replace row names with indicator names
    correlation_matrix_country = correlation_matrix_country.rename(index=feature_map)

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

    sns.heatmap(correlation_matrix_country, annot=True, cmap=cmap, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap for {country_name}")
    plt.savefig(f"heatmap_{figure_number}.png")  # Save the figure with a specific filename
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
plot_correlation_heatmap(df_filtered, "Australia", selected_indices, feature_map, country_map, 1)

# Plot correlation matrix and heatmap for Singapore
plot_correlation_heatmap(df_filtered, "Singapore", selected_indices, feature_map, country_map, 2)

# Plot correlation matrix and heatmap for India
plot_correlation_heatmap(df_filtered, "India", selected_indices, feature_map, country_map, 3)


def plot_grouped_bar_chart(df, countries, years, indicator_column, title, xlabel, ylabel, palette, figure_number):
    """
    Plot a grouped bar chart to visualize the specified indicator for selected countries over different years.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - countries (list): A list of country names to be included in the chart.
    - years (list): A list of years to be included in the chart.
    - indicator_column (str): The column name representing the indicator to be plotted.
    - title (str): The title of the chart.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - palette (str or seaborn color palette): The color palette to be used in the chart.
    - figure_number (int): The figure number for saving the chart.

    Returns:
    - None: Displays the grouped bar chart.
    """
    
    filtered_data = df[
        (df['Country Name'].isin(countries)) &
        (df['Year'].isin(years))
    ]

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Country Name', y=indicator_column, hue='Year', data=filtered_data, palette=palette)
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


def plot_gdp_scatter(df, country_names, indicator_column):
    """
    Generate a scatter plot to compare GDP per capita growth between specified countries.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - country_names (list): A list of country names to be compared in the scatter plot.
    - indicator_column (str): The column name representing the GDP per capita growth indicator.

    Returns:
    - None: Displays the scatter plot.
    """

    plt.figure(figsize=(10, 6))

    for country_name in country_names:
        # Filter data for the specified country
        country_data = df[df['Country Name'] == country_name]

        # Plot the scatter plot for each country
        plt.scatter(country_data['Year'], country_data[indicator_column], label=country_name)

    # Set labels and title
    plt.title('Scatter Plot of GDP over Different Years')
    plt.xlabel('Year')
    plt.ylabel('GDP per capita growth (in %)')

    # Add legend
    plt.legend()

    # Adjust x-axis labels rotation to avoid overlap
    plt.xticks(rotation=45, ha='right')

    # Save the figure
    plt.savefig('scatter_plot_combined.png', bbox_inches='tight')

    # Show the plot
    plt.show()

# Example usage with both United Kingdom and United States
indicator_column = 'NY.GDP.PCAP.KD.ZG'
country_names = ['United Kingdom', 'United States']

# Call the function
plot_gdp_scatter(df_filtered, country_names, indicator_column)


def plot_indicators_for_country(ax, df, country_name, selected_indicators, indicator_labels):
    """
    Plot line graphs to compare specific economic indicators for a given country over time.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object on which to plot the data.
    - df (pandas.DataFrame): The DataFrame containing the data.
    - country_name (str): The name of the country for which the indicators will be plotted.
    - selected_indicators (list): A list of column names representing the indicators to be plotted.
    - indicator_labels (list): A list of labels for the corresponding indicators.

    Returns:
    - None: Displays the line graphs on the specified Axes object.
    """

    # Selecting the data for a specific country
    country_data = df[df['Country Name'] == country_name]

    # Plotting the line graph
    for i in range(len(selected_indicators)):
        indicator = selected_indicators[i]
        label = indicator_labels[i]
        ax.plot(country_data['Year'], country_data[indicator], label=label)

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
indicator_labels_india = ["Expense (% of GDP)", "Gross Savings (% of GDP)"]
plot_indicators_for_country(axs[0], df_filtered, 'India', selected_indicators_india, indicator_labels_india)


# Example usage for Australia
selected_indicators_australia = ["GC.XPN.TOTL.GD.ZS", "NY.GNS.ICTR.ZS"]
indicator_labels_australia = ["Expense (% of GDP)", "Gross Savings (% of GDP)"]
plot_indicators_for_country(axs[1], df_filtered, 'Australia', selected_indicators_australia, indicator_labels_australia)


# Example usage for United States
selected_indicators_us = ["GC.XPN.TOTL.GD.ZS", "NY.GNS.ICTR.ZS"]
indicator_labels_us = ["Expense (% of GDP)", "Gross Savings (% of GDP)"]
plot_indicators_for_country(axs[2], df_filtered, 'United States', selected_indicators_us, indicator_labels_us)


# Adjust layout
plt.tight_layout()

# Save the figure with a specific filename
plt.savefig("indicators_subplots.png")

# Show the subplots
plt.show()