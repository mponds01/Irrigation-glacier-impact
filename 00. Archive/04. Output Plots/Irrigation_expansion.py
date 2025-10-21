# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
# Manually organizing the OCR output into a structured format

# Extracted data structured into dictionary format
data = {
    'Year': ['1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000'],
    'Afghanistan': [1773.15, 1249.20, 1385.25, 1521.30, 1657.36, 2264.56, 3342.93, 3442.09, 3278.57, 3019.42, 3019.42],
    'Kyrgyzstan': [341.54, 416.99, 500.82, 584.64, 668.47, 752.30, 836.12, 919.95, 974.80, 1072.60, 1072.00],
    'Nepal': [14.70, 14.70, 14.70, 14.70, 14.70, 26.55, 66.05, 117.00, 520.00, 985.00, 1135.00],
    'Pakistan': [5451.40, 6325.01, 7198.62, 8072.24, 8890.08, 9459.55, 10148.63, 12950.00, 14680.00, 15820.00, 18090.00],
    'India': [11660.00, 13528.57, 15397.14, 17265.71, 19015.00, 20721.67, 24880.00, 31550.00, 40500.00, 47430.00, 57291.41],
    'Uzbekistan': [1081.26, 1320.10, 1585.48, 1850.86, 2116.24, 2381.62, 2647.01, 2912.39, 3454.00, 4222.00, 4223.00],
    'Turkmenistan': [309.71, 378.12, 454.13, 530.14, 606.16, 682.17, 758.18, 834.20, 942.00, 1510.70, 1800.00],
    'Kazakhstan': [877.51, 973.65, 1069.78, 1165.92, 1262.06, 1358.20, 1454.33, 1550.47, 1867.77, 2092.81, 1855.20],
    'China': [10000.00, 11224.49, 12448.98, 13673.47, 14897.96, 17004.38, 27048.13, 37669.00, 48867.00, 49533.15, 53823.00]
}

# Create the DataFrame
df_transposed = pd.DataFrame(data)

# Create the DataFrame
plt.figure(figsize=(12, 8))
for country in df_transposed.columns[1:]:  # Skipping 'Year' column
    plt.plot(df_transposed['Year'], df_transposed[country], label=country)

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Population')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# Place legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()  # Adjust layout to fit everything

# Show the plot
plt.show()
