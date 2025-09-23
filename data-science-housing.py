import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(r"C:\Users\HP\Desktop\data sciecs project\housing.csv")
df.info()
df["bedroom"] = df['bedroom'].str.strip()
df["bedroom"] = pd.to_numeric(df['bedroom'], errors="coerce")
df["bedroom"] = df["bedroom"].fillna(0).astype(int)

df["size_sqm"] = df['size_sqm'].str.strip()
df["size_sqm"] = df["size_sqm"].str.replace(",", "")
df["size_sqm"] = pd.to_numeric(df['size_sqm'], errors="coerce")
df["size_sqm"] = df["size_sqm"].fillna(0).astype(int)

df["price"] = df['price'].str.strip()
df["price"] = df["price"].str.replace(",", "")
df["price"] = pd.to_numeric(df['price'], errors="coerce")
df["price"] = df["price"].fillna(0).astype(int)

df.info()
df.head()
df_split = df['location'].str.split(",", expand=True)
df_split.columns = ['Area Name', 'Subdivision/Compound Name', 'Settlement Name', 'City', 'Governorate']
df = pd.concat([df, df_split], axis=1)
df.head()
df.describe()
df.shape
df.isna().sum()


# Univariate Analysis for Numerical Columns
numerical_cols = ['bedroom', 'bathroom', 'size_sqm', 'price']

for col in numerical_cols:
    print(f"\nSummary for {col}")
    print(df[col].describe())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Univariate Analysis for Numerical Variable: {col}", fontsize=16)

    # Histogram
    sns.histplot(df[col], kde=False, bins=20, ax=axes[0, 0])
    axes[0, 0].set_title("Histogram")

    # Density Plot
    sns.kdeplot(df[col], fill=True, ax=axes[0, 1])
    axes[0, 1].set_title("Density Plot")

    # Box Plot
    sns.boxplot(x=df[col], ax=axes[0, 2])
    axes[0, 2].set_title("Box Plot")

    # Violin Plot
    sns.violinplot(x=df[col], ax=axes[1, 0])
    axes[1, 0].set_title("Violin Plot")

    # Strip Plot
    sns.stripplot(x=df[col], ax=axes[1, 1], jitter=True)
    axes[1, 1].set_title("Strip Plot")

    # Hide the last subplot (unused)
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Univariate Analysis for Categorical Columns
categorical_cols = ['Area Name', 'Subdivision/Compound Name', 'Settlement Name', 'City', 'Governorate']

for col in categorical_cols:
    print(f"\nValue counts for {col}")
    print(df[col].value_counts())
    
    filtered_values = df[col].value_counts()
    filtered_values = filtered_values[filtered_values >= 364].index

    df_filtered = df[df[col].isin(filtered_values)]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"Univariate Analysis for Categorical Variable: {col}", fontsize=16)

    # Count Plot
    sns.countplot(x=col, data=df_filtered, ax=axes[0])
    axes[0].set_title("Count Plot")

    # Bar Plot (manual)
    df_filtered[col].value_counts().plot(kind='bar', ax=axes[1])
    axes[1].set_title("Bar Plot")
    axes[1].set_ylabel("Frequency")

    # Pie Chart
    df_filtered[col].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=axes[2])
    axes[2].set_title("Pie Chart")
    axes[2].set_ylabel('')  # Hide y-label

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()
# List of bivariate analysis pairs
bivariate_pairs = [
    ('size_sqm', 'price'),        # Size vs Price
    ('bedroom', 'price'),         # Bedroom Count vs Price
    ('bathroom', 'price'),        # Bathroom Count vs Price
    ('Governorate', 'price'),     # Governorate vs Price
    ('Settlement Name', 'price'), # Settlement vs Price
    ('type', 'price'),            # Property Type vs Price
    ('size_sqm', 'bedroom'),      # Size vs Bedroom Count
    ('size_sqm', 'bathroom')      # Size vs Bathroom Count
]

# Iterate through the pairs and generate plots
for var1, var2 in bivariate_pairs:
    plt.figure(figsize=(30, 6))

    # If both variables are continuous, use histogram, KDE, or scatter plot
    if df[var1].dtype == 'int64' and df[var2].dtype == 'int64': 
        # Plot histogram for both variables
        plt.subplot(1, 2, 1)
        sns.histplot(df[var1], kde=True, bins=30)
        plt.title(f'Histogram & KDE of {var1}')
        
        # Plot scatter plot
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=var1, y=var2, data=df)
        plt.title(f'{var1} vs {var2} (Scatter Plot)')

    # If one variable is categorical and the other is continuous, use barplot, boxplot, or KDE
    elif df[var1].dtype == 'object' and df[var2].dtype == 'int64': 
        plt.subplot(1, 2, 1)
        sns.barplot(x=var1, y=var2, data=df)
        plt.title(f'{var1} vs {var2} (Barplot)')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=var1, y=var2, data=df)
        plt.title(f'{var1} vs {var2} (Boxplot)')

    # If both variables are categorical, use pie chart and count plot
    elif df[var1].dtype == 'object' and df[var2].dtype == 'object': 
        # Pie chart for the first categorical variable
        plt.subplot(1, 2, 1)
        df[var1].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title(f'{var1} (Pie Chart)')

        # Count plot for comparing categories
        plt.subplot(1, 2, 2)
        sns.countplot(x=var1, hue=var2, data=df)
        plt.title(f'{var1} vs {var2} (Count Plot)')

    plt.tight_layout()
    plt.show()
df.info()
df_new = df.drop(columns=["title", "location"])

# Mapping for all columns
area_mapping = {name: idx + 1 for idx, name in enumerate(df_new['Area Name'].unique())}
compound_mapping = {name: idx + 1 for idx, name in enumerate(df_new['Subdivision/Compound Name'].unique())}
settlement_mapping = {name: idx + 1 for idx, name in enumerate(df_new['Settlement Name'].unique())}
city_mapping = {name: idx + 1 for idx, name in enumerate(df_new['City'].unique())}
governorate_mapping = {name: idx + 1 for idx, name in enumerate(df_new['Governorate'].unique())}
type_mapping = {name: idx + 1 for idx, name in enumerate(df_new['type'].unique())}


# Now map those numbers to the respective columns in df_new
df_new['Area Name'] = df_new['Area Name'].map(area_mapping)
df_new['Subdivision/Compound Name'] = df_new['Subdivision/Compound Name'].map(compound_mapping)
df_new['Settlement Name'] = df_new['Settlement Name'].map(settlement_mapping)
df_new['City'] = df_new['City'].map(city_mapping)
df_new['Governorate'] = df_new['Governorate'].map(governorate_mapping)
df_new['type'] = df_new['type'].map(type_mapping)


# Check the result
df_new.head()
mat_corr = df_new.corr()
sns.heatmap(mat_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
variables = []
for i in df_new.columns:
    variables.append(i)

plt.xticks(range(len(variables)), variables, rotation=45, ha="right")
plt.yticks(range(len(variables)), variables)
df_new.info()




