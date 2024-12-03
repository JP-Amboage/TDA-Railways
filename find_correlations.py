import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as scp
import seaborn as sns

from utils import column2name

def correlate_df(df: pd.DataFrame, cols1: list, cols2: list, output_path: str = './', show: bool = False):
    keys =  df.index.to_list()
    colors = np.random.rand(len(keys), 3)
    names_on_plot = False

    for col1 in cols1:
        for col2 in cols2:
            x = df[col1].to_list()
            y = df[col2].to_list()

            pearson = scp.pearsonr(x, y)
            spearman = scp.spearmanr(x, y)
            print(f'Pearson between {col1} and {col2}: {pearson}')
            print(f'Spearman between {col1} and {col2}: {spearman}')
            print()

            pearson_coef = pearson[0]
            spearman_coef = spearman[0]

            plt.figure(figsize=(5, 5))

            if names_on_plot:
                plt.scatter(x, y, color=colors, label = keys, alpha=0.7, s=100, edgecolors='gray')

                y_offset = (max(y)-min(y))/60            
                for i, label in enumerate(keys):
                    plt.text(x[i], y[i] + y_offset, label, fontsize=8)

            else:
                for i, (x, y, label) in enumerate(zip(x, y, keys)):
                    plt.scatter(x, y, color=colors[i], label=label, edgecolor='gray', s=100)

                    # Adding the legend dynamically in the best location
                    plt.legend(title='', loc='best', fontsize=9, frameon=True)

            # Adding title and labels
            plt.title(f'{column2name(col1)} vs {column2name(col2)}: ' + f'Pearson = {round(pearson_coef,2)},  Spearman={round(spearman_coef,2)}', fontsize=11)
            plt.xlabel(column2name(col1), fontsize=10)
            plt.ylabel(column2name(col2), fontsize=10)

            # Tight layout for better spacing
            plt.tight_layout()

            # Saving the plot
            plot_name = f'scatter_{col1}_{col2}.png'
            plt.savefig(output_path + '/' + plot_name, dpi=300)

            # Showing the plot (optional)
            if show:
                plt.show()

def plot_correlation_matrix(df, output_path='./', show=False):
    # Calculate the correlation matrix using the specified correlation function
    methods = ['pearson', 'kendall', 'spearman']

    for method in methods:
        correlation_matrix = df.corr(method=method)
            
        # Set up the matplotlib figure
        plt.figure(figsize=(10, 8))
            
        # Generate a heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True, cbar=True)
            
        # Add a title to the heatmap
        plt.title(f'{method.capitalize()} Correlation Matrix', fontsize=12)
        
        # Saving the plot
        plot_name = f'matrix_{method}.png'
        plt.savefig(output_path + '/' + plot_name, dpi=300)
        
        # Display the plot
        if show:
            plt.show()


def main():
    topo_features_path = './Data/topo.csv'
    city_variables_path = './Data/vars.csv'

    topo_features_df = pd.read_csv(topo_features_path, index_col='city')
    city_variables_df = pd.read_csv(city_variables_path, index_col='city')

    cols_topo = topo_features_df.columns.to_list()
    cols_vars = city_variables_df.columns.to_list()

    new_df=topo_features_df.join(city_variables_df, how='inner')
    
    plot_correlation_matrix(new_df, output_path='./Data/plots', show=True)

    correlate_df(new_df, cols_topo, cols_vars, output_path='./Data/plots', show=True)

if __name__ == '__main__':
    main()