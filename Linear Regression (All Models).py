import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import abline_plot
import pandas as pd

fram = pd.read_csv('src/fram.txt', sep='\t')
fram.head()

# Summary of variables
variables_summary = fram.describe()
print("Summary of variables:")
print(variables_summary)

def rescale(series):
    mean = series.mean()
    std_dev = series.std()
    rescaled_series = (series - mean) / (2 * std_dev)
    return rescaled_series

def regression():
    # Iterate through each column in the DataFrame
    for column in fram.columns:
        # Skip non-numeric columns
        if fram[column].dtype != 'object':
            # Apply rescale function to the column series
            series = rescale(fram[column])
            # Add the rescaled series to the original DataFrame with "s" appended to the column name
            fram[f"s{column}"] = series

    # Print the head of the modified DataFrame
    print(fram.head())

    # Save the modified DataFrame to a CSV file
    fram.to_csv('rescaled_dataframe.csv', index=False)

    # Basic Model
    fit=smf.ols('SBP ~ sFRW + SEX + sCHOL',
    data=fram).fit()
    print(fit.summary())

    # Add sAGE
    fit=smf.ols('SBP ~ sFRW + SEX + sCHOL + sAGE' ,
    data=fram).fit()
    print(fit.summary())

    # Add Interactions and sCIG
    formula = ('SBP ~ sFRW + SEX + sFRW:SEX + sCHOL + sCHOL:sFRW + sCHOL:SEX + '
            'sAGE + sAGE:sFRW + sAGE:SEX + sAGE:sCHOL + sCIG') # + sCIG:sFRW + sCIG:SEX + sCIG:sCHOL + sCIG:sAGE

    # Fit the model
    fit = smf.ols(formula, data=fram).fit()

    # exercise 7
    p = fit.params
    
    fram[fram.SEX=="female"].plot.scatter("sFRW", "SBP")
    abline_plot(intercept=p.Intercept - p["sAGE"], slope=p.sFRW - p["sAGE:sFRW"],
    ax=plt.gca(), color="blue", label="low")
    abline_plot(intercept=p.Intercept, slope=p.sFRW,
    ax=plt.gca(), color="magenta", label="mid")
    abline_plot(intercept=p.Intercept + p["sAGE"], slope=p.sFRW + p["sAGE:sFRW"],
    ax=plt.gca(), color="red", label="high")
    plt.legend();


    plt.show()

    # Print the summary
    print(fit.summary())

    # Extract the significant variables excluding "Intercept" for "ideal" model
    significant_vars = [var for var in fit.pvalues[fit.pvalues < 0.05].index.tolist() if var != 'Intercept']

    # Create a DataFrame to store the results
    significant_results = pd.DataFrame(columns=['Coefficient', 'Std Error', 't-value', 'P-value'])

    # Populate the DataFrame with the significant variables' information
    for var in significant_vars:
        significant_results.loc[var] = [fit.params[var], fit.bse[var], fit.tvalues[var], fit.pvalues[var]]

    # Print the table
    print(significant_results)

    # Save the significant results to a CSV file
    significant_results.to_csv('significant_results.csv', index=False)

    p = fit.params

    # Find the most significant variable
    most_significant_result = significant_results['Coefficient'].idxmax()

    print("The most significant factor is:", most_significant_result)

    return (p, most_significant_result)
  

def main():
    p, most_significant_result = regression()

    fig, ax = plt.subplots(1, 3, subplot_kw={"xlim": (-1.6, 3.3), "ylim": (80, 310),
                                            "xlabel": most_significant_result, "ylabel": "SBP"},
                        figsize=(14, 4))

    # Plotting SBP against the most relevant factor
    ax[0].scatter(fram[most_significant_result][fram.sCHOL < -1.0],
                fram.SBP[fram.sCHOL < -1.0])
    abline_plot(p.Intercept - p["sCHOL"],
                p[most_significant_result] - p["sCHOL:sFRW"], color="blue", label="low", ax=ax[0])
    ax[0].set_title("Low sCHOL")

    ax[1].scatter(fram[most_significant_result][(fram.sCHOL > -1.0) & (fram.sCHOL < 1.0)],
                fram.SBP[(fram.sCHOL > -1.0) & (fram.sCHOL < 1.0)])
    abline_plot(p.Intercept, p[most_significant_result], color="magenta", label="mid", ax=ax[1])
    ax[1].set_title("Mid sCHOL")

    ax[2].scatter(fram[most_significant_result][fram.sCHOL > 1.0],
                fram.SBP[fram.sCHOL > 1.0])
    abline_plot(p.Intercept + p["sCHOL"],
                p[most_significant_result] + p["sCHOL:sFRW"], color="red", label="high", ax=ax[2])
    ax[2].set_title("High sCHOL")

    plt.show()

if __name__ == "__main__":
    main()

