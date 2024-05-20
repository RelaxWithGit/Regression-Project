import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
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

def linear_regression():
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
    fram.to_csv('point.csv', index=False)

    # Add Interactions and sCIG
    #formula = ('SBP ~ sFRW + SEX + sFRW:SEX + sCHOL + sCHOL:sFRW + sCHOL:SEX + '
    #        'sAGE + sAGE:sFRW + sAGE:SEX + sAGE:sCHOL + sCIG') # + sCIG:sFRW + sCIG:SEX + sCIG:sCHOL + sCIG:sAGE

    #Split data into 2 for training and testing
    error_basic=[]
    error_interact=[]
    np.random.seed(9)
    for i in range(100):
        train, test = train_test_split(fram)
        fit1 = smf.ols('SBP ~ sFRW + SEX', data=train).fit()
        fit2 = smf.ols('SBP ~ sFRW + SEX + sCHOL + sCHOL:sFRW + sAGE + sCIG + sCIG:sFRW', data=train).fit()
        pred1 = fit1.predict(test)
        pred2 = fit2.predict(test)
        error_basic.append(np.sqrt(np.mean((pred1 - test.SBP)**2)))
        error_interact.append(np.sqrt(np.mean((pred2 - test.SBP)**2)))
    
    RMSEs = pd.Series(error_basic).mean(), pd.Series(error_interact).mean()
    print("RMSE for basic and interaction models: ", RMSEs)

    Mannwhit = statsmodels.stats.stattools.stats.mannwhitneyu(error_basic, error_interact, alternative="two-sided")
    print("Mannwhitneyu Results: ", Mannwhit)

    # Plot fit1
    p = fit1.params

    fram.plot.scatter("sFRW", "SBP")
    int1 = p.Intercept + p["SEX[T.male]"]
    int2 = p.Intercept
    slope=p.sFRW
    abline_plot(intercept=int1, slope=slope, ax=plt.gca(), color="blue", label="male")
    abline_plot(intercept=int2, slope=slope, ax=plt.gca(), color="red", label="female")
    plt.legend();

    plt.show()


    q = fit2.params
    
    fig, ax = plt.subplots(1,3, subplot_kw={"xlim": (-1.6, 3.3), "ylim": (80,310),
    "xlabel": "sFRW", "ylabel": "SBP"},
    figsize=(14, 4))
    ax[0].scatter(fram.sFRW[(fram.SEX=="female") & (fram.sCHOL < -0.5)],
    fram.SBP[(fram.SEX=="female") & (fram.sCHOL < -0.5)])
    abline_plot(q.Intercept - q["sCHOL"],
    q.sFRW - q["sCHOL:sFRW"], color="blue", label="low", ax=ax[0])
    ax[0].set_title("female, low CHOL")
    ax[1].scatter(fram.sFRW[(fram.SEX=="female") & (fram.sCHOL > -0.5) &
    (fram.sCHOL < 0.5)],
    fram.SBP[(fram.SEX=="female") & (fram.sCHOL > -0.5) &
    (fram.sCHOL < 0.5)])
    abline_plot(q.Intercept, q.sFRW, color="magenta", label="mid", ax=ax[1])
    ax[1].set_title("female, mid CHOL")
    ax[2].scatter(fram.sFRW[(fram.SEX=="female") & (fram.sCHOL > 0.5)],
    fram.SBP[(fram.SEX=="female") & (fram.sCHOL > 0.5)])
    abline_plot(q.Intercept + q["sCHOL"],
    q.sFRW + q["sCHOL:sFRW"], color="red", label="high", ax=ax[2])
    ax[2].set_title("female, high CHOL")

    plt.show()

    # Print the summary
    print(fit1.summary())
    print(fit2.summary())

    return p, q

def main():
    p, q = linear_regression()
    
    #A Linear regression model will only allow us to estimate the most impactful factors on a dependant variable in a general sense.
    #  It cannot be used to predict an outcome for an individual data point. For that, we need a logistic regression model.
    
    #If a person has cholestherol 200, smokes 17 cigarets per day, and has weight 100, then what is the probability that
    #  he/she sometimes shows signs of coronal hear disease?

if __name__ == "__main__":
    main()