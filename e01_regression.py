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

def logistic_transform(x):
    return 1.0 / (1.0 + np.exp(-x))

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
    
    #Initialize a new DataFrame called "points" with the same columns and data as "fram"
    points = fram.copy()

    # Iterate through each column in the DataFrame
    for column in fram.columns:
        # Skip non-numeric columns
        if fram[column].dtype != 'object':
            # Apply rescale function to the column series
            rescaled_series = rescale(fram[column])
            # Add the rescaled series to the new DataFrame with "s" appended to the column name
            points[f"s{column}"] = rescaled_series
    print(points.head())

    # Best Cross-Validation Model ----------------------------------------------------------------------------------------
    # Adding a new column 'hasCHD' to the DataFrame
    points['hasCHD'] = 0

    # Looping through each row in the DataFrame to check 'CHD' values
    for index, row in points.iterrows():
        if row['CHD'] > 0:
            points.at[index, 'hasCHD'] = 1
        else:
            points.at[index, 'hasCHD'] = 0

    print(points)

    # Compute the mean of the 'hasCHD' column
    positive_cases_mean = points['hasCHD'].mean()

    print("Proportion of positive cases:", positive_cases_mean)

    error_model = []
    error_null = []
    np.random.seed(9)
    for i in range(100):
        train, test = train_test_split(points, train_size=0.8)
        fit = smf.glm(formula="hasCHD ~ sFRW + sCHOL + sCIG + sFRW:sCHOL + sFRW:sCIG + sCHOL:sCIG", data=train,
        family=sm.families.Binomial()).fit()
        pred = fit.predict(test)
        error_rate = np.mean(((pred < 0.5) & (test.hasCHD == 1)) |
        ((pred > 0.5) & (test.hasCHD == 0)))
        error_model.append(error_rate)
        error_null.append((1 - test.hasCHD).mean())
    
    print(fit.summary())
    print("The error rate is:\n", error_rate)
    print(statsmodels.stats.stattools.stats.mannwhitneyu(error_model, error_null,
    alternative="two-sided"))

    plt.scatter(points.sCIG + np.random.uniform(-0.1, 0.1, len(points)), points.hasCHD, marker="d")
    X=np.linspace(-0.5, 2.5, 1000)
    p = fit.params
    plt.plot(X, logistic_transform(X*p.sCIG + p.Intercept), color="red")

    plt.xlabel("No. Cigs")
    plt.ylabel("Pr(Has Coranary Heart Disease)")
    plt.legend();
    plt.show()

    #If a person has cholestherol 200, smokes 17 cigarets per day, and has weight 100, then what is the probability that--------------------------
    #  he/she sometimes shows signs of coronal hear disease?--------------------------------------------------------------------------------------
    
    # Prediction for individual data point--------------------------------------------------------------------------------------------------------
    point = pd.DataFrame({
        'FRW': [100],  # Weight (kg)
        'CHOL': [200], # Cholesterol (mg/dl)
        'CIG': [17]    # Units
    })

    # Display the DataFrame
    #print("Original point", point)
    #print("Original fram", fram.head())

    # Concatenate fram on top of point
    fram_and_point = pd.concat([fram, point], axis=0, ignore_index=True)

    # Print the resulting dataframe
    #print("fram_and_point dataframe after concatenation:")

    print("Initial fram_and_point is: ", fram_and_point.head())

    # Iterate through each column in the DataFrame
    for column in fram_and_point.columns:
        # Skip non-numeric columns
        if fram_and_point[column].dtype != 'object':
            # Apply rescale function to the column series
            series = rescale(fram_and_point[column])
            # Add the rescaled series to the original DataFrame with "s" appended to the column name
            fram_and_point[f"s{column}"] = series

    print("Final fram_and_point is: ", fram_and_point.head())
    
    #rename fram_and_point to point
    point = fram_and_point
    
    # Extract final row as a new DataFrame without preserving the index
    final_row = point.iloc[[-1]].reset_index(drop=True).copy()

    # Use the fitted model to predict the probability of dangerously high blood pressure
    individual_pred = fit.predict(final_row)

    predicted = individual_pred[0]
    print("Probability of showing signs of heart disease:", predicted)  # Accessing the predicted value directly
    print("The error rate is :", error_rate)
    
if __name__ == "__main__":
    main()
