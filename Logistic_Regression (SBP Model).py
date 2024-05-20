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

points = pd.read_csv('points.csv')
points.head()

def logistic_transform(x):
    return 1.0 / (1.0 + np.exp(-x))

def rescale(series):
    mean = series.mean()
    std_dev = series.std()
    rescaled_series = (series - mean) / (2 * std_dev)
    return rescaled_series

def main():

    # Best Cross-Validation Model ----------------------------------------------------------------------------------------
    points["HIGH_BP2"] = (points.SBP > 140) | (points.DBP > 90)
    points["HIGH_BP2"] = points["HIGH_BP2"].map(int)
    points["HIGH_BP2"].mean()

    error_model=[]
    error_null=[]
    np.random.seed(9)
    for i in range(100):
        train, test = train_test_split(points, train_size=0.8)
        fit_final = smf.glm(formula="HIGH_BP2 ~ sFRW + sCHOL + sCHOL:sFRW + sCIG + sCIG:sFRW", data=train,
        family=sm.families.Binomial()).fit()
        #print(model.summary())
        pred = fit_final.predict(test)
        error_rate = np.mean(((pred < 0.5) & (test.HIGH_BP2==1)) |
        ((pred > 0.5) & (test.HIGH_BP2==0)))
        error_model.append(error_rate)
        error_null.append((1-test.HIGH_BP2).mean())

    print(fit_final.summary())    
    #for model, null in zip(error_model, error_null):
    # print(model, null)
    pd.Series(error_model).mean(), pd.Series(error_null).mean()

    print(statsmodels.stats.stattools.stats.mannwhitneyu(error_model, error_null,
    alternative="two-sided"))

    # Plot with jitter
    plt.scatter(points.sFRW, points.HIGH_BP2 + np.random.uniform(-0.05, 0.05, len(points)),
    marker="d")
    X=np.linspace(-2, 4, 100)
    p = fit_final.params
    plt.plot(X, logistic_transform(X*p.sFRW + p.Intercept), color="red", label="female")
    plt.plot(X, logistic_transform(X*(p.sFRW + p["sCIG:sFRW"]) +
    p["sCHOL"] + p.Intercept), color="blue",label="male")
    plt.xlabel("Weight")
    plt.ylabel("Pr(Has high BP)")
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
    individual_pred = fit_final.predict(final_row)

    predicted = individual_pred[0]
    print("Probability of showing signs of heart disease:", predicted)  # Accessing the predicted value directly
    print("The error rate is :", error_rate)
    
if __name__ == "__main__":
    main()

