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
    
    X=np.linspace(-8, 8, 100)
    #print(X)
    plt.plot(X, logistic_transform(X));
    #plt.show()

    #defining what we diagnose as high blood pressure
    points["HIGH_BP"] = (points.SBP >= 140) | (points.DBP >= 90)
    points.HIGH_BP.head()

    #Useful metrics
    points.HIGH_BP.value_counts()
    #print(points.HIGH_BP.value_counts())

    points.HIGH_BP = points.HIGH_BP.map(int) #implicit encoding of bools as ints in statsmodels libraries are inconsistent

    points.HIGH_BP.mean()
    #print( points.HIGH_BP.mean())

    #Check rescaled data
    print(points.head)
     
    #First logistic regression model-------------------------------------------------------------------------------------
    fit1 = smf.glm(formula="HIGH_BP ~ FRW", data=points,
               family=sm.families.Binomial(sm.families.links.Logit())).fit()
    fit1.summary()
    print(fit1.summary())

    fit1.params
    print("Fitted parameters are:\n",fit1.params)

    plt.scatter(points.FRW, points.HIGH_BP, marker="d")
    X=np.linspace(40, 235, 100)
    plt.plot(X, logistic_transform(X*fit1.params.FRW + fit1.params.Intercept))
    plt.xlabel("FRW")
    plt.ylabel("HIGH_BP")
    plt.show()

    #Second Model with SEX and its interaction -------------------------------------------------------------------------
    fit2 = smf.glm(formula="HIGH_BP ~ sFRW + SEX + SEX:sFRW", data=points,
               family=sm.families.Binomial(sm.families.links.Logit())).fit()
    fit2.summary()
    print(fit2.summary())

    plt.scatter(points.sFRW, points.HIGH_BP, marker="d")
    X=np.linspace(-2, 4, 100)
    p = fit2.params
    plt.plot(X, logistic_transform(X*p.sFRW + p.Intercept), color="red", label="female")
    plt.plot(X, logistic_transform(X*(p.sFRW + p["SEX[T.male]:sFRW"]) +
    p["SEX[T.male]"] + p.Intercept), color="blue",label="male")
    plt.xlabel("Weight")
    plt.ylabel("Pr(Has high BP)")
    plt.legend();
    plt.show()

    #Adding random jitter to the y value of second model
    plt.scatter(points.sFRW, points.HIGH_BP + np.random.uniform(-0.05, 0.05, len(points)),
    marker="d")
    X=np.linspace(-2, 4, 100)
    p = fit2.params
    plt.plot(X, logistic_transform(X*p.sFRW + p.Intercept), color="red", label="female")
    plt.plot(X, logistic_transform(X*(p.sFRW + p["SEX[T.male]:sFRW"]) +
    p["SEX[T.male]"] + p.Intercept), color="blue",label="male")
    plt.xlabel("Weight")
    plt.ylabel("Pr(Has high BP)")
    plt.legend();
    plt.show()

    #Computing the fraction of mispredictions
    error_rate = np.mean(((fit2.fittedvalues < 0.5) & points.HIGH_BP) |
    ((fit2.fittedvalues > 0.5) & ~points.HIGH_BP))
    print("Fraction of mispredictions", error_rate)

    print("Base rate:", 1-np.mean(points.HIGH_BP))

    # Model with Cross-Validation -------------------------------------------------------------------------------------------
    train, test = train_test_split(points, random_state=1)
    print(len(train), len(test))

    # Instantiate the logit link function
    logit_link = sm.families.links.logit()

    # Fit the logistic regression model with the logit link function
    fit = smf.glm(formula="HIGH_BP ~ sFRW + SEX + SEX:sFRW", data=points,
                family=sm.families.Binomial(link=logit_link)).fit()    
    print(fit.summary())

    #print(test.head())
    
    pred = fit.predict(test, transform=True)
    #print(pred.describe())
    #print("Min:", pred.min())
    #print("Max:", pred.max())
    error_rate = np.mean(((pred < 0.5) & (test.HIGH_BP==1)) |
    ((pred > 0.5) & (test.HIGH_BP==0)))
    print(error_rate, 1 - test.HIGH_BP.mean())

    # Better Cross-Validation Model ------------------------------------------------------------------------------------------
    error_model=[]
    error_null=[]
    np.random.seed(1)
    
    for i in range(100):
        train, test = train_test_split(points, random_state=1)
        print(len(train), len(test))

        # Instantiate the logit link function
        logit_link = sm.families.links.logit()

        # Fit the logistic regression model with the logit link function
        fit_penultimate = smf.glm(formula="HIGH_BP ~ sFRW + SEX + SEX:sFRW", data=points,
                    family=sm.families.Binomial(link=logit_link)).fit()    
    print(fit_penultimate.summary())
    
    pred = fit_penultimate.predict(test, transform=True)
    #print(pred.describe())
    #print("Min:", pred.min())
    #print("Max:", pred.max())
    error_rate = np.mean(((pred < 0.5) & (test.HIGH_BP==1)) |
    ((pred > 0.5) & (test.HIGH_BP==0)))
    error_model.append(error_rate)
    error_null.append((1 - test.HIGH_BP).mean())
    #for model, null in zip(error_model, error_null):
    # print(model, null)
    print(pd.Series(error_model).mean(), pd.Series(error_null).mean())

    print(statsmodels.stats.stattools.stats.mannwhitneyu(error_model, error_null,
    alternative="two-sided"))

    # Best Cross-Validation Model ----------------------------------------------------------------------------------------
    points["HIGH_BP2"] = (points.SBP > 140) | (points.DBP > 90)
    points["HIGH_BP2"] = points["HIGH_BP2"].map(int)
    points["HIGH_BP2"].mean()

    error_model=[]
    error_null=[]
    np.random.seed(9)
    for i in range(100):
        train, test = train_test_split(points)
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
    
    # Prediction for individual data point-----------------------------------------------------------------------------------------------------------------------
    point = pd.DataFrame({
        'FRW': [100],
        'CHOL': [200],
        'CIG': [17]
    })

    # Display the DataFrame
    print(point)

    # Join fram and point into a new dataframe called "fram_and_point" along common columns, disregarding all columns not common to both
    fram_and_point = pd.merge(fram, point, on=["FRW", "CHOL", "CIG"], how="inner")


    # Print the final row of (fram_and_point) to debug
    print(fram_and_point.tail(1))

    # Extract final row from "combined_rescaled_df" and set equal to point
    combined_and_rescaled_df = rescale(fram_and_point)
    point = combined_and_rescaled_df.tail(1)

    # Print the head newly rescaled "point"
    print(point.head())

    # Use the fitted model to predict the probability of dangerously high blood pressure
    individual_pred = fit_final.predict(point)

    print("Probability of dangerously high blood pressure:", individual_pred)

if __name__ == "__main__":
    main()

