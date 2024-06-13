## Module 14.1: Model Validation and Imbalanced Data

### Overview

In this lesson, students will delve into the intricacies of model validation. They'll understand the challenges posed by imbalanced data and learn techniques to manage such datasets effectively, ensuring a fair representation in machine learning (ML) models.

### Class Objectives

By the end of today's class, students will be able to:

* Explain the importance of model validation and the associated techniques.

* Identify imbalanced data and apply methods to rectify it.

* Select metrics for model validation.

* Select target data and explain the impact of your decision.

* Identify overfitting and apply methods to rectify it.

---

### Instructor Notes

In this lesson, instructors will guide students through the model validation process. Discussions will also touch on the potential pitfalls and biases that arise due to imbalanced data. By the end of this lesson, students should be equipped with strategies to handle such data and ensure that their models remain unbiased. Much of todays’ class centers around discussion; feel free to change any student dos to everyone dos if you believe that the class discussion would benefit!

---

### Class Slides

The slides for this lesson can be viewed on Google Drive here: [Module 14.1 Slides](https://docs.google.com/presentation/d/1FMtJrXagQ41buw_NYT2RPjEJ-SnfXnqLojXNjbTDHNI/edit#slide=id.g21f2d3f9243_0_462).

To add the slides to the student-facing repository, download the slides as a PDF by navigating to File, selecting "Download as," and then choosing "PDF document." Then, add the PDF file to your class repository along with other necessary files. You can view instructions for this [here](https://docs.google.com/document/d/1XM90c4s9XjwZHjdUlwEMcv2iXcO_yRGx5p2iLZ3BGNI/edit).

**Note:** Editing access is not available for this document. If you wish to modify the slides, create a copy by navigating to File and selecting "Make a copy...".

---

### Time Tracker

| Start Time | Number | Activity                                           | Duration |
| ---------- | ------ | -------------------------------------------------- | -------- |
| 6:30 PM    | 1      | Instructor Do: Introduction to the Class           | 0:05     |
| 6:35 PM    | 2      | Instructor Do: Introducing Bank Marketing Data     | 0:05     |
| 6:40 PM    | 3      | Everyone Do: First Model                           | 0:10     |
| 6:50 PM    | 4      | Everyone Do: Target Selection                      | 0:15     |
| 7:05 PM    | 5      | Instructor Do: Metrics Presentation                | 0:15     |
| 7:20 PM    | 6      | Instructor Do: Metrics                             | 0:10     |
| 7:30 PM    | 7      | Students Do: Metrics                               | 0:15     |
| 7:45 PM    | 8      | Review: Metrics                                    | 0:10     |
| 7:55 PM    | 9      | BREAK                                              | 0:15     |
| 8:10 PM    | 10     | Instructor Do: Overfitting                         | 0:10     |
| 8:20 PM    | 11     | Students Do: Overfitting                           | 0:15     |
| 8:35 PM    | 12     | Review: Overfitting                                | 0:05     |
| 8:40 PM    | 13     | Everyone Do: Bank Targets and Metrics              | 0:10     |
| 8:50 PM    | 14     | Students Do: Second Model                          | 0:25     |
| 9:15 PM    | 15     | Review: Second Model                               | 0:10     |
| 9:25 PM    | 16     | End Class                                          | 0:05     |
| 9:30 PM    |        | END                                                |          |

---

### 1. Instructor Do: Introduction to the Class (5 min)

Open the slideshow and use the first few slides to facilitate your welcome to the class. Cover the following points:

* We now know how to make a model, train it, use it to make predictions, and check if those predictions were accurate.

* In real-world applications, this is only part one of a process. The next step is to use the model to make predictions on new data. It is extremely common for a model to score well in a training environment, but flop when presented with new data.

* The pitfalls we’ll cover this week are common ways that engineers can end up with misplaced confidence in a model that is ill-prepared for real-world performance.

* Overfitting: Overfitting of a model can occur when a model fails to make accurate predictions, and instead fits too closely to training or test data (Nautiyal, n.d.). When this happens, the model contains low bias but high variance, is unable to generalize new data effectively, and will generate inaccurate predictions (Sahani, 2020). Examples of overfitting might include the following:

  * A model whose predictions for home prices change wildly based on the value in the lot_size column.

  * A model that performs perfectly predicts whether a mushroom is poisonous given an image it was trained with, but fails to do so with new images.

* Underfitting: Underfitting of a model can occur when a model is too unsophisticated to capture the underlying trend and complexities within a dataset, resulting in poor performance (Nautiyal, n.d.). In other words, an underfit model cannot determine the relationships between input and output data as the model is too simple. Underfit models contain a high degree of bias but low variance (Sahani, 2020). Examples of underfitting might include the following:

  * A logistic regression model that fails to converge.

  * A medical diagnoses model that always predicts the same value.

  * A home price model where changing square footage doesn’t change the prediction.

* A well-fitted model lies somewhere between an overfitted and underfitted model and has a good balance between bias and variance. These models are able to recognize key trends for both seen and unseen datasets (Amazon Web Services, n.d.).

* Misinterpretation of results: A misinterpretation of results can occur when misunderstanding what a score or metric within a dataset really means in the context of the project. Interpreting an over- or underfit model may extrapolate inaccurate conclusions about the data or make unjustified causal interpretations (Molnar et al., 2022). Examples of misinterpreting results can include the following:

  * A model built to detect bad oranges in a packing facility has 95% accuracy. The engineer believes this means the model is performing well, but doesn’t realize that the model is predicting that *every* orange is good. It just so happens that 95% of oranges are good.

  * A model trained to predict a “total” column with dollar values gets a fabulous MSE, and the data engineer responsible for the column proposes that the model be used to predict which clients will become profitable. The engineer doesn’t realize that the “total” column measures revenue, not profit, and that many high-revenue clients are not profitable.

With this foundation, we will soon delve into the specifics of a bank marketing dataset and use it as we journey through some of the common pitfalls in ML.

---

### 2. Instructor Do: Introducing Bank Marketing Data (5 min)

**Corresponding Activity:** [01-Ins_Introducing_Bank_Marketing_Data](Activities/01-Ins_Introducing_Bank_Marketing_Data/)

Continue using the slideshow to accompany this demonstration.

* Let's start by understanding where our data comes from. This dataset is from a Portuguese bank's marketing campaign and is based on phone calls.The data helps us understand the success rate based on various customer attributes.

* Before opening the file, tell students explicitly that we are setting out to make an example of a bad model. To do this, we’ll take a lot of shortcuts, misunderstand our data, make some mistakes, and misinterpret results.

* On that note, we will spend very little time exploring the data!

Open the solution file and quickly look through the data with the class.

* To import the data and display the first 10 rows of the dataset, use the following code:

```python
# Import the data
import pandas as pd

df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m14/datasets/bank_marketing.csv')
df.head(10)
```

* The data consists of variables, such as ‘age’ (indicating the age of the client), ‘job’ (indicating the industry of the client), and ‘marital’ (indicating the client’s marital status).

* The last input variable is ‘y’. Can anyone guess what 'y' might represent in the context of a bank's marketing campaign?

  * As students give answers, try not to give them any information that is too in depth. The point is to allow students to feel uncomfortable with the fact that we *don’t* know a lot about the target column.

Remember to keep this short. If students have questions, keep the mood light but avoid diving in deep. If students are confused or seem frustrated by the lack of information, feel free to tell them directly that you are being intentionally vague and that it is *good* that they feel uncomfortable. It means they can sense that we are skipping a lot of important steps!

---

### 3. Everyone Do: First Model (10 min)

**Corresponding Activity:** [02-Evr_First_Model](Activities/02-Evr_First_Model/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

Now that we’ve briefly explored the data, this activity will give students the chance to build a “bad model.” The idea is that they will have an opportunity to spot issues and improve the model throughout the module.

* Set the stage, and let students know that they will start with a quick and dirty model, which includes code that has a wide variety of poor practice and missed opportunities.

First, we import the data using the following code:

```python
# Import the data
import pandas as pd

df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m14/datasets/bank_marketing.csv')
df.head()
```

Next, we drop rows with null values, convert the `y` value to numeric, drop all non-numeric columns, and verify the changes with the info method.

```python
# Drop rows with null values
df_clean = df.dropna().copy()

# Convert y to numeric
df_clean['y'] = pd.get_dummies(df_clean['y'], drop_first = True, dtype=int)

# Drop all non-numeric columns
df_clean = df_clean.select_dtypes(include='number')

# Verify changes with the info method
df_clean.info()
```

Next, we set up the X and y variables, and create and train a Random Forest model.

```python
# Setup X and y variables
X = df_clean.drop(columns='y')
y = df_clean['y'].values.reshape(-1,1)

# Create and train a random forest model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

```

Lastly, we display the model’s score to observe how it has performed.

```python
# Check the model's score
model.score(X, y)
```

* A perfect 100% accuracy score!

Ask the students the following:

* Are you skeptical of these results? Hint that they probably should be.

* What aspects are missing from the code? Why is that important? Some example answers are below:

  * We didn’t train_test_split the data, which leaves us without a way to test for overfitting. This could leave us feeling confident in a model that won’t perform well with new data.

  * We removed all rows with null values. We don’t know how many rows were removed or why those rows had missing values.

  * We’re using regular accuracy, which is not a good metric to use with imbalanced data. It could leave us with the impression that the model is performing better than it is.

Having seen a basic model's construction, it’s important to note that our score is still pretty good! Does this mean our model is performing well? An early step to improving a model is finding a metric that aligns with our interpretation of success. To do that, we need to know more about our data.

---

### 4. Everyone Do: Target Selection (15 min)

**Corresponding Activity:** [03-Evr_Target_Selection](Activities/03-Evr_Target_Selection/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

* Provide students with three scenarios consisting of hypothetical datasets. The goal is to explore different choices for target columns and how each might affect the value of and appropriate use cases for a model. The activity will give students an opportunity to think about target selection and to analyze each target column choice in each of the three scenarios that follow. Read each scenario to the students, and discuss each potential target column choice individually. Make sure to push students to engage in the conversation and avoid making this section a lecture.

* *Scenario 1:* You are given a medical dataset with hospital intake information on thousands of patients. The hospital staff would like to use ML to help them make better decisions about which patients should be prescribed antibiotics. The choices for target columns are:

* Choice A: A binary column that indicates whether or not the patient was diagnosed with an infection

* Choice B: A binary column that indicates whether or not the patient responded well to antibiotics while in the hospital

* Choice C: A column that indicates the patient's self-reported health 30 days after antibiotic treatment on a 1&ndash;10 scale

* Now that you have an overview of the three choices, let’s look more closely at each choice and assess how effective each of them would be as the target column.

* Choice A: A binary column that indicates whether or not the patient was diagnosed with an infection

* This choice will lead to a model that can correctly predict whether a patient will be *diagnosed* with an infection. Ask the students what shortcomings they see in this target column choice, using the following questions to prompt them if needed:

* Are diagnoses ever wrong?

* Does a diagnosis alone indicate the best treatment? Consider allergies, patient history, and complicating factors.

* Choice B: A binary column that indicates whether or not the patient responded well to antibiotics while in the hospital

* This choice will lead to a model that can correctly predict whether a patient will respond well to antibiotics. Ask the students what shortcomings they see in this column, using the following questions to prompt them if needed:

* Does “respond well” indicate that the patient has fully recovered? We must be sure that terms like “respond well” are appropriately defined so that there may be no element of subjectivity involved.

* Does responding well to antibiotics necessarily mean that antibiotics were the best choice?

* Choice C: A column that indicates the patient's self-reported health 30 days after antibiotic treatment on a 1&ndash;10 scale

* This choice will lead to a model that can predict how healthy a patient feels 30 days after antibiotic treatment. Ask the students what shortcomings they see in this column, using the following questions to prompt them if needed:

* Is a self-reported health metric valuable? Yes, but perhaps for different reasons. For example, it could help in understanding whether patients will be happy with their treatment, not just whether the treatment was successful.

* Is 30 days the best time frame? Why not 7 or 90 days? Perhaps the patient’s illness would have alleviated anyway after a long period like 30 days.

* *Scenario 2:* You are given a stock market dataset with data on every trade a particular company has made in the previous five years. The company would like to use ML to predict whether a trade will be profitable or not. The choices for target columns are:

* Choice A:  A column that gives the total profit (or loss) from each trade

* Choice B: A column that gives the percentage profit (or loss) from each trade

* Choice C: A binary column that indicates whether or not a trade made at least 10% profit

* Now that you have an overview of the three choices, let’s look more closely at each choice and assess how effective each of them would be as the target column.

* Choice A: A column that gives the total profit (or loss) from each trade
* This choice will produce a regression model that is capable of predicting the exact amount of profit each trade will produce. Ask the students what shortcomings they see in this column, using the following questions to prompt them if needed:

* Do we care how much profit a trade produces? Does it matter if it makes $50 or $51?

* If you buy more or less stock, won’t it change the profit? Wouldn’t a percentage be better?

* Choice B: A column that gives the percentage profit (or loss) from each trade

* This choice will produce a regression model that predicts the percentage of profit that will occur from a trade. Ask the students what shortcomings they see in this column, using the following questions to prompt them if needed:

* Is a regression easier or harder to do well than a classification? Remind students that classification is much easier; it is often better to choose a classification target where possible.

* Do we care whether a trade results in a 7% or 8% profit? Small differences in predicted profit percentages are not likely to make a difference, but big ones could.

* Choice C: A binary column that indicates whether or not a trade made at least 10% profit

* This choice will produce a classification model that will determine whether a trade will make at least 10% profit. Ask the students what shortcomings they see in this column, using the following questions to prompt them if needed:

* Will this model distinguish between a trade that will make 10% profit and one that will make 10,000% profit? No, it will not.

* Are there profitable trades that this model will miss? Any trades between 0% and 10% will be missed.

* *Scenario 3:* You are given a dataset with earthquake records from the past 50 years. The United States Geological Survey  would like to use ML to better predict aftershock impacts after an earthquake. The choices for target columns are:

* Choice A: A column indicating whether or not there was an aftershock after each earthquake

* Choice B: A column with the total economic impact in dollars from the aftershocks after each earthquake

* Choice C: A column indicating the number of lives lost in aftershocks after each earthquake

* Now that you have an overview of the three choices, let’s look more closely at each choice and assess how effective each of them would be as the target column.

* Choice A: A column indicating whether or not there was an aftershock after each earthquake

* This choice will produce a model that determines whether or not there will be an aftershock after a particular earthquake. Ask the students what shortcomings they see in this column, using the following questions to prompt them if needed:

* Is the presence of an aftershock enough to determine the impact of the aftershock?

* Choice B: A column with the total economic impact in dollars from the aftershocks after each earthquake

* This choice will produce a model that will predict the total economic impact (in dollars) of the aftershocks of a particular earthquake. Ask the students what shortcomings they see in this column, using the following questions to prompt them if needed:

* Is an economic impact the only type of impact an aftershock might have?

* How was the impact in dollars measured? Would a measurement by local governments change the bias of the model compared to a measurement done by insurance groups?

* Choice C: A column indicating the number of lives lost in aftershocks after each earthquake

* This choice will produce a regression model that predicts the number of lives lost in the aftershocks of a particular earthquake. Ask the students what shortcomings they see in this column, using the following questions to prompt them if needed:

* Will a model predicting the raw number of casualties bias research toward earthquakes in larger cities? Is that necessarily wrong?

* Could we imagine situations where the aftershock causes more damage than the earthquake itself? Would this target tell us anything about the magnitude of the impact of the aftershocks compared to that of the initial earthquake?

As we've seen, the choice of a target column can significantly impact our model's effectiveness and relevance. Remind students that none of the answers we saw are *wrong*, as each would be appropriate in the correct use case. Always consider the broader implications and ensure alignment with stakeholder objectives. Regardless of the choice of target, make sure you fully understand the *limitations* of your target column before diving into a project.

Continue working through slides and Q&A’s.

---

### 5. Instructor Do: Metrics Presentation (15 min)

Walk through the slides, covering the definitions of the evaluation methods, prediction metrics, problems with accuracy and imbalanced data, balanced accuracy, the AUC-ROC curve, and a comparison of the various methods.

First, define imbalanced data and remind students of the accuracy metric.

* **Imbalanced data** has an uneven distribution of classes in the target column.

* **Accuracy** measures the number of correct predictions that a model makes as a percentage of total predictions.

Then walk the students through the pros and cons of the accuracy metric.

* Pros:

  * Accuracy has a simple output and is easy to calculate.

* Cons:

  * With imbalanced data, a model can achieve high accuracy scores by only predicting the majority class.

  * Accuracy doesn’t explain which errors are being made (false positives or false negatives), and the costs of a false positive and a false negative are rarely equal in real-world situations.

  * Accuracy does not take into account the certainty of the model in its calculation.

Introduce students to the confusion matrix using the following talking points:

* Accuracy can be calculated from the output of a confusion matrix.

* There are two ways in which a model with discrete outcomes’ results can be true, and two ways in which it can be false.

* You have true and false positives, and true and false negatives. In an example of identifying edible mushrooms, a true positive occurs when the model correctly identifies an edible mushroom as edible, and a true negative occurs when the model correctly identifies a poisonous mushroom as not edible. Conversely, a false positive would be the model identifying a poisonous mushroom as edible, and a false negative would be identifying an edible mushroom as not edible.

* These four types of results can be organized into a confusion matrix, a table that groups results according to whether they are predicted as true or false, and whether they are actually true or false. An example of a confusion matrix is illustrated in the following table:

  | Confusion matrix | Predicted to be false | Predicted to be true |
  |------------------|-----------------------|----------------------|
  | Actually false   | True negative (TN)    | False positive (FP)  |
  | Actually true    | False negative (FN)   | True positive (TP)   |

* Generating one of these matrices for a model is simple as scikit-learn already includes a `confusion_matrix` function in the `metrics` package. This can be imported and used in Python with the following block of code that outputs a 2x2 array containing the number of results for each of the four categories.

    ```python
    from sklearn.metrics import confusion_matrix

    confusion_matrix(y_test, predictions)
    ```

* Accuracy is calculated from the confusion matrix as follows:

  $$Accuracy = {(TN + TP) \over (TN + FP + FN + TP)}$$

* In addition to the accuracy metric, there are other metrics that can be calculated from the confusion matrix:

  * **Sensitivity**, sometimes referred to as **recall**, is one of the possible alternatives to accuracy for measuring your model’s success. Using sensitivity as the main unit of measurement allows you to find out how many of the actually true data points were identified correctly.

    * Sensitivity is calculated as follows:

      $$Sensitivity = {TP \over (FN + TP)}$$

    * A model with no false negatives whatsoever has perfect sensitivity. This makes sensitivity the ideal measurement for models wherever avoiding false negatives is your biggest priority. This includes the models that could identify fraudulent credit card charges, poisonous mushrooms or patients with cancer. In these cases, missing a positive result could have severe consequences.

  * Where sensitivity focuses on evaluating how many true data points were identified, **specificity** measures how many of the actually false data points were correctly identified as negative, as shown in the following calculation:

    $$Specificity = {TN \over (TN + FP)}$$

  * Specificity should be a priority for models where false positives are highly undesirable. One example might be in identifying edible mushrooms.

    * When predicting whether mushrooms are edible, specificity would answer the question: When the model predicts that a mushroom is not edible, how often is the model correct?

    * Maximizing specificity in this scenario helps to minimize false positives, reducing the chances of misclassifying a potentially harmful mushroom as safe.

  * **Precision** identifies how many of the predicted true results were actually true, as shown in the following calculation:

    $$Precision = {TP \over (TP + FP)}$$

  * Situations where precision would be the preferred metric include, for example, if you are designing a model to identify bank clients who are a credit risk or to identify areas at high risk for violent crime. Positively identifying a particular person or area could negatively impact the future where someone is wrongly denied a loan or an area becomes over policed.

Introduce students to the classification report using the following talking points:

* A classification report will calculate precision, recall, and F1 score for each class in the model. We'll discuss F1 score shortly. It will also report on the model's accuracy.

* Like the confusion matrix, generating a classification report for a model is simple with scikit-learn. It can be imported and used in Python with the following simple block of code that outputs the full report.

    ```python
    from sklearn.metrics import classification_report

    classification_report(y_test, predictions)
    ```

Explain that the sensitivity, specificity, and precision metrics can help us calculate two other useful metrics:

* The **F1 score** balances sensitivity and precision. The calculation is as follows:

  $$F1 \ Score = {2(Precision*Specificity) \over (Precision + Specificity)}$$

* **Balanced accuracy** measures the accuracy of each class, then averages the results. The calculation is as follows:

  $$Balanced accuracy = {(Sensitivity + Specificity) \over 2}$$

Cover the pros and cons of balanced accuracy:

* Pros:

  * Weighs the accuracy of all classes evenly, regardless of the number of instances in each class.

  * Good metric for imbalanced datasets.

  * Simple to calculate.

* Cons:

  * Does not take into account the certainty of a model.

  * Does not explain whether errors are from false positives or false negatives.

* Using the example confusion matrix slide as a guide, explain how balanced accuracy differs from accuracy.

  * In the case where the data has 70 rows in the positive class and 30 in the negative, assume a model predicts every row to be positive.

  * The model would get 70 predictions correct, which means the accuracy would be 70/100 or 70%.

  * The model got 100% of the positive cases correct and 0% of the negative cases correct. To calculate balanced accuracy, simply average those two outputs: 50%.

Introduce the AUC-ROC curve.

* The **Receiver Operating Characteristics (ROC) curve** visualizes the true positive and false positive rate of predictions using a range of decision thresholds.

* **Area Under Curve (AUC) - ROC**, also referred to as AUC-ROC or simply AUC, is a calculation of the area under the ROC curve, giving a performance metric between 0 and 1.

Before explaining AUC ROC, take a moment to expand on how models make predictions:

* A model outputs a decimal value, normally between 0 and 1.

* That value is rounded to the nearest whole number, which becomes the prediction of the model.

* In some models, it is possible to output the decimal values before rounding.

* With sklearn this is done using the predict_proba method.

* With the decimal values, you can manually change the point at which a prediction becomes a 1 or a 0.

* The **decision threshold** is the decimal value at which a model switches from predicting a 0 to predicting a 1.

Now explain AUC-ROC using the following image from the slides:

![A graph shows the area under the ROC curve with FP Rate as the x-axis and TP rate as the y-axis.](https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_1/img/auc-diagram.png)

* The y-axis represents the true positive rate.

* The x-axis represents the false positive rate.

* The dotted line shows the relationship between true positives and false positives as the decision threshold is increased.

  * This effectively helps measure not just how often the model is correct, but also how certain it is of its correctness.

* A perfect ROC would be vertical at (0,0) and horizontal at (0,1), resulting in a perfect AUC or 1.

* A diagonal line from (0,0) to (1,1) would indicate completely random predictions.

* Like the “accuracy” metric, this means that random guessing would result in an AUC of about 0.5 with balanced datasets.

---

### 6. Instructor Do: Metrics (10 min)

**Corresponding Activity:** [04-Ins_Metrics](Activities/04-Ins_Metrics)

Continue using the slideshow to accompany this demonstration.

Review each metric briefly before continuing to the activity:

* Accuracy is only for balanced datasets where false positives and false negatives are equally costly.

* Balanced accuracy is appropriate for imbalanced datasets, but otherwise similar to accuracy.

* AUC-ROC accounts for the certainty of the model and gives more information than accuracy, but it is difficult to calculate, hard to interpret, and will still struggle with imbalanced data.

* Notice how we interpret each metric in the context of our dataset. Keep these interpretations in mind, as they'll be crucial for your upcoming activity.

Walk through the first few steps in the solved activity, then introduce the balanced accuracy function from sklearn.

* First, we import the data and the necessary items from sklearn:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score

# Import the data
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_1/datasets/app-data-imbalanced.csv")
df.head()
```

* Then, display the total number of positive and negative results. Note the imbalance!

```python
# Show the total number of positive and negative results
df['Result'].value_counts()
```

* Next, split the data into X and y variables.

```python
# Create an X and y variable
X = df.drop(columns=['Result'])
y = df['Result']
```

* Create a logistic regression model, train it using X and y, and calculate the accuracy score.

```python
# Create a Logistic Regression Model
classifier = LogisticRegression()

# Fit (train) or model using the training data
classifier.fit(X, y)

# Calculate the accuracy of the model
classifier.score(X, y)
```

* The accuracy is around 90%! That seems pretty good, but let’s dig deeper by creating a confusion matrix.

```python
# Make predictions on the test data
predictions = classifier.predict(X)

# Create a confusion matrix
print(confusion_matrix(y, predictions, labels = [1,0]))
```

* The top-left and bottom-right numbers are true positive and true negative predictions and together represent all the predictions our model got correct. That’s a lot!

* *However*, pay special attention to the top row. Note that there is a big number in the top right representing false negatives. Said another way, for every true positive we predicted, we had about five false negatives.

* Imagine if this was a screen for cancer. Out of 2,000 people who had cancer, we only correctly identified around 300!

* Accuracy has really let us down. Let’s see if our other metrics do any better at communicating the true predictive power of the model!

```python
# Calculate the balanced accuracy score
print(balanced_accuracy_score(y, predictions))
```

* The balanced accuracy calculation in sklearn is easy to create, we just need the true y values and the model’s predictions.

* The balanced accuracy of our model is only **58%**! That is in stark contrast to the accuracy score we calculated earlier. This is clearly a better representation of the model’s predictive power, which is, frankly, not very good.

* Let’s take a look at the AUC-ROC score next.

* To calculate this score, we need to gather the probability predictions from the model using the predict_proba method.

```python
# Predict values with probabilities
pred_probas = classifier.predict_proba(X)

# Print the probabilities
pred_probas
```

Make sure to point out that the probabilities are listed in pairs. Ask students if they can guess what each pair represents.

* Each pair is associated with a row in the data. The first value is the probability that the row belongs in the 0 class, the second value is the probability that the row belongs in the 1 class.

* Note that each pair adds up to 1.

* To calculate the AUC, we only need the probabilities for the 1 class. To gather the second items from the pairs, we can use the following list comprehension:

```python
pred_probas_firsts = [prob[1] for prob in pred_probas]

# Print the first 5 probabilities
pred_probas_firsts[0:5]
```

* It is possible to create a visualization of the ROC curve, but it isn’t necessary since we’re only calculating the AUC. We can use the following code to calculate the AUC:

```python
# Calculate the roc_auc_score
print(roc_auc_score(y, pred_probas_firsts))
```

Briefly ask students the following:

* Which metric best represents the true performance of the model?

* Does the use case that the model is intended for change which metric might be the best fit?

* Do any of the metrics perfectly represent all the characteristics of the model?

Lead the students to the idea that no metric is perfect, and that in any setting it is important to understand and communicate the limitations of a chosen metric.

---

### 7. Students Do: Metrics (15 min)

**Corresponding Activity:** [05-Stu_Metrics](Activities/05-Stu_Metrics/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

This activity will give students a chance to apply several metrics to a logistic regression model trained to predict whether crowdfunding projects will reach their goal.

---

### 8. Review: Metrics (10 min)

**Corresponding Activity:** [05-Stu_Metrics](Activities/05-Stu_Metrics/)

Let's regroup and review the metrics you've just explored. We'll also discuss the potential pitfalls and shortcomings of each metric.

Open the solution, share the file with the students, and go over it with the class, answering whatever questions students may have.

Cover the following key points during the discussion:

* Ask students to interpret each metric as well as the shortcomings associated with each respective metric.

* Go through the code for ROC-AUC with students.

* Cover the importance of understanding imbalanced data and how it can skew accuracy.

* Review the utility of the confusion matrix in visualizing true positives, true negatives, false positives, and false negatives.

* Discuss the significance of the balanced accuracy score and ROC AUC score, especially in imbalanced datasets.

* Ask students to provide real-world implications and examples where each metric might be more appropriate, or provide some of these examples for them if none are appropriate.

* First, we need to import the data:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score
# Import the data
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_1/datasets/crowdfunding-data-imbalanced.csv")
df.head()
```

* Next, we need to display the total number of positive and negative outcomes in order to check for class imbalances.

```python
# Show the total number of positive and negative outcomes
df['outcome'].value_counts()
```

* Then, we will create an X and y variable along with a logistic regression model. This could be any model, but we’ve chosen logistic regression for simplicity.

* We’ll also train the model and calculate its accuracy.

```python
# Create an X and y variable
X = df.drop(columns=['outcome'])
y = df['outcome']

# Create a Logistic Regression Model
classifier = LogisticRegression()

# Fit the model to the training data
classifier.fit(X, y)

# Calculate the accuracy of the model
classifier.score(X, y)
```

* An 87% accuracy score seems pretty good! Does this mean the model is performing well? Lets create a confusion matrix and a classification report to find out.

```python
# Make predictions on the test data
predictions = classifier.predict(X)

# Create a confusion matrix
print(confusion_matrix(y, predictions, labels = [1,0]))

# Create a classification report
print(classification_report(y, predictions, labels = [1, 0]))
```

Note that students’ results may vary a little based on random state selection, so using approximations can be helpful.

* Well this time it seems that of the 565 crowdfunding projects that were fully funded, the model correctly predicted around 550 of them! That’s great!

* However, of the 101 crowdfunding project that were *not* fully funded, the model only correctly identified 37 of them. Yikes.

* Since our data is imbalanced, perhaps balanced accuracy would be a better choice of metric.

```python
# Calculate the balanced accuracy score
print(balanced_accuracy_score(y, predictions))
```

* Around 65% seems to be a much better interpretation of what we saw in the confusion matrix. Let’s check AUC as well to compare. We’ll first need to use the predict_proba method and extract the predictions for the positive class.

```python
# Predict values with probabilities
pred_probas = classifier.predict_proba(X)

# Print the probabilities
pred_probas

# Each prediction includes a prediction for both the 0 class and the 1 class
# We only need the predictions for the 1 class; use a list comprehension to 
# gather the second value from each list

pred_probas_firsts = [prob[1] for prob in pred_probas]

# Print the first 5 probabilities
pred_probas_firsts[0:5]
```

* With those probabilities, we can calculate the AUC score with the roc_auc_score function.

```python
# Calculate the roc_auc_score
print(roc_auc_score(y, pred_probas_firsts))
```

* An 86% score seems pretty high. Why is AUC showing such a high score?

  * AUC is not impervious to imbalance, and in this instance it seems to be suffering from inflated results.

Conclude this section by letting students know that there are many more metrics available in statistics outside of the ones we’ve covered. If anyone would like to explore more of the metrics available from sklearn, check out sklearn’s [official documentation](https://scikit-learn.org/stable/modules/model_evaluation.html). Answer any questions before moving on.

---

### 9. BREAK (15 min)

---

### 10. Instructor Do: Overfitting (10 min)

**Corresponding Activity:** [06-Ins_Overfitting](Activities/06-Ins_Overfitting/)

Continue using the slideshow to accompany this demonstration. Begin by defining overfitting and underfitting.

* Overfitting: Overfitting of a model can occur when a model fails to make accurate predictions, and instead fits too closely to training or test data (Nautiyal, n.d.). When this happens, the model contains low bias but high variance and is unable to generalize new data effectively. The model will generate inaccurate predictions (Sahani, 2020).

* Underfitting: Underfitting of a model can occur when a model is too unsophisticated to capture the underlying trend and complexities within a dataset, resulting in poor performance (Nautiyal, n.d.). In other words, an underfit model cannot determine the relationships between input and output data as the model is too simple. Underfit models contain a high degree of bias but low variance (Sahani, 2020).

* A well-fitted model lies somewhere between an overfitted and underfitted model, and has a good balance between bias and variance. These models are able to recognize key trends for both seen and unseen datasets (AWS,n.d.).

![An image shows the predictions made by an underfit model, a model of good fit, and an overfit model.](https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_1/img/underfitting-good-fit-overfitting-charts.png)

Emphasize that overfitting and underfitting are both problems that can result in a model performing poorly when introduced to data that it was not trained with.

* Overfitting can result in models that give wildly varying predictions when presented with new data. Underfitting can result in models whose predictions hardly vary at all even with completely different input.

Explain how using hyperparameter tuning can assist in the process of balancing overfitting and underfitting.

* Hyperparameters that change the complexity and construction of a model can be tuned to find the balance between over- and underfitting.

Open the activity and walk the students through the example.

* First, we need to import the data:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import the data
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_1/datasets/app-data.csv")
df.info()
```

* Then, we need to create an X and y variable and split into training and testing sets:

```python
# Create an X and y variable
X = df.drop(columns=['Result'])
y = df['Result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
```

* `Train_test_split` is a way to check whether our model is overfitted. Compare the training score to the testing score, and if the training score is higher, the model is overfitted.

* Next, a model to train and calculate the accuracy score on the testing data:

```python
# Create a Random Forest model
classifier = RandomForestClassifier()

# Fit (train) or model using the training data
classifier.fit(X_train, y_train)

# Calculate the accuracy of the model on the testing data
classifier.score(X_test, y_test)
```

* To understand whether our model is overfit, we need to compare the score of the testing data to that of the training data.

```python
# Calculate the accuracy of the model on the training data
classifier.score(X_train, y_train)
```

Point out the difference between the training score and the testing score. While it isn’t much, our model might still benefit from closing that gap a little.

* One method to balance an overfitted model is by tuning a hyperparameter.

* Walk through the code and show that trying multiple values for a particular hyperparameter and recording the training and testing scores for each attempt allows us to plot the scores and find the optimal value for the parameter. Note that some students may ask how to do this with multiple hyperparameters at once. Inform them that this will be covered in Day 3’s lesson.

* Demonstrate with a code example adjusting a hyperparameter and observing the changes in model performance.

```python
# Create a loop to vary the max_depth parameter
# Make sure to record the train and test scores 
# for each pass.

# Depths should span from 1 up to 40 in steps of 2
depths = range(1, 40, 2)

# The scores dataframe will hold depths and scores
# to make plotting easy
scores = {'train': [], 'test': [], 'depth': []}

# Loop through each depth (this will take time to run)
for depth in depths:
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    scores['depth'].append(depth)
    scores['train'].append(train_score)
    scores['test'].append(test_score)

# Create a dataframe from the scores dictionary and
# set the index to depth
scores_df = pd.DataFrame(scores).set_index('depth')
```

* Now that we’ve gathered the training and testing scores for multiple values of max_depth, we can visualize the performance of the model to find the optimal value.

```python
# Plot the scores dataframe with the plot method
scores_df.plot()
```

* Look at how both the training and testing score rise together until max_depth is around 7. At that point, the training score continues to rise while the testing score starts to flatten out. A max_depth of 7 seems like a good place to set our max_depth.

---

### 11. Students Do: Overfitting (15 min)

**Corresponding Activity:** [05-Stu_Metrics](Activities/05-Stu_Metrics/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

This activity will give students an opportunity to work with training and testing datasets to test their knowledge on the concept of overfitting and how to identify cases of overfitting. They will then use hyperparameter variations on max_depth to determine optimal values that balance overfitting and underfitting.

---

### 12. Review: Overfitting (5 min)

**Corresponding Activity:** [07-Stu_Overfitting](Activities/07-Stu_Overfitting/)

Open the solution, share the file with the students, and go over it with the class, answering whatever questions students may have.

* First, we need to import the data:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import the data
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/lesson_2/datasets/crowdfunding-data.csv")
df.info()
```

* Then, we need to create an X and y variable:

```python
# Create an X and y variable
X = df.drop(columns=['outcome'])
y = df['outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

* Then, we will create, train, and score a Random Forest model:

```python
# Create a Random Forest model
classifier = RandomForestClassifier()

# Fit (train) or model using the training data
classifier.fit(X_train, y_train)

# Calculate the accuracy of the model on the testing data
classifier.score(X_test, y_test)
```

* To spot overfitting, we need to compare the testing score to the training score.

```python
# Calculate the accuracy of the model on the training data
classifier.score(X_train, y_train)
```

* A perfect 100% accuracy on the training data and a 94% accuracy on the testing data is a good sign that our model is overfit.

* Let’s loop max_depth to find the value that best balances between over- and underfitting.

```python
# Create a loop to vary the max_depth parameter
# Make sure to record the train and test scores 
# for each pass.

# Depths should span from 1 up to 15 in steps of 1
depths = range(1, 15)

# The scores dataframe will hold depths and scores
# to make plotting easy
scores = {'train': [], 'test': [], 'depth': []}

# Loop through each depth
for depth in depths:
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    scores['depth'].append(depth)
    scores['train'].append(train_score)
    scores['test'].append(test_score)

# Create a dataframe from the scores dictionary and
# set the index to depth
scores_df = pd.DataFrame(scores).set_index('depth')
```

* Lastly, we need to display the scores with the plot method.

```python
# Plot the scores dataframe with the plot method
scores_df.plot()
```

* Explain the chart showing the training and testing scores with different values for max_depth.

* Emphasize that comparing training and testing scores showed that the model was overfit, but offered no solution to fix it.

* The chart shows where the model becomes overfit, which allows us to avoid it in the first place.

* Emphasize that any hyperparameter could be used here (not just max_depth), and checking every combination of hyperparameter values becomes an infinitely large task very quickly. This method is best reserved for tuning specific parameters, but we’ll explore how to tune hyperparameters more broadly in Day 3’s lesson.

---

### 13. Everyone Do: Bank Targets and Metrics (10 min)

**Corresponding Activity:** [08-Evr_Bank_Targets_and_Metrics](Activities/08-Evr_Bank_Targets_and_Metrics/)

Continue through the slideshow, using the next slides as an accompaniment to this activity. This activity is discussion only with no code. Lean into this and try to get students to engage in conversation!

#### Target Column Selection

Send out the link to the [UCI page](https://archive.ics.uci.edu/dataset/222/bank+marketing) for the dataset and screenshare it to facilitate discussion around Question 1.

* **Question 1:** What is the target column of the dataset actually measuring? Use the UCI page for this dataset to find information about the target column and how the data was collected.

  * Explore the UCI page, and point out specifically the Variables Table area where features are described.

  * Use the page button to go to Page 2 of the variables and view the description of the y column.

* Given the information on the page, one answer to Question 1 might be: “Whether a client has subscribed to the bank term deposit given a set of information regarding the marketing they have received.”

* Note that our interpretation of the target includes a description of the features we have available for use in training.

You may use your own interpretation if it differs. You can make a point to disagree with the “textbook” interpretation if you would like! The point is that this is not a straightforward determination.

* **Question 2:** If a model becomes well trained in predicting this target, which of these statements can be made and which cannot?

  * Statement 1: The model can predict which clients will subscribe before any marketing efforts have been made.

  * Statement 2: The model can predict which clients currently in the marketing pipeline will eventually subscribe.

  * Statement 3: The model can predict which clients currently in the marketing pipeline have already subscribed.

* Only the third statement is applicable.
  
  * The other features the model has access to are only available after the client has entered the marketing pipeline. Unless we can gather information like age, job, marital status, and education from another source, our model is unlikely to be useful before a client is present in the marketing database. Statement 1 is likely to cause confusion among stakeholders for this reason and should be avoided.

  * The y column states whether a client *has subscribed* to the term deposit, not whether they will eventually subscribe. This is an important thing to communicate! To predict whether the client will eventually subscribe, we would need to track the same clients for a certain period of time and combine their eventual subscription status with the data we had earlier. Since that is not the data we have access to, Statement 2 is not valid.

As the instructor, you can again raise concerns with these interpretations. Ensure to point out that these conversations are often more subjective than students would expect!

* **Question 3:** Imagine you are meeting with the board of the bank. Given your answers to the previous question, how would you describe the use case of a model that is well trained on this dataset? Remember, your answers should be understandable by non-technical people.

* One answer might be as follows: “A model trained on this data will be excellent at determining which customers in our marketing pipeline have already or will soon become subscribers *without further marketing*. We could use this forecasting to preemptively drop customers from our marketing pipeline to focus on the clients that are not yet likely to have subscribed.”

If students are confused here, lean into it and explain how difficult it can be to accurately describe the use case for a model trained with specific data. Often, research and collaboration to determine the right description and use case for a model can be incredibly complex. If we were employed by this bank, it would be wise to ask many clarifying questions to people on different teams before solidifying a “pitch” for your model.  

#### Metric Selection Section

In this section you will guide students through selecting a metric to use as the primary measuring stick for our model.

* **Question 1:** Is the dataset balanced or imbalanced?

  * The dataset is imbalanced as there are far more of one class than the other. Specifically there are around 30,000 rows in the negative class and only about 4,000 rows in the positive class.

* **Question 2:** Given the use case described in the previous section, what is the risk associated with a false positive prediction? How about with a false negative prediction? Is the cost of one of the errors higher than that of the other?

  * A false positive would drop a client from our marketing list prematurely, causing us to stop marketing to a client who is not yet a subscriber and potentially losing out on recurring revenue.

  * A false negative would continue marketing for a client that is already a subscriber, wasting marketing resources.

* **Question 3:** What metrics would be best suited for this project given the answers to the previous questions?

  * Many answers are possible, including AUC and balanced accuracy.

  * Regardless of the answers they give, make sure students are considering the fact that the data is imbalanced; accuracy would be a poor choice, and AUC may not be suitable either.

  * Again, feel free to share your own thoughts on this question and remind students that this decision is not straighforward.

* **Question 4:** Assume balanced accuracy is chosen by the managers of the project. Are there any caveats or risks you would make them aware of given that choice?

  * Balanced accuracy does not differentiate between false positives and false negatives. If there is a higher cost to continuing to market to an already subscribed customer compared to stopping marketing to a customer who has *not* yet subscribed, we may need to take additional measures.

  * Balanced accuracy does not take into account the model's certainty of a particular prediction (the predict_proba method). If the model is only 51% certain that a client will become a subscriber, balanced accuracy will interpret that prediction exactly the same as a prediction where the model is 100% certain.

Now that we've done a deep dive into the targets and metrics, it's your turn to try to rebuild and assess a model on the bank marketing data!

---

### 14. Students Do: Second Model (25 min)

**Corresponding Activity:** [09-Stu_Second_Model](Activities/09-Stu_Second_Model/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

This activity will give students an opportunity to work with the previous model, but they will instead use balanced accuracy as a metric. They will have a chance to test the model for overfitting and attempt to tune the max_depth parameter in the Random Forest model to prevent over- and underfitting. If time allows, students can attempt to recreate a better model.

---

### 15. Review: Second Model (10 min)

**Corresponding Activity:** [09-Stu_Second_Model](Activities/09-Stu_Second_Model/)

* Let's regroup and discuss our findings. I'm curious to hear about your models and evaluations!

Open the solution, share the file with the students, and go over it with the class, answering whatever questions students may have.

Cover the following key points during the discussion:

* Open the provided solution model.

* First, we import the data, drop rows with null values, convert y to numeric, drop all non-numeric columns, and verify changes with the info method:

```python
# Import the data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m14/datasets/bank_marketing.csv')
df.head()

# Drop rows with null values
df_clean = df.dropna().copy()

# Convert y to numeric
df_clean['y'] = pd.get_dummies(df_clean['y'], drop_first = True, dtype=int)

# Drop all non-numeric columns
df_clean = df_clean.select_dtypes(include='number')

# Verify changes with the info method
df_clean.info()

```

* Next, let’s set up the X and y variables, and then create and train a Random Forest model.

```python
# Setup X and y variables
X = df_clean.drop(columns='y')
y = df_clean['y'].values.reshape(-1,1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
# Create and train a logistic regression model
from sklearn.ensemble import RandomForestClassifier

# Create and train a random forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

* We then check the model’s balanced accuracy on the test and training set.

```python
# Check the model's balanced accuracy on the test set

y_pred = model.predict(X_test)
print(balanced_accuracy_score(y_test, y_pred))

# Check the model's balanced accuracy on the training set

y_train_pred = model.predict(X_train)
print(balanced_accuracy_score(y_train, y_train_pred))
```  

* The balanced accuracy on the test set is ~0.710 and the training set is 1.0. This is massively overfit!

* Engage in a comparison of the original model's accuracy with the second model. Emphasize that initially we started believing that the model was performing very well, but in reality it was hardly better than a random guess.

* Discuss potential reasons for any observed discrepancies. Was the model overfit?

* Then, we try different values for the max_depth hyperparameter.

```python
# Try the following values for max_depth

max_depths = range(1, 10)
models = {'train_score': [], 'test_score': [], 'max_depth': []}

# Loop through each value in max_depths
for depth in max_depths:
    clf = RandomForestClassifier(max_depth = depth)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    train_score = balanced_accuracy_score(y_train, train_pred)
    test_score = balanced_accuracy_score(y_test, test_pred)

    models['train_score'].append(train_score)
    models['test_score'].append(test_score)
    models['max_depth'].append(depth)

# Create a dataframe from the models dictionary with max_depth as the index
models_df = pd.DataFrame(models).set_index('max_depth')
```

* Lastly, we need to plot the results using the plot method:

```python
# Plot the results
models_df.plot(l)
```

* Highlight the way the training and testing scores diverge at around a max_depth between 3 and 7.

* Ask students if anyone had differing solutions and share any exemplary models from students while discussing their approaches. Perhaps point out a snippet of the key differences in their code and have them share their final balanced accuracy score.

* Address any lingering questions and ensure conceptual clarity.

* Conclude this activity by asking the following questions:

  * The model we started class with had an accuracy of 100%, and our new model only has a balanced accuracy of ~60%. Did our model get worse?

    * No! Our evaluation of the model became far better, and we learned how poor our model truly was.

  * Why did we need to change the max depth? The balanced accuracy got worse!

    * The model was **overfit**, and our balanced accuracy was not likely to translate to new data!

Great discussions everyone! As we wrap up today's session, let's reflect on what we've learned and preview what's coming next.

---

### 16. End Class (5 min)

Congratulate students on completing this lesson.

Remind students what they learned in this class:  

*Target selection within a dataset, which is effectively the variable whose values are being modeled
*Overfitting and underfitting
*Various prediction metrics such as accuracy, balanced accuracy, ROC-AUC curve, and each of their respective pitfalls

In the next lesson, we'll dive into data preprocessing pitfalls and strategies to avoid them, ensuring our models are both robust and reliable.

**Reflection and feedback:** At the end of each day or module, encourage students to reflect on what they've learned and provide an avenue for them to give feedback. This could be a quick survey, a discussion forum, or an open-ended questionnaire.

*What was your biggest takeaway from today's lesson?

*Was there any concept you found challenging?

*How do you intend to apply what you've learned today?

*Do you have any suggestions for improving this lesson or topics you'd like to see covered in more depth?

---

## References

Allwright, S. 2022. *What is a good balanced accuracy score? Simply explained*. Available: [2023, October 17].

Amazon Web Services. n.d. *What is overfitting?* Available: [2023, October 5].

Evidently AI. n.d. *How to explain the ROC curve and ROC AUC score?* Available: [2023, October 17].

Molnar, C., König, G., Herbinger, J., Freiesleben, T., Dandl, S., Scholbeck, C.A., Casalicchio, G., Grosse-Wentrup, M. et al. 2022. General pitfalls of model-agnostic interpretation methods for machine learning models. In *xxAI - Beyond Explainable AI*. A. Holzinger, R. Goebel, R. Fong, T. Moon, K-R. Müller & W. Samek. Eds. Lecture notes in computer science. Switzerland: Springer, Cham.

Moro, S., Rita, P. & Cortez, P. 2012. *Bank marketing* [Dataset]. UCI Machine Learning Repository. Available: https://archive.ics.uci.edu/dataset/222/bank+marketing [2023, October 17]. *Note that this data set has been modified/cleaned/shortened for the purpose of this activity. You can access the modified data in the CSV file provided.* ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode))

Nautiyal, D. n.d. *ML | Underfitting and overfitting*. Geeks For Geeks. Available: [2023, October 3].

Sahani, G.R. 2020. *Elucidating bias, variance, under-fitting, and over-fitting*. Medium. Available: [2023, October 5].

---

© 2023 edX Boot Camps LLC. Confidential and Proprietary. All Rights Reserved.
