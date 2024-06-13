## Module 14.2: Advanced Preprocessing Techniques

### Overview

In this lesson, students will explore advanced preprocessing techniques to enhance data quality and usability for machine learning (ML) models. They will tackle challenges related to data leakage, missing values, and encoding strategies, and learn how to apply various preprocessing functions effectively.

### Class Objectives

By the end of today's class, students will be able to:

* Recognize and address data leakage in datasets.

* Apply innovative methods to handle missing values in data.

* Evaluate and select appropriate encoding strategies for categorical data.

* Use OneHotEncoder and OrdinalEncoder for data transformation.

* Ensure prevention of data leakage during train&ndash;test data splits.

* Construct preprocessing functions to streamline data preparation.

* Design and incorporate new features to enhance ML model performance.

---

### Instructor Notes

During this lesson, we'll further explore concepts covered in prior modules. While some ideas might feel familiar, our aim is to delve deeper, applying these principles to datasets and revealing their impact on ML modeling.

---

### Class Slides

The slides for this lesson can be viewed on Google Drive here: [Module 14.2 Slides](https://docs.google.com/presentation/d/1h2hF1oqJGWT4KF9WrIt-I9I0em7BBKKEGeJppwVHfsQ/edit#slide=id.g21f2d3f9243_0_462).

To add the slides to the student-facing repository, download the slides as a PDF by navigating to File, selecting "Download as," and then choosing "PDF document." Then, add the PDF file to your class repository along with other necessary files. You can view instructions for this [here](https://docs.google.com/document/d/1XM90c4s9XjwZHjdUlwEMcv2iXcO_yRGx5p2iLZ3BGNI/edit).

**Note:** Editing access is not available for this document. If you wish to modify the slides, create a copy by navigating to File and selecting "Make a copy...".

---

### Time Tracker

| Start Time | Number | Activity                                           | Duration |
| ---------- | ------ | -------------------------------------------------- | -------- |
| 6:30 PM    | 1      | Instructor Do: Introduction to the Class           | 0:05     |
| 6:35 PM    | 2      | Instructor Do: Bank Model Review                   | 0:05     |
| 6:40 PM    | 3      | Instructor Do: Understanding Data Leakage          | 0:20     |
| 7:00 PM    | 4      | Students Do: Spotting Data Leakage                 | 0:10     |
| 7:10 PM    | 5      | Review: Spotting Data Leakage                      | 0:10     |
| 7:20 PM    | 6      | Everyone Do: Missing Values                        | 0:20     |
| 7:40 PM    | 7      | Everyone Do: Choosing Encodings                    | 0:20     |
| 8:00 PM    | 8      | BREAK                                              | 0:15     |
| 8:15 PM    | 9      | Instructor Do: Feature Engineering                 | 0:15     |
| 8:30 PM    | 10     | Students Do: Third Model                           | 0:40     |
| 9:10 PM    | 11     | Review: Third Model                                | 0:15     |
| 9:25 PM    | 12     | End Class                                          | 0:05     |
| 9:30 PM    |        | END                                                |          |

---

### 1. Instructor Do: Introduction to the Class (5 min)

* Welcome back, everyone! Today, we dive into advanced preprocessing, a pivotal aspect of Machine Learning (ML). We'll navigate data leakage challenges and explore strategies to refine datasets for peak model performance.

* In this lesson, you will learn how to identify and prevent data leakage within datasets using innovative methods, such as imputation to rectify missing values in data and encoding text to numeric values. For encoding, you will make use of OneHotEncoder and OrdinalEncoder from sklearn. You will also explore some preprocessing techniques to streamline data preparation and practice designing new features, in the form of feature engineering, to enhance ML model performance.

---

### 2. Instructor Do: Bank Model Review (5 min)

**Corresponding Activity:** [01-Ins_Bank_Model_Review](Activities/01-Ins_Bank_Model_Review/)

Continue using the slideshow to accompany this demonstration.

Let’s review the bank model we used in Day 1’s lesson.

Begin by importing and presenting the bank dataset. Highlight the challenges faced in prior lessons, particularly with balanced accuracy and hyperparameter tuning.

In Day 1’s lesson, you may recall that we worked on the second model. To refresh, we imported the data, dropped rows with null values, converted y to numeric, and then dropped all non-numeric columns. We then created X and y variables, split the data into training and test sets, then created and trained a Random Forest model. The model’s balanced accuracy on the test and training set was 0.71 and 1.0 respectively, which indicated that the data was extremely overfitted. Thereafter, we compared the second model with the original model and realized that the initial thought that the model performed well was not the case. Then, we tried hyperparameter tuning, but it did not meaningfully improve the model. In the end, the evaluation of the model improved, and we learned that the model was not very good to begin with. The balanced accuracy metric deteriorated after changing the max_depth parameter. The model was overfitted, and the balanced accuracy was unlikely to translate to new data.

It's evident that hyperparameter tuning alone couldn't resolve the overfitting of the imbalanced data. Notice the rows we omitted due to missing values and the non-numeric columns we eliminated. Can preprocessing be the key?

Let's dive deeper into one such preprocessing challenge called data leakage, which can cause inflated scores during training and testing if left unchecked.

---

### 3. Instructor Do: Understanding Data Leakage (20 min)

**Corresponding Activity:** [01-Ins_Bank_Model_Review](Activities/01-Ins_Bank_Model_Review/)

Continue using the slideshow to accompany this demonstration.

* Data leakage occurs when a model is trained using information that will not be available when making predictions in production. Data leakage results in excellent training and testing scores that do not translate into real-world success.

* Let’s go through some examples of data leakage:

* **Example 1:** If the target column is accidentally left in the X data, the model will make perfect predictions during training. Of course, when in production, the target column won’t be available, and that’s why we need a ML model to make predictions!

* **Example 2:** A model is trained on data that contains a “row_number” column.
This is a form of data leakage. You should never use “id” or “row_number” columns in training, especially if they were added after the data was collected. The best case scenario is that they aren’t useful to the model at all, but worst case scenario is that the column contains hidden information (we’ll see an example of this in the demo).

* **Example 3:** A model is trained to predict whether an item will sell on eBay. The target column is simply “sold” and is either a 1 or a 0. In the X data, there is another column named “sale_price”, which contains a -1 if the item didn’t sell or the sale price if the item did sell. The sale price column contains information that will only be known after an item auction finishes. This is another example of data leakage.

* **Example 4:** X and y variables are created from a dataset, then the X values are scaled, and the data is split into training and testing sets. This is hard to spot, but the scaling function used information about the testing set to transform the training set! This is data leakage, and while this type of leakage is especially problematic when dealing with time series data, it can still affect model performance on any dataset. To avoid this type of leakage, always split data into training and testing sets before doing any preprocessing. Fit any transformations (like one-hot encoding, scaling, etc.) on just your training data, and then apply those transformations to the testing data.

Ask the students if they can think of any other examples of data leakage.

* Open the activity, and go through it together with the students.

* First, we need to import the data:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Import the data
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_2/datasets/app-data-leakage.csv")
df.head()
```

* Point out that the app number column appears to be random. This column should not be useful to a model.

* Create an X and y variable:

```python
# Create an X and y variable
X = df.drop(columns=['Result'])
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 13)
```

* Then, we need to create a logistic regression model, train the model using the training data, and calculate the accuracy of the model with the training data:

```python
# Create a Logistic Regression Model
classifier = RandomForestClassifier(random_state=13)

# Fit (train) our model using the training data
classifier.fit(X_train, y_train)

# Calculate the accuracy of the model with training data
classifier.score(X_train, y_train)
```

* Point out that the training and testing scores of our first model are both 100%. Is this a perfect model? This seems too good to be true. Be skeptical, as a testing score of 100% might allude to data leakage.

* Check the correlation of columns within the dataset.

```python
# Check correlation of columns
df.corr()['Result'].sort_values()
```

* Point out that the correlation between Result and app_number is 86%; this is clearly data leakage. There should be zero correlation between a truly random ID number and the target column. Why would the ID number have any predictive value?

* Lastly, plot app_number and Result in a scatter plot:

```python
# Plot app_number and Result in a scatter plot
df.plot(kind='scatter', x='app_number', y='Result')
```

* Point out that the plot of the data shows that any app number above ~15,000 is malware and anything under ~15,000 is not malware. The app_number column must have been created while the data was sorted by the target column.

* We found this instance of data leakage by being suspicious of “too good to be true” results, which prompted us to examine the correlation between the target column and another suspicious column. However, emphasize that there is no one “correct” way to identify data leakage. Creativity and watchfulness are critical!

* Discuss real-world implications of data leakage using illustrative examples.

* The eBay example is particularly compelling. In the eBay example, if an item didn’t sell, the variable sale_price was marked with a -1, and if it sold, then it would be marked at the sale price. The issue is the sale_price is only known after the auction has been completed, so data leakage occurs. The model should not have access to that variable as it will result in overly optimistic predictions, as it uses information “from the future” to make predictions about the past.

* Data leakage can occur whenever “future information” is used to make predictions. This could result in misleading predictions, and could translate to expensive and misinformed decision-making.

* Emphasize the need for skepticism when confronted with results that seem too good to be true.

Understanding data leakage is just half the battle. Let's see how well you can spot it in practice!

---

### 4. Students Do: Spotting Data Leakage (10 min)

**Corresponding Activity:** [03-Stu_Spotting_Data_Leakage](Activities/03-Stu_Spotting_Data_Leakage/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

In this activity, you'll attempt to pinpoint data leakage in given datasets. Remember the strategies we discussed. It's a blend of understanding the data and vigilance.

Once you've given it a shot, we'll reconvene and discuss our findings.

---

### 5. Review: Spotting Data Leakage (10 min)

**Corresponding Activity:** [03-Stu_Spotting_Data_Leakage](Activities/03-Stu_Spotting_Data_Leakage/)

* Open the solution, share the file with the students, and go over it with the class, answering whatever questions students may have.

* Walk through the activity steps, noting that these steps are just examples of how data leakage might be spotted.

#### Part 1: Crowdfunding

* First, import the data:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import the data
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_2/datasets/crowdfunding-data-leakage.csv")
```

* Then, create an X and y variable and the Random Forest model, train the model using the training data, and then calculate the accuracy of the model with the training data.

```python
# Create an X and y variable
X = df.drop(columns=['outcome'])
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 13)
# Create a Random Forest Model
classifier = RandomForestClassifier(random_state=13)

# Fit (train) or model using the training data
classifier.fit(X_train, y_train)

# Calculate the accuracy of the model with training data
classifier.score(X_train, y_train)
```

* Calculate the accuracy of the model with testing data.

```python
# Calculate the accuracy of the model with testing data
classifier.score(X_test, y_test)
```

* The accuracy of the model outputs 1.0, a perfect score. This should serve as a warning that there is a high chance of data leakage. Let’s check for any potential data leakage.

* Identify any columns that could be leaking data.

```python
# Identify any columns that could be leaking data
df.head()
```

* Calculate the correlation of columns, as it would show which variables may be problematic in terms of data leakage.

```python
# Check correlation of columns
df.corr()['outcome'].sort_values()
```

Let the students discuss, and encourage them to share their thoughts, but guide the conversation through the values and context we can glean from each column.

* The values in the rewards given column don’t seem suspicious, but the context does. In many crowdfunding systems, rewards are only given to backers if the goal is reached. Is it possible that this column is leaking information about the success of the campaigns?

* Plot rewards_given and outcome in a scatter plot. This will help us visualize the potential data leakage.

```python
# Plot rewards_given and outcome in a scatter plot
df.plot(kind='scatter', x='rewards_given', y='outcome')
```

* Note that all of the rows with an outcome of 0 also have 0 rewards given. This column indirectly contains the information we are trying to predict! This is data leakage.

#### Part 2: Start Up Success

* Display content from the CSV file, create an X and y variable, create a Random Forest model, train the model using training data, and calculate the accuracy of the model with both training and then testing data:

```python
df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_2/datasets/start-up-success-leakage.csv')
# Create an X and y variable
X = df.drop(columns=['Firm Category'])
y = df['Firm Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 13)

# Create a Random Forest Model
classifier = RandomForestClassifier(random_state=13)

# Fit (train) or model using the training data
classifier.fit(X_train, y_train)

# Calculate the accuracy of the model with training data
classifier.score(X_train, y_train)
# Calculate the accuracy of the model with testing data
classifier.score(X_test, y_test)
```

* Identify any columns that could be leaking data, and check the correlation of columns as abnormally highly correlated columns may also allude to data leakage.

```python

# Identify any columns that could be leaking data
df.head()

# Check correlation of columns
df.corr()['Firm Category'].sort_values()
```

* Lastly, plot Firm ID and Firm Category in a scatter plot so we can visually inspect the data for unusual patterns.

```python
# Plot Firm ID and Firm Category in a scatter plot
df.plot(kind='scatter', x='Firm ID', y='Firm Category')
```

* Especially point out that Firm ID in step 2 does not show a very high correlation to the Firm Category column. Binary columns do not always perform as expected when correlation is calculated. A plot is helpful to spot this issue.

* Discuss common areas where students correctly identified data leakage.

* Address frequent oversights or misconceptions that students had.

* Provide any additional insights you have into nuanced data leakage scenarios not covered in the activity.

* Conclude the activity by reminding students that the best tools in the fight against data leakage are skepticism and domain knowledge.

With a clearer understanding of data leakage and how to spot it, you're better equipped to handle and preprocess datasets for optimal ML model performance.

---

### 6. Everyone Do: Missing Values (20 min)

**Corresponding Activity:** [04-Evr_Missing_Values](Activities/04-Evr_Missing_Values/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

Having delved into the foundations of preprocessing, let's continue by addressing a common issue in datasets: missing values.

* Datasets are not always complete. Often, they may contain missing values, which can introduce elements of bias into a dataset. Dropping rows with missing data can also dramatically reduce the amount of data that we are able to train with!

* In addition, missing data values are rarely a random occurrence, and there is usually a specific reason that a value may be missing. Discovering what the reason is for the missing values is pivotal for preprocessing, and can assist us in formulating the best strategy to fill the data.

* Imputation can be one possible solution to working with missing values. Imputation involves the use of alternative values in place of missing data. More complex techniques may also be employed, depending on the dataset and its specific features.

* Emphasize that the key to identifying missing data lies in domain knowledge. The better you understand the data, the easier it may be to spot and address issues relating to missing data. It’s a good habit to always begin by inspecting the dataset and attempting to identify any patterns, and considering potential reasons for any missing values.

* Remember, imputing missing values is not just about filling gaps; it's about ensuring the integrity and reliability of your dataset.

* Let’s walk through the code. First, import the data:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
# Import the data
import numpy as np
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_2/datasets/crowdfunding-missing-data.csv")
df.head()
```

* Then, we need to split into training and testing datasets, and find the percentage of rows missing values in each column.

* Remember to start by splitting the data into train and test, then start working on ONLY the training data to avoid data leakage. If you work on test data, you may encounter data leakage issues.

```python
# Split into training and testing sets
X = df.drop(columns = 'outcome')
y = df['outcome'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)

# Find the percentage of rows missing values in each column
X_train.isna().sum()/len(df)
```

* The backers count is missing from about 10% of data. This is a lot!

* Describe the other columns in the rows with missing values. We want to examine any stark differences between the rows that are missing values and the rows that are not.

```python
# Describe the other columns in the rows with missing values
X_train.loc[X_train['backers_count'].isna()].describe()
```

* Describe the whole dataset:

```python
# Describe the whole dataset
X_train.describe()
```

* By describing the data from the whole dataset and comparing it to the output for only the rows where backers count is missing, we can look for any differences that stand out. Ask the students what stands out to them. They should notice the following:

  * The minimums of every column stayed about the same.

  * The means and maxes changed a good bit, especially in the days active column.

* Since there seems to be a major difference in the days active column for rows missing a backers count, let’s make a quick visualization to see if we can come up with an explanation.

* For this specific situation, it might be helpful to create two histograms, one showing the values of days active when backers count is missing and the other showing days_active across the whole dataset.

```python
# Perform other exploratory analysis here
# For this specific example, try making histograms
# of days_active for the whole dataset and then
# again only when backers_count is missing.

X_train['days_active'].plot(kind='hist', alpha=0.2)
X_train.loc[df['backers_count'].isna(), 'days_active'].plot(kind='hist')
print(X_train.loc[df['backers_count'].isna(), 'days_active'].unique())
```

* Point out that the rows with missing data have all been active for less than a week! Maybe that column isn’t populated until a week after the campaign launches?

* If so, could we use campaigns in Week 2 and find their backers count to help impute the values for campaigns in Week 1?

* Walk through calculating the mean of backers count for campaigns in Week 2.

* Create the X_preprocess function to fill data with half of the mean of Week 2 backers counts.

  * Note that we should still calculate this value using only the training data to avoid data leakage. If the model gets used in production for extended periods of time, this value may be updated regularly to match more recent data.

* Run the X_preprocess function on both training and testing data.

* Print the count of nulls for all columns in both training and testing sets.

* We need to fill in the data using the backers counts from campaigns in Week 2.

```python
# Since backers_count seems to be missing in the first week
# of a campaign, removing the data would be detrimental.
# A good choice might be to fill the data using the backers
# counts from campaigns in week 2.

mean_of_week_2_backers_counts = X_train.loc[(X_train['days_active'] >= 6) & (X_train['days_active'] <= 13), 'backers_count'].mean()
mean_of_week_2_backers_counts
```

* Create a function to fill missing values with half of the mean of Week 2, preprocess the training and testing data, and then finally check for missing values.

```python
# Create a function to fill missing values with half of the mean of week 2

def X_preprocess(X_data):
    X_data['backers_count'] = X_data['backers_count'].fillna(int(round(mean_of_week_2_backers_counts/2)))
    return X_data
# Preprocess the training and testing data

X_train_clean = X_preprocess(X_train)
X_test_clean = X_preprocess(X_test)
# Check for missing values
print(X_train_clean.isna().sum()/len(X_train_clean))
print(X_test_clean.isna().sum()/len(X_test_clean))
```

Finally, ask the students the following questions:

1. Beyond losing roughly 10% of the data for training, what other impacts might have come from dropping all rows with missing data?

* We would have lost **all** data on campaigns that were less than a week old. That’s a big deal!

2. Was the value we chose to fill in the best possible choice? Should we have found some way to factor in the overall goal of the project or the current pledged amount?

* Factoring in the overall goal of each project to our calculation may well have made a better fill in value! That said, it’s easy to end up overcomplicating the imputation process, and it doesn’t always result in a better model. Be judicious in choosing where to spend time!

3. Sometimes the fact that a value is missing is important information in and of itself. How could we have changed our process to preserve that information for the model?

* There are two common ways that data scientists represent missing or unknown values. First, an additional one-hot column named “backers_count_missing” could be created with zeros where the backers count was present and ones where it was missing. Second, since the backers count column can never be negative naturally, we could fill the missing values with -1 to indicate that the value was missing.

---

### 7. Everyone Do: Choosing Encodings (20 min)

**Corresponding Activity:** [05-Evr_Choosing_Encodings](Activities/05-Evr_Choosing_Encodings/)

Now that we've tackled missing values, let's transition into the world of categorical data. Why might encoding be necessary, and how do we choose the right technique? Continue through the slideshow, using the next slides as an accompaniment to this activity.

* Encoding, in a ML context, refers to the transformation of categorical variables from strings into numbers. Encoding may be necessary as raw data often includes textual or categorical information. Many ML models **require** numeric inputs, so we are left with the choice of dropping those columns or converting them into numbers somehow.

* **Categorical** variables will take on one type of a fixed set of values. There are two types of categorical variables that we will cover today, nominal and ordinal variables.

* **Nominal** variables don’t have an inherent order and can be thought of as categories, which can be distinguished from each other (Cavin, 2022). Nominal variables can be encoded using OneHotEncoder from sklearn.

* *Ordinal variables*, on the other hand, have an inherent order, and can be ranked in an ascending or descending fashion. These variables can be encoded using the Ordinal Encoder from sklearn.

* Ordinal encoding is valuable when there's a clear order to the categories, but it should be used cautiously! Consider the following example:

  * A column of As, Bs, and Cs is converted to 1s, 2s, and 3s using an ordinal encoder. The ML model is now trained to believe that C > B and that 2 x A = B! Make sure that your data makes sense when expressed numerically whenever using ordinal encoding.

* Your choice of encoding should always align with the nature of the data and the problem you're addressing.

* Note that encoding a variable is harder when we are trying to avoid data leakage. What happens if data in the testing set includes a category that wasn’t in the training set? How do we handle that?

* Point out that OneHotEncoder and OrdinalEncoder both have parameters to define how unknown values AND missing values are dealt with. This way we can build a preprocessing function that will work even with unexpected values in new data!

* Open the activity, and walk through it with the students.

* Tell students to bookmark the [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) and [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) documentation as they may need to refer to it from time to time.

* As you progress through the following code, ensure that the students are thinking about the appropriate parameter settings for each encoder on each column.

* First, we need to import the data:

```python
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
# Import the data
df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_2/datasets/text-data.csv')
df.head()
```

* Next, create X and y variables, and split into training and testing sets.

```python
# Create X and y and split into training and testing sets
X = df.drop(columns='arrived')
y = df['arrived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
```

* Decide how to encode the backpack_color column. Let’s see what values the backpack_color column can take on.

```python
# Decide how to encode the backpack_color column
X_train['backpack_color'].value_counts()
```

* Because there is no inherent order to the values, it seems that backpack_color contains nominal data.

* Create a one-hot encoder for the backpack_color column. Remember to tell the encoder how to handle unknown values! In the OneHotEncoder, ‘ignore’ will simply assign zeros to all output columns if the value to encode is unknown. Setting sparse_output to False will make the output easier to convert back to a pandas DataFrame.

```python
# Create an encoder for the backpack_color column
backpack_color_ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Train the encoder
backpack_color_ohe.fit(X_train['backpack_color'].values.reshape(-1,1))
```

* Note that we have to use the reshape method to train the encoder!

* Next, decide how to encode the grade column. Let’s see what values the grade column can take on.

```python
# Decide how to encode the grade column
df['grade'].value_counts()
```

* The grade column seems to contain ordinal values!

* Create an encoder for the grade column, and train the encoder. This ensures that the input data is in the correct format.

* The OrdinalEncoder function has several parameters that should be specified:

  * Categories should contain a double bracketed list of values, listed in order from smallest to largest.

  * encoded_missing_value will set a value to encode if the encoder encounters a NaN or missing value.

  * The handle_unknown parameter tells the encoder whether to throw an error or do something else whenever a value is encountered that the encoder wasn’t trained with, for example, in this grade column if we were later to try to encode a value of “B+”. We want to specify a value, so we will choose ‘use_encoded_value’.

  * The unknown_value parameter allows us to specify the value that will be encoded when the encoder encounters a missing value. This value will only be used if the handle_unknown parameter is set to ‘use_encoded_value’.

```python
# Create an encoder for the grade column
grade_ord_enc = OrdinalEncoder(categories = [['F', 'D', 'C', 'B', 'A']], encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)

# Train the encoder
grade_ord_enc.fit(X_train['grade'].values.reshape(-1,1))
```

* Decide how to encode the favorite_creature column.

```python
# Decide how to encode the favorite_creature column
df['favorite_creature'].value_counts()
```

* Create an encoder for the favorite_creature column, and train the encoder.

```python
# Create an encoder for the backpack_color column
creature_ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, min_frequency=0.2)

# Train the encoder
creature_ohe.fit(X_train['favorite_creature'].values.reshape(-1,1))
```

* Next, we’ll create a function using the pretrained encoders to use on any new data, including test data.

* This step is not strictly a necessity, but it can be incredibly helpful if you plan to use a model regularly with new data.

```python
# Create a function using the pretrained encoders to use on
# any new data (including the testing data)

def X_preprocess(X_data):
    # Transform each column into numpy arrays
    backpack_color_encoded = backpack_color_ohe.transform(X_data['backpack_color'].values.reshape(-1,1))
    grade_encoded = grade_ord_enc.transform(X_data['grade'].values.reshape(-1,1))
    favorite_creature_encoded = creature_ohe.transform(X_data['favorite_creature'].values.reshape(-1,1))

    # Reorganize the numpy arrays into a DataFrame
    backpack_color_df = pd.DataFrame(backpack_color_encoded, columns = backpack_color_ohe.get_feature_names_out())
    creature_df = pd.DataFrame(favorite_creature_encoded, columns= creature_ohe.get_feature_names_out())
    out_df = pd.concat([backpack_color_df, creature_df], axis = 1)
    out_df['grade'] = grade_encoded

    # Return the DataFrame
    return out_df
```

* Next, preprocess the training and testing data.

```python
# Preprocess the training data
X_preprocess(X_train)

# Preprocess the testing data
X_preprocess(X_test)
```

* Use the documentation to explain any choices the students didn’t think of (especially sparse output).

---

### 8. BREAK (15 min)

---

### 9. Instructor Do: Feature Engineering (15 min)

**Corresponding Activity:** [06-Ins_Feature_Engineering](Activities/06-Ins_Feature_Engineering/)

Cleaned your data? Check. Encoded categorical variables? Check. Now, let’s elevate our dataset further by crafting new features. Let's uncover the magic of feature engineering. Continue using the slideshow to accompany this demonstration.

* Feature engineering refers to the conversion of raw observations into features that can be used in supervised ML (Patel, 2021). Feature engineering can help a model uncover relationships in the data that might have otherwise stayed hidden.

* Simple operations, like computing ratios or aggregating data, can sometimes unveil hidden patterns or detect outliers.

* Features can be added by combining data with external sources, such as using the zip code of an entry to add weather information, using a client’s identification number to add sales information, and so on. Features can also be created by performing operations on existing columns, such as multiplying number_of_orders by avg_order in order to obtain total_order_value.

* Feature engineering is all about creativity and domain knowledge. It's an iterative process of hypothesis and validation. The creation of useful features can often lead to increased model efficiency and is preferred over a more complex model.

* Remember, while adding more features can provide more information, it can also introduce noise. Always validate the effectiveness of a new feature before implementing it.

* Open the activity and walk through it briefly with the students.

* First, we need to import the data:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import the data
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/datasets/crowdfunding-data.csv")
df.head()
```

* Next, create a column “pledged_per_backer”. This step will create a new column that is calculated by dividing the pledged amount by the backers_count for each row. It will allow us to see how much has been pledged per backer, which may be a useful ratio for the model when making predictions.

```python
# Create a column "pledged_per_backer"
df['pledged_per_backer'] = df['pledged'] / df['backers_count']

df.head()
```

* Then, fill in the missing values with zeros. This step converts “NaN” values to “0” for data completeness purposes, so that the data is suitable for any further analysis. The “NaN” values occurred only in places where there were no backers, so filling them with zeros makes sense.

```python
# Fill the missing values with zeros
df['pledged_per_backer'] = df['pledged_per_backer'].fillna(0)
df.head()
```

* Next, create a “backers_per_day” column. This step gives us an idea of how many backers a campaign is gaining on a daily basis.

```python
# Create a backers_per_day column
df['backers_per_day'] = df['backers_count'] / df['days_active']

df.head()
```

* Lastly, create a “days_to_goal” column. This will give us an idea of how many days it would take to reach any stated fundraising goals.

```python
# Create a days_to_goal column
def days_to_goal(row):
    amount_remaining = row['goal'] - row['pledged']
    pledged_per_day = row['pledged_per_backer'] * row['backers_per_day']
    # Note that we can't divide by zero:
    # return a large number if pledged_per_day is zero
    if pledged_per_day == 0:
        return 10000
    return (amount_remaining)/(pledged_per_day)

df['days_to_goal'] = df.apply(days_to_goal, axis=1)
df.head()
```

* These columns give the model insight into relationships that seem obvious to us, but may not be obvious mathematically.

* Before concluding, ask the students whether adding these features would help a model make better predictions.

  * We don’t know! We would need to compare model results with and without the columns, perhaps even performing feature selection through VIF or using some other method to make sure we’re avoiding multicollinearity.

* Continue to the next activity where you can practice your newly learned skills on the bank marketing dataset.

---

### 10. Students Do: Third Model (40 min)

**Corresponding Activity:** [07-Stu_Third_Model](Activities/07-Stu_Third_Model/)

Equipped with a deeper understanding of preprocessing and feature engineering, it's time to put these concepts into practice. Let's embark on building our third model and witness the transformation of our efforts.

As the instructor, you can decide between making this a group or solo activity. The decision should be based on student confidence and class dynamics.

* Let students know that they can start with the model provided in the first activity of today’s class, they can start from scratch, or they can use any work they’ve done on their own.

This activity will give students the opportunity to practice all the new skills they’ve learned on the bank marketing dataset. Before they begin, intro the activity with the following points.

* Remind students to start by assessing the dataset. Given the preprocessing steps we've explored today, which ones do you feel are pertinent for this task?

* Let students know that they should experiment with different encoding and imputation techniques, and evaluate how each different encoding and imputation technique impacts model performance.

* Remind students about feature engineering. Feature engineering is your playground. Think outside the box and consider novel features that might be relevant.

* Lastly, ensure students validate their model adequately. A well-preprocessed dataset paired with proper validation strategies is a recipe for a robust model.

Monitor students’ progress; if any students are struggling, feel free to send them the solution file. Instead of writing their own code, give students the option to read and understand the code in the solution instead.

---

### 11. Review: Third Model (15 min)

 **Corresponding Activity:** [07-Stu_Third_Model](Activities/07-Stu_Third_Model/)

Having dedicated time to crafting our third model, it's crucial we come together to share insights, challenges, and breakthroughs. This collaborative review can provide diverse perspectives on the task. Have students share their findings, and encourage students to share their screens whenever possible to show the code they’ve written. Remember that the solution file is merely an example, and students’ code can and should look completely different. It is far better to focus on the students’ code rather than to walk through the solution!

Cover the following key points during the discussion:

* Approximately 80% of the work that data scientists do relates to preprocessing. A significant portion of a data scientist’s time is spent ensuring a high degree of data quality and integrity. Often in data science, collaboration and discussion pave the way for innovative solutions and insights. Let's dive into our results.

* By a show of hands, who found their balanced accuracy score significantly improved through preprocessing? Let's hear about the techniques you applied.

* Did anyone outperform the solution file? If so, what techniques led to that improvement? Did the amount of work involved in preprocessing surprise anyone?

* It's okay if the solution file outperforms your model; the objective is continuous learning. Remember, sometimes the simplest preprocessing can yield the most dramatic results.

* Finding the right preprocessing steps for a particular dataset can take a lot of trial and error; if none of your attempts helped, don’t get discouraged!

---

### 12. End Class (5 min)

Today, we've journeyed through some intense preprocessing techniques, model iterations, and collaborations. As we wind down, let's take a moment to reflect and project into our upcoming lessons.

* In this lesson, you’ve learned about data leakage and what it means for the integrity of datasets, the application of innovative methods in order to handle missing values, encoding strategies, preprocessing techniques to streamline data preparation, and enhancing model efficiency.

* In the next lesson, you will continue working with the bank marketing model, delve deeper into hyperparameter tuning, explore resampling techniques, and tweak a model to enhance its overall efficiency using the techniques we have covered.

**Reflection and feedback:** At the end of each day or module, encourage students to reflect on what they've learned and provide an avenue for them to give feedback. Use the following questions as a jumping-off point and update based on that day’s learnings.

Take a few minutes to reflect on your learning:

* Were there any specific preprocessing techniques that resonated particularly well or seemed especially effective?

* Has your perception of the importance of preprocessing in data science shifted or evolved throughout today's lesson?

* Is there anything you're still curious about, or are there any lingering questions you hope to address in future lessons?

* What did you find most challenging today, and what felt like a breakthrough?

---

## References

Cavin. A. 2022. *6 Ways to encode features for machine learning algorithms*. Medium. Available: [2023, October 10].

Patel. H. 2021. *What is feature engineering — importance, tools and techniques for machine learning*. Medium. Available: [2023, October 10].

—--

© 2023 edX Boot Camps LLC. Confidential and Proprietary. All Rights Reserved.
