## Module 14.3: Tuning Models and Sampling Data

### Overview

In this lesson, students will explore advanced techniques for model tuning and data resampling. By applying these methods to a bank marketing dataset, students will gain hands-on experience refining their predictive models.

### Class Objectives

By the end of today's class, students will be able to:

* Perform hyperparameter tuning.

* Use random and synthetic resampling to address class imbalance.

* Apply new techniques to bank marketing data.

* Demonstrate the application of an existing model to new data.

---

### Instructor Notes

In today's lesson, emphasize the significance of hyperparameter tuning and highlight the challenges of class imbalances in datasets. Illustrate your points with real-life examples. Maintain class engagement by pausing regularly for questions and fostering discussions.

---

### Class Slides

The slides for this lesson can be viewed on Google Drive here: [Module 14.3 Slides](https://docs.google.com/presentation/d/1qlR3kp6GbHZuclKCdwFahUgUFnNujk93rhkcg-YqLYI/edit#slide=id.g21f2d3f9243_0_462).

To add the slides to the student-facing repository, download the slides as a PDF by navigating to File, selecting "Download as," and then choosing "PDF document." Then, add the PDF file to your class repository along with other necessary files. You can view instructions for this [here](https://docs.google.com/document/d/1XM90c4s9XjwZHjdUlwEMcv2iXcO_yRGx5p2iLZ3BGNI/edit).

**Note:** Editing access is not available for this document. If you wish to modify the slides, create a copy by navigating to File and selecting "Make a copy...".

---

### Time Tracker

| Start Time | Number | Activity                                           | Duration |
| ---------- | ------ | -------------------------------------------------- | -------- |
| 6:30 PM    | 1      | Instructor Do: Introduction to the Class           | 0:05     |
| 6:35 PM    | 2      | Instructor Do: Review Bank Marketing Model         | 0:10     |
| 6:45 PM    | 3      | Everyone Do: Hyperparameter Tuning                 | 0:15     |
| 7:00 PM    | 4      | Instructor Do: Resampling                          | 0:10     |
| 7:10 PM    | 5      | Students Do: Bank Marketing Resampling             | 0:15     |
| 7:25 PM    | 6      | Review: Bank Marketing Resampling                  | 0:05     |
| 7:30 PM    | 7      | Everyone Do: What Else?                            | 0:15     |
| 7:45 PM    | 8      | BREAK                                              | 0:15     |
| 8:00 PM    | 9      | Students Do: Fourth Model                          | 0:50     |
| 8:50 PM    | 10     | Review: Fourth Model                               | 0:10     |
| 9:00 PM    | 11     | Students Do: New Data                              | 0:15     |
| 9:15 PM    | 12     | Review: New Data                                   | 0:10     |
| 9:25 PM    | 13     | End Class                                          | 0:05     |
| 9:30 PM    |        | END                                                |          |

---

### 1. Instructor Do: Introduction to the Class (5 min)

Welcome the students, and explain that in today’s lesson they will be reviewing the work we’ve done on the bank marketing model as well as exploring hyperparameter tuning and resampling. In the previous lessons, we’ve focused on measuring our performance and preprocessing data in human readable ways. Today we’ll focus on the model itself. How can we adjust the model to improve performance?  

At the end of today’s lesson, students will have the opportunity to practice their skills on the bank marketing data.

---

### 2. Instructor Do: Review Bank Marketing Model (10 min)

**Corresponding Activity:** [01-Ins_Third_Model_Review](Activities/01-Ins_Third_Model_Review/)

Open the solution, and begin with the objective of providing a broad overview of the bank marketing model thus far. We have performed the following actions on the model:

*We’ve cleaned the data to ensure no errors carry through to further analysis.

*Then, we proactively chose a performance metric, the balanced accuracy score, and tuned the model to guard against overfitting.

*Next, we implemented hyperparameter testing in an effort to improve the model’s accuracy and the balanced accuracy score.

*Lastly, we charted the results.

Discuss the importance of defining the target column and the process behind it.

* Dive into metrics, their significance, and the rationale behind their selection (balanced accuracy score, ROC-AUC score, predict_proba).

* Highlight the challenges and solutions for handling missing values (data leakage, hyperparameters, imputation, encodings).

* Wrap-up with the encoding of categorical data, its relevance, and implications.

Avoid going through the code line by line. Instead, give a broad-strokes overview of what has been done to the data so far and ask the students if they have any thoughts or questions.

* Does anyone have questions on the following topics?

* Defining the target column
 *Selecting and implementing metrics
 *Filling missing values
 *Encoding categorical data

* If there aren’t any questions, feel free to cut this section short and save time for the activities at the end of class.

---

### 3. Everyone Do: Hyperparameter Tuning (15 min)

**Corresponding Activity:** [02-Evr_Hyperparameter_Tuning](Activities/02-Evr_Hyperparameter_Tuning/)

Continue using the slideshow to accompany this demonstration.

* Start by defining hyperparameter tuning and its pivotal role in model accuracy. Hyperparameter tuning refers to the process of adjusting or tweaking a model’s parameters for optimal model performance. Hyperparameter tuning can have anywhere from no impact to a dramatic impact on model performance; the results will depend on the data.

* Introduce students to tools like [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), emphasizing their functionalities and differences.

* GridSearchCV may be used to obtain optimal values for the hyperparameter of a model. Manually adjusting hyperparameter values in search for the optimal value is time consuming, and GridSearchCV is used to speed up the process. The GridSearchCV function loops through predefined hyperparameters and fits the model on the training set.

* RandomizedSearchCV is an alternative method to obtain the optimal values for the hyperparameter of a model. Instead of trying every single combination of hyperparameter values like GridSearchCV, RandomizedSearchCV checks randomly selected values. This method uses less resources and time, yet delivers similar results (Wang, 2021).

* Note to students that understanding the detailed functionality of every hyperparameter in every model is a lifelong pursuit. Until students have enough experience with machine learning (ML) to understand the implications of each parameter in more depth, using tools like RandomizedSearchCV can automate a trial-and-error approach with surprisingly comparable results to manual tuning from a more senior ML engineer.

* Open the unsolved file in Jupyter notebook and go over the following details:

  * The dataset has been artificially generated using the `make_moons` dataset for `sklearn`.

  * After the dataset has been created, we create a scatter plot to visualize the data and then we can start creating a model.
  
  * The dataset is split into testing and training sets for validation.

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    ```

  * Next, we create a support vector machine (SVM) model with the `rbf` kernel, since our data is non-linear, as the following code shows:

    ```python
    from sklearn.svm import SVC 
    model = SVC(kernel='rbf')
    model
    ```

  * Now we will optimize two hyperparameters of the SVM model: `C` and `gamma`:

    * `C` is the regularization parameter that can be used to reduce overfitting. The strength of the regularization is inversely proportional to `C`. In other words, a smaller `C` means a stronger regularization.

    * `Gamma` determines how far the influence of a single observation reaches. It is typically used for non-linear hyperplanes.

    * Make sure students know that it’s ok if they don’t fully understand these parameters. However, if they’d like to learn more, they should explore the sklearn documentation, which has extensive definitions and guides available for each model.

    * There are four values each for `C` and `gamma`, as the following code shows:

      ```python
      # Optimize two hyperparameters of the SVM model.
      param_grid = {'C': [1, 5, 10, 50],
                'gamma': [0.0001, 0.0005, 0.001, 0.005]}
      ```

  * Next, explain that we use the `GridSearchCV` module to search through the best `param_grid` values, as the following code shows:

      ```python
      from sklearn.model_selection import GridSearchCV
      grid_clf = GridSearchCV(model, param_grid, verbose=3)
      ```

    * The `GridSearchCV` module runs a `for` loop with all the different possible pairings of the `param_grid` values.
    * The first argument of the `GridSearchCV` method is the `SVM` model.
    * The second argument is the dictionary of `param_grid` values.
    * The third, optional argument specifies the level of details printed to the screen.

  * Then, we train the model.

    ```python
    # Fit the model by using the grid search classifier. 
    # This will take the SVC model and try each combination of parameters.
    grid_clf.fit(X_train, y_train)
    ```

  * The output from fitting the model will produce the results of 80 total fits: **5 folds for each of the16 candidates**.

  * We can pick out the best parameters from the 80 possible combinations by using the `best_params_` attribute, as the following code shows::

    ```python
    # List the best parameters for this dataset
    print(grid_clf.best_params_)
    ```

    * The best parameters are `{'C': 50, 'gamma': 0.005}`.

  * Similarly, we can get the accuracy using the `best_score_` attribute, as the following code shows:

    ```python
    # List the best score
    print(grid_clf.best_score_)
    ```

    * Running the code will produce an accuracy of 87.33%.

  * Next, we can use our optimized model to make predictions, as the following code shows:

    ```python
    # Make predictions with the hyper-tuned model.
    predictions = grid_clf.predict(X_test)
    predictions
    ```

    * Running the code will produce an array of binary values.

  * Next, we can get the score of the model's performance on the testing data, as the following code shows:

    ```python
    # Get the score on the testing data.
    grid_clf.score(X_test, y_test)
    ```

  * Running the code will produce an accuracy of 86%.

  * Finally, we generate a classification report, as the following code shows:

    ```python
    # Calculate the classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_test, predictions,
                                target_names=["class1", "class2"]))
    ```

    * The classification report shows that the model has a high accuracy of 86%, but it is not predicting at a high precision whether the test data should be in `class1` or `class2`.

  * Next, explain that we can use `RandomizedSearchCV` instead of `GridSearchCV`.

    * `GridSearchCV` performs an exhaustive search for optimal hyperparameters, but it can be computationally expensive. In contrast, `RandomizedSearchCV`, selects hyperparameter combinations randomly and is useful when there are many hyperparameters to optimize.

  * Walk through the code and note the similarities between `GridSearchCV` and `RandomizedSearchCV`.

  * Point out that with `RandomizedSearchCV` we can generate large arrays of values for `C` and `gamma`, as the following code shows:

      ```python
      big_param_grid = {
          'C' : np.arange(0, 100, 1),
          'gamma': np.arange(0, 0.01, .0001),
      }
      big_param_grid
      ```

  * Explain that instantiating `RandomizedSearchCV` is similar to the process for `GridSearchCV`. The model and parameter grid are specified and the number of iterations can also be specified, as the following code shows:

      ```python
      # Create the randomized search estimator along with a parameter object containing the values to adjust
      from sklearn.model_selection import RandomizedSearchCV
      random_clf = RandomizedSearchCV(model, big_param_grid, n_iter=100, random_state=1, verbose=3)
      random_clf
      ```

  * The model is trained.

    ```python
    random_clf.fit(X_train, y_train)
    ```

    * This time we'll generate a total of 500 fits: **5 folds for each of the 100 candidates**.

  * Explain that the optimal parameters can be identified from the trained model as well as the accuracy score of those parameters.

    ```python
    # List the best parameters for this dataset
    print(random_clf.best_params_)
    # List the best score
    print(random_clf.best_score_)
    ```

    * Running the code will produce an accuracy of 88%.

  * Next, we can use our optimized model to make predictions based on the testing data.

    ```python
    # Make predictions with the hyper-tuned model
    predictions = random_clf.predict(X_test)
    ```

  * Finally, we generate a classification report, as the following code shows:

    ```python
    # Calculate the classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_test, predictions,
                                target_names=["class1", "class2"]))
    ```
  
    * The classification report returns the same results as the `GridSearchCV` model, which is expected since the data is dummy data.

* Answer any questions before moving on.

---

### 4. Instructor Do: Resampling (10 min)

* Sampling refers to the use of a subset of data from the entire population. Data resampling refers to the process of making repeat subsets of the population in an effort to enhance the accuracy and limit uncertainty (Ambati, 2021). Resampling ensures robustness of a model and allows it to perform well regardless of data variability. The model is enhanced when trained on multiple differentiated patterns available in the dataset through the use of synthetic data for imbalanced data or replicated data from the original dataset (Ambati, 2021).

* Inform students that data resampling is NOT magic and will never be a replacement for collecting more “real” data.

*Resampling tends to repeat information about the existing minority class and sometimes the information that gets repeated is actually random noise.

* NEVER resample data in the testing set! Resampling is ONLY to be used for the training set.

  * The point of resampling data is to provide the model with more examples of the minority class during training. Remember that in production, we won’t know which row belongs to which class, so resampling will be impossible!

* Open the activity and show some of the code.

* IMPORTANT: Focus on the repetitive nature of the code; there are many sampling algorithms, but they are all implemented with almost identical code.

* There is no need to walk line by line through every section, but point out that the actual code is nearly identical for every form of resampling. The key is to understand the differences between the models and try several versions to see if any help.

  * Review the code as follows:

  * First, we import the `RandomUnderSampler` from the `imblearn.under_sampling` package

  * The minority class is value 1. In the original dataset, we had 3,753 versus 34 instances in the training set. The `RandomUnderSampler` model will randomly choose which of the majority class labels not to keep until it is the same size as the minority class.

  * Using `value_counts()` on the `y_resampled` data, we find that the resampled sets are both 34.

* Now, explain the following points, and add the code that follows.

* Now, we fit a Random Forest classifier to the resampled dataset. Then we generate a set of test predictions for the original training set, as the following code shows:

    ```python
    # Instantiate an initial RandomForestClassifier instance
    model = RandomForestClassifier()

    # Fit the initial model based the training data
    model.fit(X_train, y_train)

    # Instantiate a second RandomForestClassifier instance
    model_resampled = RandomForestClassifier()

    # Fit the second model based the resampled data
    model_resampled.fit(X_resampled, y_resampled)

    # Make predictions using the initial model
    y_pred = model.predict(X_test)
    ```

* Next, we make predictions using the model based on the resampled data and plot the original `y_test`, the predictions based on the original test data, and the resampled data, as the following code shows:

    ```python
      # Make predictions using the model based on the resampled data
      y_pred_resampled = model_resampled.predict(X_test)

      # Plot the data using the original y_test information
      plt.scatter(
          x=X_test[0],
          y=X_test[1],
          c=y_test)
      plt.show()

      # Plot the data using the predictions based on the original test data
      plt.scatter(
          x=X_test[0],
          y=X_test[1],
          c=y_pred)
      plt.show()

      # Plot the data using the predictions based on the resampled test data
      plt.scatter(
          x=X_test[0],
          y=X_test[1],
          c=y_pred_resampled)
      plt.show()
      ```

* Point out that `y_pred_resampled` identified many more points as being part of the minority class.
  
* Finally, we print the classification report with `y_pred` and `y_pred_resampled` datasets, as the following code shows:
  
    ```python
    # Print classification report
    print(classification_report(y_test, y_pred))
    print(classification_report(y_test, y_pred_resampled))
    ```

* Answer any questions students have about the process for sampling, then quickly scan through the rest of the code.

* While discussing the code, ensure to highlight the similarities in the patterns of implementation across different methods, making it easier for students to grasp the fundamental concepts.

* As you work through the rest of the code, define each of the sampling techniques:

  * Random undersampling removes randomly selected instances from the majority class to reduce the size of the class, typically to the size of the minority class.

  * Random oversampling randomly selects instances of the minority class, duplicates them, and adds them to the training set until the majority and minority classes are balanced.

  * **Cluster centroids** are a synthetic method of resampling data to achieve a balance between the majority and minority classes. Cluster centroids will identify clusters of data from the majority class, calculate the centroids of those clusters, then replace those clusters with fewer data points using new, synthetic data to represent each cluster.

  * **SMOTE:** The **synthetic minority oversampling technique** (SMOTE) is another synthetic resampling technique. It also works by using k-nearest neighbors to generate synthetic data. However, instead of balancing the imbalanced classes, SMOTE creates synthetic points in the minority class, which are oversampled until they are the same size as the majority class.

  * **SMOTEENN** is a combination of SMOTE oversampling and undersampling by removing misclassified points using edited nearest neighbors (ENN).

    * **ENN** is a rule that uses the three nearest neighbors to find a misclassified point and then remove it. You can use the following [`EditedNearestNeighbors` documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html) as a reference.

* Close this section by asking students how they might decide which resampling method to use in practice. Example answers are below:

  * If there are too few instances of the minority class, undersampling would remove too much data, so oversampling might be a better choice.

  * If there is extreme class imbalance, synthetic methods may have better luck in maintaining all the information from the original data.

  * If no method is the obvious choice, try several! It’s always a good idea to try multiple methods and compare results.

Answer any questions before introducing the next activity.

---

### 5. Students Do: Bank Marketing Resampling (15 min)

**Corresponding Activity:** [05-Stu_Bank_Marketing_Resampling](Activities/05-Stu_Bank_Marketing_Resampling/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

In this activity, students will use random sampling techniques in an attempt to balance the bank marketing data.

* Make sure you emphasize that students need not go through EVERY resampling technique; just do a few and compare them.

---

### 6. Review: Bank Marketing Resampling (5 min)

**Corresponding Activity:** [05-Stu_Bank_Marketing_Resampling](Activities/05-Stu_Bank_Marketing_Resampling/)

Share the solution file with the students, and lead a brief discussion on the following points with the class:

* The differences in classification reports for models trained on original, undersampled, and oversampled data

* The impact of class imbalance on model performance

* How undersampling and oversampling can change model predictions

Make sure to note that the results from this activity are NOT universal; other datasets and models may respond differently to different methods! Before moving on, answer any questions the students have about the code or about resampling methods.

---

### 7. Everyone Do: What Else? (15 min)

Continue through the slideshow, and lead a discussion with students about what else could be done to improve our model.

* Thus far, we have explored various metrics, such as balanced accuracy, ROC-AUC score, and target selection within datasets. You also learned how the problem of data leakage and missing data can be overcome through preprocessing, hyperparameter tweaking, imputation, encoding, and feature engineering. Some of the aforementioned techniques can enhance the accuracy of the model. However, there are limitations on the current dataset and challenges in predicting customer behavior.

* Questions to ask students:

  * What methods have we learned that we have NOT yet applied to the data? Possible answers include PCA, feature selection with variance inflation factors, scaling, etc.

  * Given the dataset we have, if we had infinite time to work on our preprocessing and on our model, is a balanced accuracy of 100% achievable? Why, or why not? An example answer: No, because the data is noisy. Two clients with the exact same values in all the available columns could easily make two different choices about whether to sign up.

  * Methods will often have zero or a negative impact on model performance. That is normal. Getting a little bit of improvement takes a combination of skill, luck, and patience.

  * What balanced accuracy score do you think *is* possible given unlimited time? Many answers are acceptable. Maybe we’ve already found the max! Maybe with a bit more time, we could get up to 65%. A score of 70% would be amazing, and 80% seems too good to be true.

* Is there any additional data that the bank could collect that would help our model? Example answers below:

  * Has the client responded well to prior marketing?

  * How long has the client been with the bank?

  * How often does the client make deposits?

---

### 8. BREAK (15 min)

---

### 9. Students Do: Fourth Model (50 min)

**Corresponding Activity:** [05-Stu_Fourth_Model](Activities/05-Stu_Fourth_Model/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

* You have the option of making this activity a Student Do, Everyone Do, or Group Do, depending on the dynamics of the class.

* If you think the students will benefit from it, send out the example solution at the start of the activity. This solution is not the “correct” answer in the typical sense; it is merely an example of how someone might format their ML model. Specifically, this solution takes the previous solution and refactors it, focusing on cleaning up the primary notebook by moving the definitions for preprocessing functions into the ml_utils.py file and importing them.

This activity will give students the freedom to use any of the techniques they’ve learned in order to create the best possible model.

* As students progress with the activity, check in with them. Keep the conversation light, and allow them to share any “in the moment” successes or observations, such as model improvements, instances of overfitting, etc.

In the next section, we'll review your approach to resampling the bank marketing dataset and discuss the results.

---

### 10. Review: Fourth Model (10 min)

**Corresponding Activity:** [05-Stu_Fourth_Model](Activities/05-Stu_Fourth_Model/)

Ask the following questions during the discussion:

* Did any changes the students make have significant impacts on the models’ performance?

---

### 11. Students Do: New Data (15 min)

**Corresponding Activity:** [06-Stu_New_Data](Activities/06-Stu_New_Data/)

Continue through the slideshow, using the next slides as an accompaniment to this activity.

This activity will give students the opportunity to use all the techniques that they have learned to improve the fourth model with new data.

* Share the link for the new dataset from the bank

* Tell students to attempt to import this new data, apply their preprocessing steps, and predict the data using their model. Make sure they understand that they should NOT train their model with the new data. They should use the model that they already trained to make predictions only.

* Have them calculate their balanced accuracy score if they are successful in importing and processing the data.

After this hands-on experience, let's come together to discuss our observations and insights from using the model on new data.

---

### 12. Review: New Data (10 min)

**Corresponding Activity:** [06-Stu_New_Data](Activities/06-Stu_New_Data/)

Open the solution, share the file with the students, and go over it with the class, answering whatever questions students may have.

* Ask the students how they felt about the activity. Did anyone improve their model? Were there any unsuccessful attempts? Did anyone’s model behave very well on the original data but poorly on the new data? Encourage students to share their experiences with everyone, as it is a fantastic opportunity for students to learn from and engage with each other.

* Encourage students to share even if they weren’t successful. It is HIGHLY likely that students struggled with this.

* Discuss how the students think the best possible model would perform. Is 70% or 80% balanced accuracy possible? (Make sure you discuss your own opinions and reasoning.)

* A possible opinion: “Perhaps 65% may be the maximum with the data we currently have. There is a lot of noise in the data, and a perfect model is definitely outside the realm of possibility.”

* Discuss the differences in model performance on the training dataset versus the new dataset.

* Discuss potential reasons for any observed discrepancies in performance.

* Discuss any adjustments or techniques that could improve the model's predictions on new data.

* Let students know that we spend a lot of time on this process in this module. Approaching problems like this more quickly is something that can only be learned through practice and repetition. Feel free to mention where to find datasets to practice with, as well as places like kaggle.com that specialize in hosting educational competitions using ML.

* The solved version of this activity is an EXAMPLE ONLY and students’ final products will vary widely. The solution was created with refactoring in mind and may serve as a good example of how to better organize large ML projects like this one.

* Send the solution out if you feel it would be useful for anyone who feels like their code got too “messy” or disorganized. Refactoring is a great skill to learn, and sometimes seeing someone else’s code can help with ideas on how to better organize your own!

As we near the end of today's session, let's wrap up by reflecting on our main takeaways and looking ahead to our next steps.

---

### 13. End Class (5 min)

Congratulations on completing this module. In this lesson, you explored various techniques all with the objective of improving a model’s performance. You went deeper into hyperparameter tuning, the use of random and synthetic resampling to address class imbalance within a dataset, applied new techniques to the previous bank marketing dataset that we worked with, and then applied the techniques to a new model’s dataset.

The content we covered is quite challenging and you do not need to worry if you didn’t grasp all the concepts covered in a few days. You may need to repeat some of the activities to better understand some of the content, especially in this final lesson. Remember, improving a model’s performance is not a straightforward task and generally involves an iterative process. To continue practicing, visit [Kaggle](https://www.kaggle.com/) where you can find a massive repository of community-published models, data, and code.

**Reflection and feedback:** At the end of each day or module, encourage students to reflect on what they've learned and provide an avenue for them to give feedback. Take the day’s learning objectives and rephrase them as reflection questions. Use the following questions as a starting point and update based on that day’s learnings.

Take a few minutes to reflect on your learning:

* What new topics did you learn in this module?
* How has your understanding changed or evolved?
* What are you wondering about?
* What questions do you have?

## References

Wang. X. 2021. *RandomizedSearchCV*. Medium. Available: [2023, October 11].

Ambati. V. 2021. *Deep Dive into Resampling*. Medium. Available: [2023, October 11].

—--

© 2023 edX Boot Camps LLC. Confidential and Proprietary. All Rights Reserved.
