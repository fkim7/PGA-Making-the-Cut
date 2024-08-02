# PGA-Making-the-Cut
Jupyter Notebook- ML/AI Statistical analysis of each PGA tournament from 2018 to 2021, comprising of 14,000 instances and 64 features to predict best F1 scores using various classification algorithms as well as Deep Neural Networks.

`made_cut` was chosen as target feature because:
- Made_cut is a binary feature
- 58% of this feature belongs to the Positive class (class of interest)
- Saves from complexities of imbalanced datasets

Feature selection was done based on business knowledge and then compared to feature selection by classifiers.

Business knowledge wise, focus was in on the newest golf statistic - strokes gained.
- sg_ott is strokes gained off the tee
- sg_app is strokes gained approach
- sg_arg is strokes gained around the green
- sg_putt is strokes gained putting
- sg_t2g is strokes gained tee to green and is equal to sg_ott + sg_app + sg_arg
- sg_total = sg_t2g + sg_putt

This resulted in SGD Classifier - Accuracy 0.807, F1 0.818
Support Vector Classification - Accuracy 0.791, F1 0.818 with Sigmoid, Accuracy 0.822 and F1 0.837 with RBF
3 Layer Neural Net - Accuracy 0.829, F1 0.837

Feature selection using classifiers was deployed through ExtraTreesClassifier and RandomForestClassifier ensemble methods.
ex
```
Feature	Random Forest Importance	Extra Trees Importance
sg_total	0.2583497382	0.2039989467
sg_t2g	0.1129757101	0.1110418078
sg_putt	0.1060760718	0.1047393172
sg_app	0.07141883916	0.07227098882
sg_ott	0.05181390465	0.04972135997
sg_arg	0.0509874355	0.05081426643
Drive Yards	0.0257559248	0.03275789749
Fairways Hit	0.0249032326	0.03147306954
Age	0.02145152182	0.0315064301
PUTTS/HOLE	0.02106567969	0.0335337726
Weight lbs	0.01851451048	0.02940825806
Height cm	0.01521322929	0.02857584692
Consecutive_Cuts_Made	0.01387400245	0.02804792567
Length	0.0124851145	0.007517469257
winddirDegree	0.01153848545	0.006724490312
moon_illumination	0.01062470368	0.007565999154
Slope	0.01015089579	0.008316386746
humidity	0.009924848973	0.006528902359
cloudcover	0.009894277397	0.006750037825
pressure	0.009601711688	0.006581443677
```
**Algorithms New scores Previous Scores**
Random Forest Accuracy 0.866, F1 Score 0.879 NA
SVM Accuracy 0.856, F1 Score 0.872 Accuracy 0.791, F1 0.818
MLP CLassifier Accuracy 0.86, F1 Score 0.88 NA
Deep Neural Network Accuracy 0.85, F1 Score 0.87 Accuracy 0.829, F1 0.837

**Takeaways**
Always interesting to see how the model performs after employing feature selection + business intuition
NN may not always be the best answer, if your data is linear, linear regression is your best friend
Exploratory analysis should guide you if your data has a more of a linear relationship or a more complex relationship
Remember to encode data as per need. Is your data ordinal, nominal, or just numerical? 
