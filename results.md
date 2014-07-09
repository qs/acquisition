### SGDClassifier
 * auc:  0.672532394817
 * precision:  0.390243902439
 * recall:  0.421052631579

### KNeighborsClassifier
 * auc:  0.637398016317
 * precision:  0.954545454545
 * recall:  0.276315789474

### SVC
 * auc:  0.886878099504
 * precision:  0.7625
 * recall:  0.802631578947

### DecisionTreeClassifier
 * auc:  0.899536074228
 * precision:  0.849315068493
 * recall:  0.815789473684

### AdaBoostClassifier - base_estimator=DecisionTreeClassifier()
 * auc:  0.92889137738
 * precision:  0.904109589041
 * recall:  0.868421052632

### AdaBoostClassifier - base_estimator=SVC(kernel="linear", C=0.025)
 * auc:  0.887637977924
 * precision:  0.772151898734
 * recall:  0.802631578947

### GradientBoostingClassifier - base_estimator=DecisionTreeClassifier()
 * auc:  0.914973604223
 * precision:  0.888888888889
 * recall:  0.842105263158

### GaussianNB
 * auc:  0.927111662134
 * precision:  0.788235294118
 * recall:  0.881578947368
