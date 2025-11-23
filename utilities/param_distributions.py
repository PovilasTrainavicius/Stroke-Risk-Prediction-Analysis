from scipy.stats import randint, uniform, loguniform


param_distributions = {
    "LogisticRegression": {
        "classifier__C": loguniform(1e-3, 1e2),              
        "classifier__tol": loguniform(1e-3, 1e-1),    
        "classifier__penalty": ["l1", "l2", "elasticnet"],
        "classifier__l1_ratio": uniform(0.0, 1.0),
        "classifier__max_iter": randint(1000, 4000),          
        "classifier__multi_class": ["auto", "ovr", "multinomial"],
        "classifier__solver": ["saga", "liblinear"]
},
    "XGBClassifier": {
        "classifier__gamma": uniform(0, 5),
        "classifier__max_depth": randint(3, 10),
        "classifier__grow_policy": ["depthwise", "lossguide"],
        "classifier__min_child_weight": randint(1, 8),
        "classifier__subsample": uniform(0.5, 0.5),
        "classifier__n_estimators": randint(300, 1500),
        "classifier__learning_rate": loguniform(1e-4, 1e-1),
        "classifier__colsample_bytree": uniform(0.5, 0.5),
        "classifier__reg_alpha": loguniform(1e-4, 1e-1),
        "classifier__reg_lambda": loguniform(1e-4, 1e-1),
    },
    "LGBMClassifier": {
        "classifier__boosting_type": ["gbdt", "dart"],
        "classifier__num_leaves": randint(20, 200),
        "classifier__learning_rate": loguniform(1e-4, 1e-1),
        "classifier__n_estimators": randint(200, 1000),
        "classifier__max_depth": randint(-1, 10),
        "classifier__feature_fraction": uniform(0.5, 0.4),
        "classifier__bagging_freq": randint(0, 5),
        "classifier__subsample": uniform(0.5, 0.4),
        "classifier__min_child_weight": loguniform(1e-4, 1e-1),
        "classifier__min_child_samples": randint(1, 50),
        "classifier__reg_alpha": loguniform(1e-4, 1e-1),
        "classifier__reg_lambda": loguniform(1e-4, 1e-1),
    },
    "CatBoostClassifier": {
        "classifier__depth": randint(3, 10),
        "classifier__learning_rate": loguniform(1e-4, 1e-1),
        "classifier__l2_leaf_reg": loguniform(1e-4, 1e-1),
        "classifier__n_estimators": randint(200, 1000),
        "classifier__bootstrap_type": ["Bernoulli", "MVS"],
        "classifier__subsample": uniform(0.5, 0.4),
    }
}