skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44)

 for train_index, test_index in skf.split(x_train, y_train):
        x0, x1 = x_train.iloc[train_index], x_train.iloc[test_index]
        y0, y1 = y_train.iloc[train_index], y_train.iloc[test_index] 
        clf.fit(x0, y0, eval_set=[(x1, y1)],
               eval_metric='rmse', verbose=False,early_stopping_rounds=10)                
        prval = clf.predict(x1)
        # calculate the error here
