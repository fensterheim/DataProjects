from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
class NanFiller():
    def __init__(self, df):
        self.classifier = XGBClassifier(silent=False, n_estimators=70, gamma=0, subsample=1)
        self.regressor = XGBRegressor(silent=False, n_estimators=70, gamma=0, subsample=1)
        self.dataframe = df
        self.dataframe.set_index('Id', inplace=True)
    
    def split_train_test_by_nan(self, feature, df):
        '''
        Used to split the df to two seperate dataframes, 
        one containing nans in the feature and the other NOT containing nans in  the feature.
        '''
        test_new = df[df[feature].isnull()]
        train_new = df[~df[feature].isnull()]
        return  train_new, test_new
    
    def cv(self, X_train, y_train, task):
        '''
        Root mean squared error of crossval
        '''
        print('about to perform cv')
        print(f'X_train shape: {X_train.shape}')
        print(f'y_train shape: {y_train.shape}')
        X_train = pd.get_dummies(X_train)
        if task == 'c':
            rmse = cross_val_score(self.classifier, X_train, y_train, cv = 5)
        elif task == 'r':
            rmse = np.sqrt(-cross_val_score(self.regressor, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
        return(rmse)
    
    def copy_missing_columns(self, columnsa, columnsb, dfa, dfb):
        '''
        Compares the columns from both df and fills in the missing in each by adding a series filled with zeros.
        This is good for dataframe that contain dummy variables.
        '''
        col_iter = list(columnsb)
        for col in col_iter:
            if col in columnsa:
                columnsb.remove(col)
                columnsa.remove(col)
            else:
                col_series = dfb[col]
                col_dtype = col_series.dtype
                col_name = col_series.name
                dfa[col] = (pd.Series(np.zeros((dfa.shape[0],)).astype(col_dtype), dtype=col_dtype, name=col_name))
    
    def sync_columns(self, df1_, df2_):
        '''
        Syncronises the columns from both dataframes so that 
        '''
        columns1 = df1_.columns.tolist()
        columns2 = df2_.columns.tolist()
        #col1_iter = list(columns1)
        df1 = df1_
        df2 = df2_
        self.copy_missing_columns(columns2, columns1, df2, df1)
        self.copy_missing_columns(columns1, columns2, df1,  df2)
        columns1 = df1.columns.tolist()
        columns2 = df2.columns.tolist()
        df1 = df1.reindex(sorted(df1.columns), axis=1)
        df2 = df2.reindex(sorted(df2.columns), axis=1)
        return df1, df2
                
    
    def predict_nans(self, train_new, to_predict, feature):
        '''
        
        '''
        print(f'train_new shape: {train_new.shape}')
        print(f'to_predict shape: {to_predict.shape}')
        y = train_new[feature]
        train_new.drop([feature], axis=1, inplace=True)
        to_predict.drop([feature], axis=1, inplace=True)
        if self.dataframe[feature].dtype == 'object': #classification task
            task = 'c'
        else:
            task = 'r'
        cv_score = self.cv(train_new, y, task).mean()
        print(f'task: {task}')
        print(f'feature: {feature}')
        print(f'cv_score: {cv_score}')
        
        X_train = pd.get_dummies(train_new)
        X_test = pd.get_dummies(to_predict)
        X_train, X_test = self.sync_columns(X_train, X_test)
        if task=='c':
            model = self.classifier
        elif task == 'r':
            model = self.regressor
        else:
            return False
        model.fit(X_train, y)
        predicted = model.predict(X_test)
        print(f'predicted: {predicted.tolist()}')
        return predicted, y
    
    #def split_train_test_by_nan(self, feature, dataframe):
    #    '''
    #    will split the dataframe to train and test according to the feature
    #    train will have no nans in the feature column and test will have nans
    #    '''
    #    train_feature = dataframe.dropna([feature])
    #    test_feature = pd.concat([dataframe, train_feature]).drop_duplicates(keep=False)
    #    return train_feature, test_feature
    
    def remove_test_set_with_specific_nans(self, test_feature, feature):
        '''
        draws a set to_predict from  the test_feature
        drops the set from the test_feature
        the set has specific columns in all the rows that are nans
        returns the list of features which were dropped + the set that was created
        '''
        first_row = test_feature[0:1]
        print(f'first_row: {first_row}')
        first_row.drop([feature], axis=1, inplace=True)
        features_to_drop = first_row.columns[first_row.isna().any()].tolist()
        print(f'features_to_drop: {features_to_drop}')
        test_feature_before_drop = test_feature.copy()
        test_no_feature = test_feature.drop([feature], axis=1)
        print(f'test_feature_before_drop shape: {test_feature_before_drop.shape}')
        if len(features_to_drop) == 0:
            to_predict = test_feature[~test_feature.drop(feature, axis=1).isnull().any(1)]
        else:
            to_predict = test_feature[test_no_feature[features_to_drop].isnull().all(1)&~test_no_feature.drop(features_to_drop, axis=1).isnull().any(1)]
        #to_predict = test_feature[test_feature[features_to_drop].isnull().all(1)&~test_feature.drop(features_to_drop, axis=1).isnull().any(1)]
        print(f'test_feature after drop: {to_predict.shape}')
        test_feature = pd.concat([test_feature, to_predict]).drop_duplicates(keep=False)
        print(f'to_predict shape: {to_predict.shape}')
        return test_feature, features_to_drop, to_predict
    
    def fill_nans(self, nanful_features):
        '''
        :param nanful_feature: a list of feature names which contain nans
        :param dataframe: the dataframe
        '''
        print(self.dataframe.index)
        print(list(nanful_features))
        for feature in nanful_features:
            print('\n---------------------------------------------------------------------------')
            print(f'feature: ' + feature)
            print(f'type: {type(feature)}')
            other_nanful_features = list(nanful_features)
            other_nanful_features.remove(feature)
            
            #prepair
            train_feature, test_feature = self.split_train_test_by_nan(feature, self.dataframe)
                
                
            while not test_feature.empty:
                
                #1
                #removes a set with specific nans from the dropped_samples and decreases the size of dropped_samples
                test_feature, features_to_drop, to_predict = self.remove_test_set_with_specific_nans(test_feature, feature)
                nanful_features_without_dropped_cols = list(other_nanful_features)
                [nanful_features_without_dropped_cols.remove(x) for x in features_to_drop if x in nanful_features_without_dropped_cols]
                
                #2
                #remove from train rows with nans in other columns
                train_with_relevant_rows = train_feature.dropna(subset=nanful_features_without_dropped_cols)
                
                #3
                #remove cols from all relevant
                to_predict_with_rellevant_columns = to_predict.drop(features_to_drop, axis=1)
                train_new_with_rellevant_columns = train_with_relevant_rows.drop(features_to_drop, axis=1)
                
                #4
                #predict
                predicted, y = self.predict_nans(train_new_with_rellevant_columns, to_predict_with_rellevant_columns, feature)
                to_predict[feature] = predicted
                to_predict[feature].astype(y.dtype)
                to_predict[feature].rename(y.name)
                                
                #5
                #include to_predict which has the features filled out in the train_new
                train_feature = pd.concat([train_feature,to_predict]).drop_duplicates()
                
                #6
                #replace old rows with the new ones after prediction
                self.dataframe.loc[to_predict.index] = to_predict
                
        return self.dataframe
            
Id = pd.concat([train['Id'], test['Id']])
combine_df['Id'] = Id.values
nan_filler = NanFiller(combine_df)
nanless_df = nan_filler.fill_nans(nan_cols_descending)
