def app():
    import streamlit as st
    import zipfile
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.impute import KNNImputer
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    uploaded_file = st.file_uploader('Upload file' , type = "csv")
    show_file = st.empty()
    
    if uploaded_file is not None:
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, low_memory=False)
        st.write(type(data))
        st.dataframe(data.head())

        column_names = tuple(data.columns)
        option = st.selectbox("SELECT THE TARGET VARIABLE" , column_names)
    
        st.write("You selected: " , option)
    
        target = data[str(option)]
        train = data.drop([str(option)] , axis = 1)
        
        if train.isnull().values.any():
            st.write("YOUR DATA CONTAINS NAN VALUES")
            imputer_option = st.selectbox("Select How to Fill the NaN values" , ("Simple Imputer" , "KNN Imputer"))
            if imputer_option == "Simple Imputer":
                imputer = SimpleImputer(strategy = "most_frequent")
                train = imputer.fit_transform(train)
                target = imputer.fit_transform(target)
            elif imputer_option == "KNN Imputer":
                imputer = SimpleImputer(strategy = "most_frequent")
                train = imputer.fit_transform(train)
                target = imputer.fit_transform(target)
        #target = target.values.reshape(-1,1)
        encoder_option = st.selectbox("Choose a way to encode your categorical datas" , ("One Hot Encoding" , "Label Encoding" , "No Encoding"))
        if encoder_option == "Label Encoding":
            le1 = LabelEncoder()
            le2 = LabelEncoder()
            train = train.apply(le1.fit_transform)
            st.dataframe(target)
            #target = le2.fit_transform(target)
            #st.dataframe(target)
        if encoder_option == "One Hot Encoding":
            pass
        if encoder_option == "No Encoding":
            st.write("You are good to go")
        
        scaling_option = st.selectbox("Choose a way to scale your data" , ("Standard Scaling" , "Min Max Scaler" , "No scaling"))
        if scaling_option == "Standard Scaling":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            train = sc.fit_transform(train)
            st.dataframe(train)
        if scaling_option == "Min Max Scaler":
            from sklearn.preprocessing import MinMaxScaler
            mms = MinMaxScaler()
            train = mms.fit_transform(train)
            st.dataframe(train)
        if scaling_option == "No scaling":
            st.write("You are good to go")
            st.dataframe(train)
            
        split_slider = st.slider("CHOOSE YOUR TEST SIZE" , 0.1 , 0.5 ,step = 0.01)        
        from sklearn.model_selection import train_test_split as split
        target = target.values.reshape(-1 , 1)
        X_train , X_test , y_train , y_test = split(train , target , test_size = split_slider)
        
        Model = st.selectbox("Choose your Regression model" , ("XGBoost" , "CatBoost" , "Linear Regression" , "ANN" , "Logistic Regression"))
        if Model == "XGBoost":
            st.write("YOU HAVE CHOSEN XGBOOST REGRESSION MODEL")
            import xgboost as xgb
            from sklearn.metrics import mean_squared_error
            
            st.markdown("Booster Parameters: Guide the individual booster (tree/regression) at each step")
        
            eta = st.text_input("Learning rate:", "Type Here", 1 , key='a') 
            
            min_child_weight = st.text_input("Petal Length", "Type Here",1 , key = 'b') 
            
            max_depth  = st.text_input("Max Depth", "Type Here",6 , key = 'c') 
            
            max_leaf_nodes = st.text_input("Max leaf Nodes", "Type Here",32 , key = 'd')
            
            gamma = st.text_input("Gamma", "Type Here",0 , key = 'e')
            
            subsample  = st.text_input("Sub Sample", "Type Here",1 , key = 'f')
            
            colsample_bytree= st.text_input("Column Sample by Tree", "Type Here",1 , key = 'g')
            
            colsample_bylevel  = st.text_input("Column Sample by Level", "Type Here",1 , key = 'h')
            
            Lambda = st.text_input("Lambda ", "Type Here",1 , key = 'i')
            
            alpha = st.text_input("MAx Delta Step", "Type Here",0 , key = 'j')
            
            scale_pos_weight  = st.text_input("MAx Delta Step", "Type Here",1 , key = 'k')
            
            
            st.markdown("Learning Task Parameters: Guide the optimization performed")
            
            
            
            #objective  = st.text_input("MAx Delta Step", "Type Here", "reg:linear")
            eval_metric  = st.text_input("Evaluation metrics","Type Here" , key= 'l')
            seed= st.text_input("Evaluation metrics","Type Here",0 , key = 'm')
            
            
            
            
            
            if st.button("Predict"):           
            
                xg_reg = xgb.XGBRegressor(colsample_bytree, learning_rate ,max_leaf_nodes,gamma,
                            max_depth , alpha, n_estimators,scale_pos_weight,lamba,eval_metric,seed,eta,min_child_weight,colsample_bylevel,colsample_bytree,)
            
                xg_reg.fit(X_train,y_train)
            
                preds = xg_reg.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                st.write(rmse)
            
            """
            def cv():
            params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                        'max_depth': 5, 'alpha': 10}
            cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                            num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
            cv_results.head()
            xgb.plot_importance(xg_reg)
            plt.rcParams['figure.figsize'] = [5, 5]
            plt.show()
                
            """
        
        
        
        