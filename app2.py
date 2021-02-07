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
            target = le2.fit_transform(target)
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
        
        
        
        
        