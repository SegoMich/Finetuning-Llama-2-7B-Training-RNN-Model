 #PREDICTIVE MODEL
 import pandas as pd
 import numpy as np
 from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
 from sklearn.model_selection import train_test_split
 from keras.models import Sequential
 from keras.layers import Dense, GRU
 from keras.optimizers import Adam
 from sklearn.cluster import KMeans
 
 # Load lead data
 leads_df = pd.read_csv('lead.csv')
 
 # Preprocess industry and lead status
 label_encoder_industry = LabelEncoder()
 leads_df['industry_encoded'] = label_encoder_industry.fit_transform(leads_df['industry'])
 
 label_encoder_status = LabelEncoder()
 leads_df['lead_status_encoded'] = label_encoder_status.fit_transform(leads_df['lead_status'])
 
 # Features
 X = leads_df[['industry_encoded', 'lead_status_encoded']].values
 
 # Normalize features
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)
 
 # Perform KMeans clustering
 k = 5  # Assuming 5 clusters
 kmeans = KMeans(n_clusters=k, random_state=42)
 leads_df['lead_cluster'] = kmeans.fit_predict(X_scaled)
 
 # Scale the cluster assignments to the range [0, 10]
 scaler_cluster = MinMaxScaler(feature_range=(0, 10))
 leads_df['lead_score'] = scaler_cluster.fit_transform(leads_df['lead_cluster'].values.reshape(-1, 1))
 
 # Features and target variable for the RNN model
 X_rnn = leads_df[['industry_encoded', 'lead_status_encoded']].values
 y_rnn = leads_df['lead_score'].values.reshape(-1, 1)
 
 # Normalize features for the RNN model
 scaler_rnn = StandardScaler()
 X_scaled_rnn = scaler_rnn.fit_transform(X_rnn)
 
 # Reshape input data for the RNN model
 X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_scaled_rnn, y_rnn, test_size=0.2, random_state=42)
 X_train_rnn = X_train_rnn.reshape(X_train_rnn.shape[0], 1, X_train_rnn.shape[1])
 X_test_rnn = X_test_rnn.reshape(X_test_rnn.shape[0], 1, X_test_rnn.shape[1])
 
 # Define RNN model
 model_rnn = Sequential()
 model_rnn.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
 model_rnn.add(Dense(units=1, activation='linear'))
 
 # Compile model
 model_rnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
 
 # Train model
 model_rnn.fit(X_train_rnn, y_train_rnn, epochs=20, batch_size=32, validation_data=(X_test_rnn, y_test_rnn), verbose=2)
 
 # Predict lead scores
 y_pred_rnn = model_rnn.predict(X_test_rnn)
 
 # Scale predicted lead scores to the range [0, 10]
 scaler_lead_score = MinMaxScaler(feature_range=(0, 10))
 y_pred_scaled = scaler_lead_score.fit_transform(y_pred_rnn)
 
 # Evaluate model performance
 mse, mae = model_rnn.evaluate(X_test_rnn, y_test_rnn, verbose=0)
 print(f"Mean Squared Error: {mse}")
 print(f"Mean Absolute Error: {mae}")
 
 # Example prediction for a new lead
 def score(industry, status):
 
     new_lead_features = [[label_encoder_industry.transform([industry])[0],
                           label_encoder_status.transform([status.replace(industry,"")[1:]])[0]]]
     new_lead_features_scaled = scaler_rnn.transform(new_lead_features)
     new_lead_features_scaled_reshaped = new_lead_features_scaled.reshape(1, 1, 2)
     predicted_lead_score = model_rnn.predict(new_lead_features_scaled_reshaped)
     predicted_lead_score_scaled = scaler_lead_score.transform(predicted_lead_score)[0][0]
 
     print(f"Predicted lead score for the new lead: {predicted_lead_score_scaled}")
     return str(round(predicted_lead_score_scaled,1))
 
 
 
 
 
 
 
 #GENERATIVE MODEL
 
 
 from flask import Flask, request, jsonify
 import random
 
 from langchain.llms import CTransformers
 from langchain import PromptTemplate, LLMChain
 from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
 
 llm = CTransformers(model="model", model_file = 'llama-7b-v1.0.gguf', callbacks=[StreamingStdOutCallbackHandler()])
 
 template = """
 [INST] <<SYS>>
 You are a helpful, respectful and honest assistant for A&B consulting company. Your answers are always brief.stop at [/INST].
 <</SYS>>
 {text}[/INST]
 """
 
 prompt = PromptTemplate(template=template, input_variables=["text"])
 
 llm_chain = LLMChain(prompt=prompt, llm=llm)
 
 #control model
 import pandas as pd
 import re
 
 
 # Create DataFrame
 
 def analyze_text(text):
     # Define patterns for top, lead, least, and number
     top_pattern = r'top\s*(\w+)'
     lead_pattern = r'lead*(\w+)'
     least_pattern = r'least\s*(\w+)'
     number_pattern = r'\b\d+\b'
 
     # Find matches in the text
     top_match = re.search(top_pattern, text, re.IGNORECASE)
     lead_match = re.search(lead_pattern, text, re.IGNORECASE)
     least_match = re.search(least_pattern, text, re.IGNORECASE)
     number_match = re.search(number_pattern, text)
 
     # Extract the matched values
     top_word = top_match.group(1) if top_match else None
     lead_word = lead_match.group(1) if lead_match else None
     least_word = least_match.group(1) if least_match else None
     number = int(number_match.group()) if number_match else None
 
     return top_word, lead_word, least_word, number
 
 def perform_actions(top_word, lead_word, least_word, number):
      if lead_word:
         lead = ""
         if top_word:
             highest = 0
             if number:
                 highest = leads_df['lead_score'].nlargest(number).index
                 lead+="\n"+str(number) + (" top leads\n")
 
             else:
                 highest = leads_df['lead_score'].nlargest(1).index
                 lead+="\nTop lead(s)"
             lead+= str(leads_df.loc[highest])
         if least_word:
             lowest = 0
             if number:
                 lowest = leads_df['lead_score'].nsmallest(number).index
                 lead+="\n"+str(number)+" bottom leads\n"
             else:
                 lowest = leads_df['lead_score'].nsmallest(1).index
                 lead+="\nLeast lead(s)\n"
             lead+= str(leads_df[['first_name','last_name','company','lead_status','industry']].loc[lowest])
         if len(lead) == 0:return 0
         else:return lead
      else:
          return 0
 
 
 #here
 
 
 
 app = Flask(__name__)
 
 def receive_data_predictive():
     data = request.json
     print("Received data from Java:", data)
 
     # Process the received data in Python
 
     text = data['text']
     top_word, lead_word, least_word, number = analyze_text(text)
 
     response_data = perform_actions(top_word, lead_word, least_word, number)
     if response_data == 0:
         response_data = llm_chain.invoke(data)
 
     return jsonify(response_data)
 
 
 
 @app.route('/api/data', methods=['POST'])
 def receive_and_respond():
     return receive_data_predictive()
 
 if __name__ == '__main__':
     app.run(port=5000)
