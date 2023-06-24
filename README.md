# Random forest image classification
 
Testing Model Accuracy: 79.4%

- Activating virtual env: `Scripts\activate`

- Deactivating virtual env: `Scripts\deactivate`

- Installing dependencies: `pip install -r requirements.txt`
  
- To run the program: `python main.py`

- Copy the dataset in root directory in the following format:

&emsp;dataset/ <br/>
&emsp;&emsp;iqbal/ <br/>
&emsp;&emsp;jalal/ <br/>
&emsp;&emsp;kaptaan/ <br/>
&emsp;&emsp;pahari/ <br/>

Note: images should be of .jpg extension

- To test a single image, copy it in root directory with name `image.jpg`
- Delete `rf_model.pkl` to retrain the model
- Delete `processed_data` to reprocess the dataset
