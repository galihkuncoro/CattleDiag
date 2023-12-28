# ML-Bangkit-Capstone-Project-CattleDiag-CH2-PR603


## Title of Capsone Project 
CattleDiag

## Abstract
"CattleDiag" is a groundbreaking Android mobile application designed to revolutionize the way local farmers diagnose and treat their cattle. Aimed at enhancing the health management of livestock, this innovative app serves as a comprehensive tool for farmers engaged in raising cattle. The primary goal of "CattleDiag" is to empower farmers with a user-friendly interface that facilitates the quick and accurate diagnosis of common cattle health issues.
The application features an intuitive symptom checker, leveraging advanced algorithms to identify potential illnesses based on observed physical symptoms. "CattleDiag" goes beyond mere diagnosis by providing detailed information on treatment options, preventive measures, and nutritional advice tailored to the specific needs of individual cattle. This holistic approach enables farmers, even those with limited veterinary knowledge, to make informed decisions for the well-being of their livestock.
Furthermore, "CattleDiag" incorporates a user-friendly database to track the health history of each cattle, facilitating long-term monitoring and proactive disease prevention. With real-time updates on best farming practices, disease outbreaks, and treatment innovations, the app ensures that local farmers stay abreast of the latest advancements in cattle healthcare.
"CattleDiag" strives to bridge the gap between traditional farming practices and modern technology, fostering a sustainable and thriving environment for both farmers and their cattle. This application stands as a testament to the transformative potential of mobile technology in supporting the livelihoods of local farmers and contributing to the overall health and productivity of their cattle.

## Author
Created and developed by the Machine Learning Cohort of Google Bangkit 2023, with the following profiles:
Bangkit ID  | Full Name                       |                
----------- | ------------------------------- |               
M248BSY1167 | Galih Kuncoro Jati  | 



## Related Link to Our Project
1. [Project Brief](https://docs.google.com/document/d/1PMCEKqtCIhXkZkfvbOby7mso87_mXo0LEJmka7PsVAE/edit?usp=sharing)


# Overall Development Flow 
1. [Data_Understanding](#Data_Understanding)  
2. [Data_Preparation](#Data_Preparation)    
3. [Modeling](#Modeling)  

# Further Explanation
## Data_Understanding
Dataset CattleDiag merupakan kumpulan data untuk memberikan sumber untuk membuat sistem terkait perawatan kesehatan hewan ternak. Ada kolom yang berisi penyakit, gejalanya, pengobatan, tindakan pencegahan, dan bobotnya.
Nama File Dataset   | Deskripsi               |  Jumlah Data                      |              
-----------         | ------------------------|   ------------------------------- |             
 dataset.csv        | Berisi data penyakit dan gejalanya | 4467                       |
 Symptom-severity.csv        |  Berisi data gejala dan bobot/tingkat keparahannya | 148 |
 Symptom_precaution.csv      |  Berisi data penyakit dan pencegahannya             | 33 |


## Data_Preparation
### Libraries Used
- Pandas: For data manipulation and analysis in DataFrame format.
- NumPy: For scientific computation, especially in array manipulation and mathematical operations on arrays.
- Scikit-learn: For machine learning tasks like building models, data splitting, and model evaluation.
- Matplotlib: For creating visualizations such as plots and graphs.
- Seaborn: For more visually appealing data visualizations.
- Joblib: For saving (serializing) models and objects into files for reuse.
- TensorFlow and Keras: For building and training Neural Network models.


### Handling Null / missing values
Filling Null or missing values with the number 0 can be achieved using the fillna(value=0) function. However, the dataframe needs to be reshaped beforehand.

    ```
    df = pd.DataFrame(s, columns = df.columns)
    df = df.fillna(value=0)
    ``` 

### Filling symptom data with their respective weights
Filling symptoms with their corresponding severity weights can be accomplished by assigning each symptom its respective severity value.

    ```
    vals = df.values
    symptoms = weight['Gejala'].unique()
    for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = weight[weight['Gejala'] == symptoms[i]]['Bobot'].values[0]
    ``` 


### Dividing the dataset into evaluation data, training data, and testing data.
The dataset is then split into training data and testing data, with a composition of 85% for training and 15% for testing.

    ```
    x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.85)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    ``` 
### Converting class vectors (integer numbers) to binary class matrices.
The labels of the training and testing datasets need to be converted from categorical to integer values. Subsequently, these integer labels are transformed into binary class matrices to align with the Neural Network's output format.

    ```
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    
    encoder =  LabelEncoder()
    y_train_e = encoder.fit_transform(y_train)
    y_train_c = to_categorical(y_train_e, num_classes = 41)
    y_test_e = encoder.fit_transform(y_test)
    y_test_c = to_categorical(y_test_e, num_classes = 41)
    
    joblib.dump(encoder, "/content/drive/MyDrive/CattleDiag/encoder.pkl")
    ``` 


## Modeling
### Support Vector Machine,
- SVC F1-score% = 97.39
- SVC Accuracy% = 97.91

### Neural Networks.
- Neural Network F1-scores% = 93.54
- Neural Network Accuracy% = 96.42

![Model_Loss](https://github.com/galihkuncoro/CattleDiag/blob/main/Model/Perubahan%20Loss%20pada%20tiap%20Epoch.png)  
![Model_Accuracy](https://github.com/galihkuncoro/CattleDiag/blob/main/Model/Perubahan%20Akurasi%20pada%20tiap%20Epoch.png)   
