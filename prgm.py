import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#File path
IRIS_FILE_PATH = r"D:/Python Projects/ai/IRIS_FLOWER_CLASSIFICATION/Dataset/IRIS.csv"

#load the dataset and store the x and y values
df = pd. read_csv(IRIS_FILE_PATH)
X =  df.drop("species",axis=1)
_,y = np.unique(df["species"],return_inverse=True)

#visualise the features and save them
pair_plot = sns.pairplot(df,hue="species",palette="Set1")
plt.suptitle("Feature Relationships by Species", y=1.02)
plt.savefig(r"D:/Python Projects/ai/IRIS_FLOWER_CLASSIFICATION/Visualizations/pairplot_species_relationships.png",dpi=500,bbox_inches = "tight")
plt.close()

#split the dataset for the purpose of training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

#Scale the dataset to a value between 0 and 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Use the knn model
model_knn = KNeighborsClassifier(n_neighbors=4)
model_knn.fit(X_train,y_train)

#make prediction 
y_pred = model_knn.predict(X_test)

#evaluating the model
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy of the model is {accuracy*100}%")
conf_matrix = confusion_matrix(y_test,y_pred)
print("Confusion matrix \n",conf_matrix)
class_report  = classification_report(y_test,y_pred,target_names=df["species"].unique())
print("Classification Report:\n", class_report)

#plot the results
conf_matrix_graph = sns.heatmap(conf_matrix,annot=True,fmt="d",xticklabels=df["species"].unique(),yticklabels=df["species"].unique())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(r"D:/Python Projects/ai/IRIS_FLOWER_CLASSIFICATION/Visualizations/Confusion_Matrix_Graph_IRIS.png",dpi=500,bbox_inches="tight")
plt.show()