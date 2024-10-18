# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Import necessary libraries, such as `pandas` for data manipulation and `matplotlib` for plotting.

2. **Load the Dataset**: Load the dataset `Mall_Customers.csv` into a DataFrame using `pandas` and check the first few rows and information about the dataset.

3. **Check for Missing Values**: Use `isnull().sum()` to check if there are any missing values in the dataset.

4. **Initialize WCSS List**: Create an empty list `wcss` to store the Within-Cluster Sum of Squares (WCSS) values for different numbers of clusters.

5. **Apply K-Means for Different Clusters**: Loop through cluster numbers from 1 to 10, apply K-Means clustering for each value of `n_clusters`, and calculate WCSS using `kmeans.inertia_`. Append the WCSS values to the `wcss` list.

6. **Plot the Elbow Method**: Plot the WCSS values against the number of clusters to use the Elbow Method for identifying the optimal number of clusters.

7. **Fit K-Means with 5 Clusters**: Apply K-Means clustering with `n_clusters=5` and fit the model to the selected features of the dataset (`Annual Income` and `Spending Score`).

8. **Predict Cluster Labels**: Use the trained K-Means model to predict the cluster labels for each data point in the dataset.

9. **Assign Cluster Labels to Data**: Create a new column `cluster` in the dataset and store the predicted cluster labels for each customer.

10. **Visualize the Clusters**: Use `matplotlib` to plot the customers based on their annual income and spending score, assigning different colors to each cluster and displaying the plot with legends and titles.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Pandidharan.G.R
RegisterNumber:  212222040111
*/
```
```c
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
  kmeans = KMeans (n_clusters = i, init ="k-means++")
  kmeans.fit(data.iloc[:,3:])
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("no of cluster")
plt.ylabel("wcss")
plt.title("Elbow Metthod")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]

plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="pink",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="green",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="black",label="cluster4")
plt.legend()
plt.title("Customer Segments")

```

## Output:
## Dataset:

![278795578-afbcb3f0-41a0-4e9e-bafd-a0fafb3334c0](https://github.com/charumathiramesh/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120204455/ed5b0fe2-4e98-43bd-9b21-f3468d493335)

## Dataset information:
![278795602-253e3aeb-6850-4bfd-b39b-9e82bf90d834](https://github.com/charumathiramesh/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120204455/8d4aa9d8-45dd-4539-8271-f7da5d00dc99)
![278795605-b654a77d-774e-4cc0-9396-4353a521f7cc](https://github.com/charumathiramesh/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120204455/f3c984b3-7e9b-4852-9552-a4026fbfd52d)




## Elbow method graph (wcss vs each iteration):
![278795631-c7b3a69a-089c-4ff3-b253-9a2a0b6e0daa](https://github.com/charumathiramesh/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120204455/65a43453-3ac2-4bea-8187-97cb6cc060d1)


## Cluster represnting customer segments-graph:

![278795641-c898bcee-13f7-49d2-918c-dc151a9702f3](https://github.com/charumathiramesh/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120204455/5796e66a-5977-4d88-857d-822ee784388d)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
