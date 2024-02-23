import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

class DataReader:
    @staticmethod
    def read_csv(file_path):
        return pd.read_csv(file_path)

class LinearRegression:
    @staticmethod
    def fit(x, y):
        x = np.column_stack((np.ones(len(y)), x))
        theta = np.linalg.inv(x.T @ x) @ x.T @ y
        return theta

class KNNRegressor:
    @staticmethod
    def fit(x, y):
        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(x, y)
        return knn

class ModelEvaluator:
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return np.corrcoef(y_true, y_pred)[0, 1]

class DataPlotter:
    @staticmethod
    def plot_regression(id_feature, target_feature, predicted_values, title, save_path):
        plt.figure(figsize=(8, 6))
        plt.scatter(id_feature, target_feature, label='Real Data')
        plt.plot(id_feature, predicted_values, color='red', label='Predicted Values')
        plt.xlabel('Product ID')
        plt.ylabel(title)
        plt.title(f'Regression Plot for {title}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(save_path, dpi=300)

def main():
    file_path = 'assets/sales.csv'
    df = DataReader.read_csv(file_path)

    product_id = df['product_id']
    company_size = df['company_size']
    segment_id = df['segment_id']
    average_sales_time = df['average_sales_time']
    sales_value = df['sales_value']
    salesperson_id = df['salesperson_id']  

    theta_average_sales_time = LinearRegression.fit(np.column_stack((product_id, company_size)), average_sales_time)
    theta_sales_value = LinearRegression.fit(np.column_stack((company_size, segment_id)), sales_value)
    knn_salesperson = KNNRegressor.fit(np.column_stack((product_id, company_size, segment_id)), salesperson_id) 

    predicted_average_sales_time = np.column_stack((np.ones(len(average_sales_time)), product_id, company_size)) @ theta_average_sales_time
    predicted_sales_value = np.column_stack((np.ones(len(sales_value)), company_size, segment_id)) @ theta_sales_value
    predicted_salesperson = knn_salesperson.predict(np.column_stack((2, 3, 1)))  

    DataPlotter.plot_regression(product_id, average_sales_time, predicted_average_sales_time, 'Average Sales Time', 'regression_tmv.jpg')
    DataPlotter.plot_regression(product_id, sales_value, predicted_sales_value, 'Projected Sales Value', 'regression_projected_sales_value.jpg')

    print("The best salesperson is: " + str(predicted_salesperson))

if __name__ == "__main__":
    main()
