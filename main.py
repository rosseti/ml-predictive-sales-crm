import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

class ModelEvaluator:
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return np.corrcoef(y_true, y_pred)[0, 1]

class DataPlotter:
    @staticmethod
    def plot_regression(id_feature, target_feature, predicted_values, accuracy, title, save_path):
        plt.figure(figsize=(8, 6))
        plt.scatter(id_feature, target_feature, label='Dados Reais')
        plt.plot(id_feature, predicted_values, color='red', label=f'Regressão Linear (Acurácia: {accuracy:.2f})')
        plt.xlabel('ID Produto')
        plt.ylabel(title)
        plt.title(f'Regressão Linear para {title}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(save_path, dpi=300)

def main():
    file_path = 'assets/sales.csv'
    df = DataReader.read_csv(file_path)

    id_produto = df['id_produto']
    id_porte = df['id_porte']
    id_segmento = df['id_segmento']
    tempo_medio_venda = df['tempo_medio_venda']
    valor_venda = df['valor_venda']

    # Ajustar modelos
    theta_tempo_venda = LinearRegression.fit(np.column_stack((id_produto, id_porte)), tempo_medio_venda)
    theta_valor_venda = LinearRegression.fit(np.column_stack((id_porte, id_segmento)), valor_venda)

    # Prever valores
    tempo_venda_predito = np.column_stack((np.ones(len(tempo_medio_venda)), id_produto, id_porte)) @ theta_tempo_venda
    valor_venda_predito = np.column_stack((np.ones(len(valor_venda)), id_porte, id_segmento)) @ theta_valor_venda

    # Calcular acurácia
    acuracia_tempo_venda = ModelEvaluator.calculate_accuracy(tempo_medio_venda, tempo_venda_predito)
    acuracia_valor_venda = ModelEvaluator.calculate_accuracy(valor_venda, valor_venda_predito)

    # Plotar resultados
    DataPlotter.plot_regression(id_produto, tempo_medio_venda, tempo_venda_predito, acuracia_tempo_venda, 'Tempo Médio de Venda', 'regressao_tmv.jpg')
    DataPlotter.plot_regression(id_produto, valor_venda, valor_venda_predito, acuracia_valor_venda, 'Valor Previsto de Venda', 'regressao_valor_previsto_venda.jpg')

if __name__ == "__main__":
    main()
