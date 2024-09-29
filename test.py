import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vnstock3 import *
from scipy.optimize import minimize

# Bước 1: Lấy dữ liệu lịch sử giá của các mã VN30
tickers = ['VIC', 'VHM', 'VNM', 'HPG', 'MSN']  # Thay thế bằng các mã VN30 khác
data = pd.DataFrame()

# Lấy dữ liệu lịch sử từng mã cổ phiếu từ vnstock
for ticker in tickers:
    stock_data = stock_historical_data(ticker, "2020-01-01", "2023-01-01")
    stock_data = stock_data[['date', 'close']]  # Chỉ lấy cột ngày và giá đóng cửa
    stock_data.columns = ['date', ticker]  # Đổi tên cột theo mã cổ phiếu
    stock_data.set_index('date', inplace=True)
    if data.empty:
        data = stock_data
    else:
        data = data.join(stock_data)

# Bước 2: Tính tỷ suất sinh lợi hàng ngày
returns = data.pct_change().dropna()

# Bước 3: Tính toán ma trận hiệp phương sai và trung bình tỷ suất sinh lợi
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Bước 4: Hàm tối ưu hóa danh mục đầu tư (rủi ro nhỏ nhất cho lợi nhuận nhất định)
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(weights * mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std_dev, returns

# Bước 5: Hàm tối ưu hóa cho rủi ro nhỏ nhất
def minimize_risk(weights, mean_returns, cov_matrix, target_return):
    std_dev, returns = portfolio_performance(weights, mean_returns, cov_matrix)
    return std_dev

# Bước 6: Xây dựng đường biên hiệu quả (Efficient Frontier)
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        
        std_dev, returns = portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0, i] = std_dev  # Rủi ro
        results[1, i] = returns  # Lợi nhuận kỳ vọng
        results[2, i] = returns / std_dev  # Tỷ lệ Sharpe
        weights_record.append(weights)
    
    return results, weights_record

# Tạo đường biên hiệu quả
num_portfolios = 10000
results, weights_record = efficient_frontier(mean_returns, cov_matrix, num_portfolios)

# Bước 7: Vẽ đường biên hiệu quả
plt.figure(figsize=(10, 7))
plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
plt.colorbar(label='Tỷ lệ Sharpe')
plt.title('Đường biên hiệu quả Markowitz - VN30')
plt.xlabel('Rủi ro (Độ lệch chuẩn)')
plt.ylabel('Lợi nhuận kỳ vọng')
plt.show()
