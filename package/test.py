import pandas as pd

# Đường dẫn đến file Excel cũ
file_path = '2008 Contoso Data.xlsx'

# Số dòng bạn muốn tách
num_rows = 20

# Đọc file Excel
df = pd.read_excel(file_path)

# Lấy 5000 dòng đầu tiên
df_subset = df.head(num_rows)

# Tạo tên file mới
new_file_name = 'file_moi1.xlsx'

# Lưu 5000 dòng đầu tiên vào file mới
df_subset.to_excel(new_file_name, index=False)

print(f'Đã lưu: {new_file_name}')