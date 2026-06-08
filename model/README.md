# Folder Penyimpanan Model (.pkl)

Letakkan semua file model yang telah Anda ekspor dari Google Colab menggunakan `joblib` di folder ini.

### Contoh file yang disarankan:
- `svm_model.pkl`
- `knn_model.pkl`
- `decision_tree_model.pkl`
- `neural_network_model.pkl`

### Cara Memuat Model di Python:
```python
import joblib

model = joblib.load('models/nama_file_model.pkl')
```
