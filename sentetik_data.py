#Sentetik Spektrum 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# Bu fonksiyon sentetik spektrum oluşturur
def create_synthetic_raman_spectrum(x, peaks_params, baseline_params, noise_level=0.2):

    # Temel çizgiyi oluştur (polinom + eksponansiyel bileşen)
    coeffs = np.polyfit(x, np.random.randn(len(x)) * baseline_params[1], baseline_params[0])
    baseline = np.polyval(coeffs, x) + baseline_params[1] * np.exp(-x/1000)
    
    # Piklerler
    spectrum = np.zeros_like(x)
    for center, height, width in peaks_params:
        spectrum += height * (width**2) / ((x - center)**2 + width**2)
    
    # Gürültü ekleyen kısım
    noise = np.random.normal(0, noise_level, len(x)) + \
            np.random.poisson(0.1, len(x)) * 0.05
    
    return baseline + spectrum + noise, spectrum, baseline

# Parametreler
x = np.linspace(500, 3500, 2000)  # Raman shift (cm-1)
peaks = [(800, 50, 30), (1200, 80, 40), (1600, 120, 50), 
         (2200, 70, 60), (2800, 90, 40)]
baseline = (3, 50)  # 3. derece polinom, 50 genlik

# Sentetik spektrum oluştur
raw_spectrum, true_spectrum, true_baseline = create_synthetic_raman_spectrum(
    x, peaks, baseline, noise_level=0.3)

# Görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(x, raw_spectrum, label='Ham Spektrum (Gürültülü)')
plt.plot(x, true_spectrum + true_baseline, 'r--', label='Gerçek Sinyal + Temel Çizgi')
plt.plot(x, true_baseline, 'g--', label='Gerçek Temel Çizgi')
plt.xlabel('Raman Kayması (cm⁻¹)')
plt.ylabel('Yoğunluk (a.u.)')
plt.title('Sentetik Raman Spektrumu')
plt.legend()
plt.grid(True)
plt.show()


#%%



# Gürültü azaltma teknikleri
def denoise_spectrum(spectrum, method='savgol', **kwargs):
    if method == 'savgol':
        # Savitzky-Golay filtresi
        window = kwargs.get('window', 21)
        polyorder = kwargs.get('polyorder', 3)
        return savgol_filter(spectrum, window, polyorder)
    elif method == 'wavelet':
        # Dalgacık dönüşümü (basitleştirilmiş)
        import pywt
        coeffs = pywt.wavedec(spectrum, 'sym5', level=3)
        coeffs[1:] = [pywt.threshold(c, value=0.1*np.max(c)) for c in coeffs[1:]]
        return pywt.waverec(coeffs, 'sym5')
    elif method == 'moving_avg':
        # Hareketli ortalama
        window = kwargs.get('window', 15)
        return np.convolve(spectrum, np.ones(window)/window, mode='same')
    else:
        return spectrum

# Farklı yöntemleri uygula
denoised_savgol = denoise_spectrum(raw_spectrum, 'savgol', window=21, polyorder=3)
denoised_wavelet = denoise_spectrum(raw_spectrum, 'wavelet')
denoised_avg = denoise_spectrum(raw_spectrum, 'moving_avg', window=15)

# Gürültü azaltma sonuçlarını karşılaştır
plt.figure(figsize=(14, 8))
plt.plot(x, raw_spectrum, 'k-', alpha=0.3, label='Ham Spektrum')
plt.plot(x, true_spectrum + true_baseline, 'r--', label='Gerçek Sinyal')
plt.plot(x, denoised_savgol, 'b-', label='Savitzky-Golay (window=21)')
plt.plot(x, denoised_wavelet, 'g-', label='Dalgacık Dönüşümü (sym5)')
plt.plot(x, denoised_avg, 'm-', label='Hareketli Ortalama (window=15)')
plt.xlabel('Raman Kayması (cm⁻¹)')
plt.ylabel('Yoğunluk (a.u.)')
plt.title('Farklı Gürültü Azaltma Yöntemlerinin Karşılaştırılması')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Temel çizgi düzeltme fonksiyonları
def baseline_correction(spectrum, method='asls', **kwargs):
    x = np.arange(len(spectrum))
    
    if method == 'polyfit':
        # Polinom uydurma
        degree = kwargs.get('degree', 3)
        coeffs = np.polyfit(x, spectrum, degree)
        baseline = np.polyval(coeffs, x)
        return spectrum - baseline
    
    elif method == 'asls':
        # Asimetrik En Küçük Kareler (AsLS)
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
        
        lam = kwargs.get('lam', 1e5)  # Düzgünlük parametresi
        p = kwargs.get('p', 0.01)     # Asimetri parametresi
        n_iter = kwargs.get('n_iter', 10)
        
        L = len(spectrum)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        
        for _ in range(n_iter):
            W = diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.T)
            baseline = spsolve(Z, w*spectrum)
            w = p * (spectrum > baseline) + (1-p) * (spectrum < baseline)
        
        return spectrum - baseline
    
    elif method == 'rubberband':
        # Rubberband (lastik bant) yöntemi
        from scipy.spatial import ConvexHull
        
        points = np.column_stack([x, spectrum])
        hull = ConvexHull(points)
        vertices = hull.vertices
        vertices = np.sort(vertices[vertices >= 0])
        
        baseline = np.interp(x, x[vertices], spectrum[vertices])
        return spectrum - baseline
    
    else:
        return spectrum

# Farklı temel çizgi düzeltme yöntemlerini uygula
corrected_poly = baseline_correction(denoised_savgol, 'polyfit', degree=3)
corrected_asls = baseline_correction(denoised_savgol, 'asls', lam=1e5, p=0.01)
corrected_rubber = baseline_correction(denoised_savgol, 'rubberband')

# Sonuçları karşılaştır
plt.figure(figsize=(14, 8))
plt.plot(x, raw_spectrum, 'k-', alpha=0.3, label='Ham Spektrum')
plt.plot(x, true_spectrum, 'r--', label='Gerçek Sinyal (Temel Çizgi Olmadan)')
plt.plot(x, corrected_poly, 'b-', label='Polinom Temel Çizgi Düzeltme (3. derece)')
plt.plot(x, corrected_asls, 'g-', label='AsLS Temel Çizgi Düzeltme')
#plt.plot(x, corrected_rubber, 'm-', label='Rubberband Temel Çizgi Düzeltme')
plt.xlabel('Raman Kayması (cm⁻¹)')
plt.ylabel('Yoğunluk (a.u.)')
plt.title('Farklı Temel Çizgi Düzeltme Yöntemlerinin Karşılaştırılması')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
from scipy.integrate import trapezoid  # trapz yerine trapezoid kullanımı
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_spectrum(spectrum, method='area'):
    """
    Düzeltilmiş spektrum normalizasyon fonksiyonu
    
    Parametreler:
        spectrum (np.array): 1D spektrum verisi
        method (str): Normalizasyon yöntemi ('area', 'max', 'snv', 'msc')
    
    Döndürür:
        np.array: Normalize edilmiş spektrum
    """
    if len(spectrum.shape) != 1:
        raise ValueError("Spektrum 1D olmalıdır")
    
    if method == 'area':
        # Toplam alana göre normalizasyon (trapezoid kullanarak)
        return spectrum / trapezoid(spectrum)
    
    elif method == 'max':
        # Maksimum değere göre normalizasyon
        return spectrum / np.max(spectrum)
    
    elif method == 'snv':
        # Standart Normal Varyant (SNV)
        return (spectrum - np.mean(spectrum)) / np.std(spectrum)
    
    elif method == 'msc':
        # Multiplicative Scatter Correction (MSC) - Düzeltilmiş versiyon
        # Referans spektrum olarak ortalama kullan
        mean_spec = np.mean(spectrum)
        if np.std(spectrum) < 1e-6:  # Sıfıra bölmeyi önle
            return spectrum
        return (spectrum - mean_spec) / np.std(spectrum)
    
    else:
        return spectrum

# Test verisi oluştur
x = np.linspace(500, 3500, 2000)
peaks = [(800, 50, 30), (1200, 80, 40), (1600, 120, 50)]
spectrum = np.zeros_like(x)
for center, height, width in peaks:
    spectrum += height * np.exp(-((x - center)/width)**2)
spectrum += np.random.normal(0, 0.1, len(x))  # Gürültü ekle

# Farklı normalizasyon yöntemlerini uygula
norm_area = normalize_spectrum(spectrum, 'area')
norm_max = normalize_spectrum(spectrum, 'max')
norm_snv = normalize_spectrum(spectrum, 'snv')
norm_msc = normalize_spectrum(spectrum, 'msc')

# Ölçeklendirme (Scaling) - 1D array'i 2D'ye çevirerek
scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

scaled_std = scaler_std.fit_transform(spectrum.reshape(-1, 1)).flatten()
scaled_minmax = scaler_minmax.fit_transform(spectrum.reshape(-1, 1)).flatten()

# Sonuçları görselleştir
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(x, norm_area,'1', label='Toplam Alan Normalizasyonu')
plt.plot(x, norm_max, '2',label='Maksimum Değer Normalizasyonu')
plt.plot(x, norm_snv, '3', label='SNV Normalizasyonu')
plt.plot(x, norm_msc, '4',label='MSC Normalizasyonu')
plt.title('Düzeltilmiş Normalizasyon Yöntemleri')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x, scaled_std, label='Standartlaştırma (Z-score)')
plt.plot(x, scaled_minmax, label='Min-Max Ölçeklendirme')
plt.title('Ölçeklendirme Yöntemleri')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
# Özellik seçimi ve çıkarımı
def feature_selection_extraction(spectra, method='pca', n_components=3):
    if method == 'pca':
        # Temel Bileşen Analizi
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(spectra)
        return transformed, pca
    
    elif method == 'variance':
        # Varyans eşiği ile özellik seçimi
        selector = VarianceThreshold(threshold=0.1)
        selected = selector.fit_transform(spectra)
        return selected, selector
    
    elif method == 'peaks':
        # Pik tespiti ile özellik seçimi
        peaks, _ = find_peaks(spectra.mean(axis=0), prominence=0.5)
        return spectra[:, peaks], peaks
    
    else:
        return spectra, None

# Çoklu spektrum oluştur (demonstrasyon için)
num_spectra = 50
spectra = np.zeros((num_spectra, len(x)))
for i in range(num_spectra):
    # Rastgele pik yükseklikleri ile spektrum oluştur
    rand_peaks = [(p[0], p[1]*np.random.uniform(0.8, 1.2), p[2]) for p in peaks]
    spectra[i], _, _ = create_synthetic_raman_spectrum(x, rand_peaks, baseline)

# Ön işleme uygula
processed_spectra = np.array([baseline_correction(denoise_spectrum(s, 'savgol'), 'asls') 
                           for s in spectra])
processed_spectra = np.array([normalize_spectrum(s, 'snv') for s in processed_spectra])

# Özellik çıkarımı uygula
pca_features, pca_model = feature_selection_extraction(processed_spectra, 'pca', 3)
variance_features, _ = feature_selection_extraction(processed_spectra, 'variance')
peaks_features, peak_indices = feature_selection_extraction(processed_spectra.mean(axis=0).reshape(1, -1), 'peaks')

# Sonuçları görselleştir
plt.figure(figsize=(15, 10))

# PCA sonuçları
plt.subplot(2, 2, 1)
plt.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.7)
plt.xlabel('PC1 (Açıklanan Varyans: %.1f%%)' % (pca_model.explained_variance_ratio_[0]*100))
plt.ylabel('PC2 (Açıklanan Varyans: %.1f%%)' % (pca_model.explained_variance_ratio_[1]*100))
plt.title('PCA ile Boyut İndirgeme')
plt.grid(True)

# Varyans eşiği ile seçilen özellikler
plt.subplot(2, 2, 2)
plt.plot(x, processed_spectra.mean(axis=0), label='Ortalama Spektrum')
plt.plot(x[variance_features.shape[1]:], np.zeros(len(x)-variance_features.shape[1]), 'rx', 
         label='Elenen Bölgeler')
plt.title('Varyans Eşiği ile Özellik Seçimi (%d özellik seçildi)' % variance_features.shape[1])
plt.legend()
plt.grid(True)

# Pik tabanlı özellik seçimi
plt.subplot(2, 2, 3)
plt.plot(x, processed_spectra.mean(axis=0), label='Ortalama Spektrum')
plt.plot(x[peak_indices], processed_spectra.mean(axis=0)[peak_indices], 'ro', 
         label='Seçilen Pikler')
plt.title('Pik Tespiti ile Özellik Seçimi (%d pik seçildi)' % len(peak_indices))
plt.legend()
plt.grid(True)

# PCA bileşenlerinin spektral yükleri
plt.subplot(2, 2, 4)
for i in range(min(3, pca_model.n_components_)):
    plt.plot(x, pca_model.components_[i], 
             label='PC%d (%.1f%%)' % (i+1, pca_model.explained_variance_ratio_[i]*100))
plt.title('PCA Bileşenlerinin Spektral Yükleri')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()