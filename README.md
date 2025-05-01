# YOLOv11 ile Trafik Işığı ve Rengi Tanıma

## Tanıtım 
Bu proje, YOLOv11 derin öğrenme modelini kullanarak resimlerdeki trafik ışıklarını ve renklerini algılayan bilgisayarlı görü uygulamasıdır.

YOLOv11 modeli, gerçek zamanlı nesne tanıma ve sınıflandırma özelliklerine sahip bir derin öğrenme modelidir. Bu proje özellikle trafik ışıklarını ve kırmızı, sarı veya yeşil renklerinden hangisinin yandığını ayırt etmeye yarar.

## Eğitim Detayları

### Veri Seti
Projenin eğitiminde `cinTA_v2` veri seti kullanılmıştır. 
Bu veri setinde `2397` tane `Red`, `Yellow` ve `Green` olarak etiketlenmiş resim bulunmaktadır.

#### Ön İşleme
* `EXIF-orientation stripping`
* `416x416`

#### Augmentation
* Rastgele yatay veya dikey döndürme
* Rastgele kırpma
* Rastgele parlaklık değişimi
* Rastgele `exposure` değişimi
* Rastgele `Gaussian Blur` uygulaması

### Eğitim Süreci
| Parametre  | Açıklama                                                                             | Değer          |
|------------|--------------------------------------------------------------------------------------|----------------|
| Epoch      | Eğitimin süreceği tur sayısı                                                         | 200            |
| Patience   | Doğrulama metriklerinde herhangi bir değişiklik görülmeden harcanacak bekleme süresi | 100            |
| Batch      | Modelin tek seferde kaç parça veri birden alacağı                                    | 16             |
| Image Size | Resimlerin modele gönderilirkenki boyutu                                             | 640            |
| Device     | Eğitimin yapılacağı cihaz (CPU, GPU, TPU)                                            | 0 (Nvidia GPU) |

## Özellikler
Bu proje kapsamında eğittiğimiz model ile kullanıcıya eğitim ve nesne tanıma özellikleri sunduk. Kullanıcı dilerse modeli kullanarak hedef resim veya klasörde tanıma yapabilir veya kendi veri setini kullanarak modelin eğitimini sürdürebilir.


### Tanıma

Program kendisine verilen resim veya videoda tanıdığı trafik ışıklarının etrafına bir `bounding box` çizer ve ışığın rengine göre sınıflandırır.

***Şekil 1: Örnek Tanıma***

![Örnek Tanıma](./assets/detected.jpg)

Programın tanımada kullanabildiği dosya uzantıları:
- .bmp
- .dng
- .jpeg
- .jpg
- .mpo
- .png
- .tif
- .tiff
- .webp
- .pfm
- .HEIC
- .asf
- .avi
- .gif
- .m4v
- .mkv
- .mov
- .mp4
- .mpeg
- .mpg
- .ts
- .wmv
- .webm

## Gereksinimler
- Git
- Python 3
- Nvidia GPU ve CUDA sürücüleri (Daha hızlı tanıma yapmak için gerekli)

## Kurulum

0) Program öncelikle aşağıdaki komut ile klonlanmalıdır:
    ```
    git clone https://github.com/TarikEren/traffic_light_detection.git
    ```

1) Programın olduğu dizinde bir terminal / komut istemi açılmalı veya
    ```
    cd "Proje dizini"
    ```
    komutuyla projenin dizinine geçilmelidir.

2) Gerekli kütüphaneleri yüklemek için
    ```
    pip install -r requirements.txt
    ```
    komutu çalıştırılmalıdır.

## Kullanım
Program kurulumu bitirildikten sonra tanıma veya eğitim amaçlı olarak kullanılabilir.


### Eğitim
```
python.exe ./train.py [parametreler]
```

| Parametre  | Açıklama                                                                                | Varsayılan Değer         |
|------------|-----------------------------------------------------------------------------------------|--------------------------|
| --yaml     | YOLO model eğitimi için kullanılacak olan .yaml dosyasının bağıl konumu                 | ./data.yaml              |
| --dataset  | YOLO formatında olan veri setinin bağıl konumu                                          | ./dataset                |
| --model    | Eğitimin yapılacağı modelin bağıl konumu                                                | ./traffic_light_model.pt |
| --resume   | Yarıda kesilen eğitim kaldığı yerden devam etmeli mi (Pozisyonel argüman, tek başına kullanılır)                                     | False                    |
| --epochs   | Eğitimin devam edeceği tur sayısı                                                       | 100                      |
| --patience | Performans metriklerinde kaç epoch değişik gözlemlenmediği takdirde eğitim durdurulmalı | 100                      |
| --batch    | Model tek seferde kaç parça veri (Resim ve etiket çifti) almalı                         | 16                       |
| --lr       | Modelin başlangıçtaki `learning rate` değeri kaç olmalı                                 | 0.01                     |
| --gpu      | GPU hızlandırma kullanılmalı mı (Pozisyonel argüman, tek başına kullanılır)             | False                    |

* `--model` argümanından sonra `latest` kelimesi kullanıldığında model, yeni eğitim sonucu oluşturulan `runs` dizinine bakarak en son `train` dizinine bakar ve oradaki `last.pt` modelini kullanır

Örnek komut 1:
```
python.exe ./train.py --yaml ./dataset/data.yaml --epochs 10 --gpu
```

Bu komut:
- `./dataset/data.yaml` konumundaki .yaml dosyasını kullanır.
- `10` epoch boyunca eğitim yapar.
- Ekran kartı desteği kullanır. Program CUDA driverı arar ve bulamazsa kullanıcıyı bilgilendirip CPU kullanarak devam eder.
- Geri kalan argümanlar varsayılan değerleriyle kullanılır.

Örnek komut 2:
```
python.exe ./train.py --model latest --resume --epoch 5 --lr 0.1
```
Bu komut:
- Yeni model eğitilmişse en yeni modeli kullanır, eğitilmemişse varsayılan modeli kullanır.
- Kaldığı yerden devam eder.
- 5 epoch boyunca eğitim yapar.
- Başlangıçta learning rate değeri olarak `0.1` değerini kullanır.
- Geri kalan argümanlar varsayılan değerleriyle kullanılır.

Tanıma eğitimi sonuçları `./runs/detect` dizini içerisinde `train` dizinlerine kaydedilir.

Örnek dizin yapısı:
```
runs/
├─ detect/
   ├─ train/
   ├─ train2/
   ├─ train3/
```

### Tanıma
```
python.exe ./detect.py [parametreler]
```
| Parametre    | Açıklama                                                                                                          | Varsayılan Değer         |
|--------------|-------------------------------------------------------------------------------------------------------------------|--------------------------|
| -t, --target | Tanıma yapılacak olan hedef dizin veya dosyanın bağıl konumu (Zorunlu parametre)                                  | -                        |
|-m, --model   | Tanıma yapacak olan modelin bağıl konumu. Varsayılan olarak halihazırda eğitilmiş olan modelin konumu kullanılır. | ./traffic_light_model.pt |

* `--model` argümanından sonra `latest` kelimesi kullanıldığında model, yeni eğitim sonucu oluşturulan `runs` dizinine bakarak en son `train` dizinine bakar ve oradaki `last.pt` modelini kullanır

Örnek komut 1:
```
python.exe ./detect.py --target ./test_source
```
Bu komut:
- `./test_source` dizinindeki dosyalar üzerinde tanıma yapar
- Model olarak varsayılan modeli (`traffic_light_model.pt`) kullanır

Örnek komut 2:
```
python.exe ./detect.py --model latest --target ./test_source/test.jpg
```
Bu komut:
- `./test_source/test.jpg` konumundaki .jpg dosyası üzerinde tanıma yapar
- Yeni model eğitilmişse en yeni modeli kullanır, eğitilmemişse varsayılan modeli kullanır.

Tanıma sonucu elde edilen sonuçların hepsi `detections` isimli bir dizinin altında her bir tanıma için ayrı klasör oluşturulacak şekilde kaydedilir. Örneğin, `./test.jpg` dosyası üzerinde tanıma yapacak olursak kayıt konumu `./detections/detect1/test.jpg` olur.

Program, `detections` klasörünün içine bakar ve başka bir `detect` klasörü varsa olan klasörlerin isimlerini dikkate alarak yeni bir klasör oluşturur.

Örnek olarak `detections` dizini:
```
detections/
├─ detect1/
├─ detect2/
├─ detect3/
```
şeklindeyse, program yeni tanıma sonuçlarını `detect4` dizini içine kaydeder.