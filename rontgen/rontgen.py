import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

# Ana klasör yolu
dataset_folder = '../dataset/'  # rontgen klasörü içinden çalıştıracağımız için bir üst dizine çıkıyoruz

# Resimlerin ve XML dosyalarının bulunduğu klasör yolları
image_folder = os.path.join(dataset_folder, 'images')
xml_folder = os.path.join(dataset_folder, 'xmls')

# İşlenmiş resimlerin kaydedileceği klasör yolu
output_folder = os.path.join(dataset_folder, 'processed_images')

# Klasörler yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Klasördeki tüm resim dosyalarını listele
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# İşlenen dosyaların kaydı
processed_files = []

# Hataları kaydetmek için liste
error_files = []

# Her bir resim dosyası için işlem yap
for image_file in image_files:
    try:
        # Görüntüyü yükleme
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path, 0)
        
        if img is None:
            print(f"Error loading image: {img_path}")
            error_files.append(image_file)
            continue

        # Histogram eşitleme
        equalized_image = cv2.equalizeHist(img)

        # 5x5 boyutunda kernel oluşturma
        kernel = np.ones((5, 5), np.uint8)

        # Dilation işlemi
        img_dilation = cv2.dilate(equalized_image, kernel, iterations=1)

        # Erozyon işlemi
        eroded_image = cv2.erode(img_dilation, kernel, iterations=1)

        # Closing işlemi
        closing = cv2.morphologyEx(eroded_image, cv2.MORPH_CLOSE, kernel)

        # Kenar bulma işlemi
        edges = cv2.Canny(closing, 100, 200)

        # Kenar tespitinden sonra konturları çizme
        contour_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Kontur bulma
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Kontur analizi ve kırık olabilecek konturları belirleme
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Filtreleme kriterleri
            if 100 < area < 2000 and 0.5 < aspect_ratio < 2 and w > 5 and h > 5 and x > 10 and y > 10:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) > 4:  # Daha karmaşık şekiller için
                    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Kırmızı renk
                    print(f"Kırmızı kutu çizildi: x={x}, y={y}, w={w}, h={h}, aspect_ratio={aspect_ratio}")

        # XML dosyasını yükleme
        xml_path = os.path.join(xml_folder, os.path.splitext(image_file)[0] + '.xml')
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # XML dosyasındaki kırık koordinatlarını okuma
            for box in root.iter('bndbox'):
                xmin = int(box.find('xmin').text)
                ymin = int(box.find('ymin').text)
                xmax = int(box.find('xmax').text)
                ymax = int(box.find('ymax').text)
                cv2.rectangle(contour_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Mavi renk

        # İşlenmiş resmi kaydet
        output_path = os.path.join(output_folder, 'processed_' + image_file)
        cv2.imwrite(output_path, contour_image)
        processed_files.append(image_file)

    except Exception as e:
        print(f"Error processing file {image_file}: {e}")
        error_files.append(image_file)

print(f"Processed {len(processed_files)} files.")
print(f"Encountered errors in {len(error_files)} files.")
print("Tüm resimler işlendi ve kaydedildi.")
cv2.waitKey(0)
cv2.destroyAllWindows()
