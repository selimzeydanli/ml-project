import json
import random

# Random float sayı üretmek için fonksiyon
def generate_random_float(min_value, max_value, decimals):
    return round(random.uniform(min_value, max_value), decimals)

# Boş bir liste oluşturun
data = []

# 5000 rastgele öğe oluşturun
for _ in range(1450):
    # 123 ile 34230 (dahil) arasında rastgele bir tam sayı üretin
    sup_id = random.randint(123, 34230)

    # 37.2 ile 39.5 arasında 2 ondalıklı rastgele bir float sayı üretin
    arr_lat = generate_random_float(37.2, 39.5, 2)

    # 27.2 ile 44.4 arasında 2 ondalıklı rastgele bir float sayı üretin
    arr_lon = generate_random_float(27.2, 44.4, 2)

    # Öğeyi listeye ekleyin
    data.append({'SupID': sup_id, 'Arr. Lat': arr_lat, 'Arr. Lon': arr_lon})

# Listeyi JSON dosyası olarak kaydedin
with open('C:\\Users\\Selim\\Desktop\\ml-project\\input\\suppliers_lstm.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

print('Dictionary saved successfully to C:\\Users\\Selim\\Desktop\\ml-project\\input\\suppliers_lstm.json')