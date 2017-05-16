import csv
import os


def read_csv():
    with open('./Temp_data_Set/Test_Dataset_csv/Label.csv') as csvfile:
        rows = csv.reader(csvfile)

        age_list = []
        for row in rows:
            age_list.append(int(row[0]))
        print(sorted(set(age_list)))
        label_age = {i: val for i, val in enumerate(sorted(set(age_list)))}
        print(label_age)
    return age_list


def write_csv():
    image_dir_path = './Temp_data_Set/Test_Dataset_png/'
    image_dir = [image_dir_path + filename for filename in os.listdir(image_dir_path)]
    # label_dir_path = './Temp_data_Set/Test_Dataset_csv/Label.csv'
    label_dir_path = './csv_label.csv'
    with open(label_dir_path) as csvfile:
        rows = csv.reader(csvfile)
        age_list = []
        for row in rows:
            age_list.append(int(row[0]))
    data = zip(image_dir, age_list)

    with open("name_encode_label.csv", "w", newline='') as f:
        fieldnames = ['filename', 'label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in data:
            writer.writerow({'filename': i[0], 'label': i[1]})


if __name__ == '__main__':
    print(read_csv())
    # write_csv()
