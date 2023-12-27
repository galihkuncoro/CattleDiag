import tensorflow as tf
import numpy as np
import streamlit as st
import joblib

# Memuat model TensorFlow yang telah disimpan
model = tf.keras.models.load_model('cattlediag_tf.h5')
#model = joblib.load('svc.pkl')


# Judul web
st.title('Prediksi Penyakit Pada Hewan Ternak')

# Widget multiselect untuk gejala
gejala_options = ['Kesulitan bernapas', 'Tidak nafsu makan', 'Keluar cairan pada hidung', 'Cepat terengah-engah', 
                  'Terlihat lesu', 'Terlihat gelisah', 'Sering berbaring', 'Kematian mendadak', 'Stres',
                  'Demam', 'Dehidrasi', 'Kotoran berdarah', 'Demam tinggi', 'Berjalan sempoyongan',
                  'Gemetar', 'Kondisi lemah', 'Mudah ambruk', 'Perdarahan lubang kumlah', 'Keguguran', 
                  'Kemandulan temporer', 'Kemandulan permanen', 'Penurunan produksi susu', 'Cairan janin keruh',
                  'Radang buah pelir', 'Radang saluran sperma', 'Pembengkakan persendian lutut', 'Mata sayu dan berair',
                  'Mata hiperemik', 'Suara ngorok', 'Konstipasi', 'Pembengkakan kepala', 'Pembengkakan tenggorokan',
                  'Pembengkakan leher', 'Keluar cairan pada mata', 'Erosi permukaan lidah', 'Moncong kering',
                  'Moncong pecah terisi eksudat', 'Hidung tersumbat kerak', 'Tubuh kurus', 'Kornea mata keruh keputihan',
                  'Radang kulit', 'Pembengkakan kelenjar limfe', 'Sembelit', 'Radang otak' , 'Kelumpuhan sebelum mati', 
                  'Bulu kusam',  'Radang lidah',  'Radang mulut',  'Hypersalivasi',  'Kesulitan mengunyah',
                  'Lepuh pada gusi', 'Kuku dapat terlepas', 'Selaput lendir lidah terkelupas', 'Kekakuan anggota gerak',
                  'Pincang', 'Pembengkakan punggung', 'Bulu rontok', 'Anemia', 'Busung daerah dagu', 'Mencret', 'Bulu berdiri', 
                  'Perut membesar', 'Selaput lendir pucat kekuningan', 'Busung di bawah perut', 'Hambatan pertumbuhan',
                  'Kulit kering', 'Busung di bawah rahang', 'Kotoran bau asam butirat', 'Kotoran berlendir dan berdarah',
                  'Pembengkakan mata', 'Luka cornea mata', 'Kekeruhan cornea mata', 'Photophobia', 'Lakrimasi', 'Terdapat cacing di mata',
                  'Kebutaan', 'Gatal hebat', 'Menggosok tubuh ke dinding', 'Menggigit gigit tubuh', 'Luka-luka dan lecet',
                  'Lepuh bernanah pada kulit', 'Kulit mengeras', 'Kulit menebal serta melipat-lipat','Perut sebelah kiri membesar',
                  'Mengerang  kesakitan', 'Perut kembung', 'Menjulurkan lehernya saat berbaring', 'Luka pada selaput lendir mulut',
                  'Berkeringat darah', 'Mumifikasi fetus', 'Kelahiran cacat', 'Gangguan regurgitasi', 'Pembesaran organ', 'Kegagalan jantung', 'Edema kornea',
                  'Red nose', 'Radang hidung dan tenggorokan', 'Kelumpuhan', 'Pembengkakan otot gerak',
                  'Terdengar suara krepitasi', 'Kenaikan suhu rektal', 'Pembengkakan paha', 'Pembengkakan bahu',
                  'Koma', 'Suhu subnormal', 'Selaput lendir pucat', 'Mulut basah',
                  'Denyut jantung tak teratur', 'Sendi bengkak', 'Meningitis', 'Keropeng/kudis', 'Kulit kasar',
                  'Lymphodenopathy', 'Pembengkakan paru-paru', 'Pembengkakan limfa', 'Hydrothorax', 'Mulut dan hidung berbusa',
                  'Mulut bergerak seperti mengunyah', 'kemerahan kulit (eritema)', 'Keropeng pada mulut',
                  'Popula', 'Vesikula', 'Lesi kulit terbentuk di mulut', 'Radang basah', 'Kulit menebal',
                  'Konjungtiva', 'Urin berwarna gelap', 'Kulit dan mata kekuningan', 'Pembengkakan ambing dan putting',
                  'Perubahan warna susu', 'Gangren ambing', 'Susu bening encer', 'Ambing keras dan memerah',
                  'Konstriksi pupil', 'Pembengkakan rahang', 'Sekresi mata purulen',
                  'Lesi pada alat kelamin', 'Batuk', 'Kembung rumen', 'Radang paru', 'Kejang-kejang',
                  'Kencing berdarah', 'Pembesaran jantung', 'Mukosa cepat kuning', 'Konvulsi',
                  'Pembengkakan ambing dan puting', 'Mata lembab', 'Pembesaran pembuluh darah kornea'  ] 

Gejala = st.multiselect('Pilih Gejala', gejala_options)

# Tombol prediksi
if st.button('Prediksi penyakit'):
    total_gejala = 148  # Jumlah total gejala pada model Anda
    input_data = np.zeros(total_gejala)

    # Batas jumlah gejala yang dapat dipilih untuk prediksi
    batas_gejala = 19

    # Loop gejala yang dipilih dan update input_data
    for gejala in gejala_options[:batas_gejala]:
        if gejala in Gejala:
            index = gejala_options.index(gejala)
            input_data[index] = 1

    # Ubah bentuk input_data agar sesuai dengan bentuk input model
    input_data = input_data.reshape(1, -1)  # Ubah menjadi (1, total_gejala) jika hanya satu sampel

    # Hanya ambil 19 gejala pertama untuk diprediksi
    input_data = input_data[:, :19]

    # Melakukan prediksi menggunakan model
    prediction = model.predict(input_data)

    # Dapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(prediction)

    # Daftar nama penyakit untuk memetakan indeks prediksi
    nama_penyakit = [   "Pneumonia", "Diare", "Anthrax", "Brucellosis", "Septichaemia Epizootica (SE)",
                    "Malignant Catrrahal Fever", "Apthea Epizootica (AE)", "Bovine Epheral Fever (BEF)",
                    "Trypanosomiasis", "Fasciolasis", "Nematodosis", "Thelazia", "Scabies", "Bloat",
                    "Jembrana", "Akabane", "Enzootic Bovine leukosis (EBL)", "Infectious Bovine Rhinotracheitis (IBR)",
                    "Blackleg", "Colibacillosis", "Dermatophilosis", "Heartwater", "Orf", "Leptospirosis",
                    "Listeriosis", "Mastitis", "Paratuberkulosis", "Pink eye", "Salmonellosis", "Tuberkulosis",
                    "Aspergillosis", "Anaplasmosis", "Babesiosis"]

    # Dapatkan nama penyakit yang diprediksi
    penyakit_terprediksi = nama_penyakit[predicted_class_index]

    # Menampilkan hasil prediksi
    st.write(f'Prediksi: {penyakit_terprediksi}')
