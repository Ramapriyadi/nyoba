# LIBRARY
# -------------------------------------------------------

# Framework python untuk tampilan web
import streamlit as st
# Function navigation bar
from streamlit_option_menu import option_menu
# Session state untuk control flow
from streamlit import session_state as ss
# File dummy untuk method automatic re-run/refresh
import dummy
# File functions untuk mengambil semua fungsi yang dibuat
from functions import *
# Untuk menyimpan dan memuat data sebagai objek biner
import pickle
    
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Disable warning
st.set_option("deprecation.showPyplotGlobalUse", False)

# -------------------------------------------------------

# SETTING UTILITIES
# -------------------------------------------------------

# Global set untuk setting halaman
st.set_page_config(
    page_title= "Rama priyadi",
    layout= "wide",
    page_icon= "globe",
    initial_sidebar_state= "expanded",
)

# Modifikasi utilitas
# Hide menu, headerm and footer
st.markdown(
    '''
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .st-emotion-cache-z5fcl4 {padding-top: 1rem;}
    </style>
    ''',
    unsafe_allow_html= True,
)

# CSS on style.css
with open("assets\css\style.css") as f:
    st.markdown(
        "<style>{}</style>".format(f.read()),
        unsafe_allow_html= True,
    )

# SESSION STATE
# -------------------------------------------------------
    
if "stts_resize" not in ss:
    ss.stts_resize = False

# if "stts_augment" not in ss:
#     ss.stts_augment = False

if "stts_clf" not in ss:
    ss.stts_clf = False

# MAIN PROGRAM
# -------------------------------------------------------

def main():
    # INISIASI
    # ---------------------------------------------------
        
    # Variabel untuk kondisi tampilan error
    stts_error = True

    # Inisialisasi PATH file
    PATH = "dataset"

    # NAVBAR
    # ---------------------------------------------------
    
    with st.sidebar:
        selected = option_menu(
            "",
            ["Beranda", "Preprocessing", "Ekstraksi Fitur", "Klasifikasi", "Evaluasi", "Prediksi"],
            icons= ["house", "aspect-ratio", "images", "box-arrow-up", "tags", "bar-chart"],
            styles={
                "container": {"padding": "0 !important", "background-color": "#E6E6EA"},
                "icon": {"color": "#020122", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#020122"},
                "nav-link-selected": {"background-color": "#F4F4F8"},
            }
        )

        st.caption("Rama priyadi")

    # Inisialisasi container
    with st.container():

        # BERANDA
        # -----------------------------------------------
        
        # Try-catch
        try:
            if selected == "Beranda": # Cek pilihan user
                ms_20() # Tambahkan margin
                # Judul penelitian
                st.markdown(
                    '''
                    <h3>
                        KLASIFIKASI PENYAKIT PADA DAUN CABE JAMU MENGGUNAKAN METODE K-NEAREST NEIGHBOR (KNN) DENGAN  EKSTRAKSI FITUR WARNA HSV
                    </h3>
                    ''',
                    unsafe_allow_html= True,
                )
                    
                ms_60() # Tambahkan margin
                # Tampilan isi beranda
                with ml_main():
                    # Abstrak penelitian
                    st.markdown(
                        '''
                        <div>
                            <p>
                                Cabe Jamu (Piper retrofractum Vahl) tergolong pada jenis sirih-sirihan 
                                atau piperaceae yang dibudidayakan di lahan kering dan iklim yang tropis. 
                                Mayoritas tanaman ini tumbuh secara liar di pekarangan.
                                Banyak sekali manfaat yang dimiliki cabe jamu yang masih jarang diketahui oleh banyak masyarakat.
                                Beberapa manfaat dan khasiat pada daun cabe jamu yaitu untuk mengatasi tekanan darah rendah,
                                masuk angin, lemah syahwat dan juga membersihkan rahim setelah melahirkan.
                                Pada proses pembuatan jamu atau obat, bagian yang digunakan adalah buah, akar, dan daun tetapi harus dikeringkan terlebih dahulu.
                            </p>
                        </div>
                        ''',
                        unsafe_allow_html= True,
                    )
        # Exception
        except Exception as e:
            ms_20() # Tambahkan margin
            with ml_main(): # Tampilan sistem
                st.error("Terjadi masalah...")
                # Cek kondisi status error
                if stts_error:
                    st.exception(e) # Tampilkan detail error
                else:
                    pass

        # PREPROCESSING
        # -----------------------------------------------
        
        # Try-catch
        try:
            if selected == "Preprocessing": # Cek pilihan user
                ms_20() # Tambahkan margin
                prn_judul(
                    "Data Citra Daun Cabe Jamu", 
                    size= 3, 
                    line= True
                ) # Tampilkan judul
                
                left, right = ml_right() # Layouting tampilan web
                # Tampilan layout kiri
                with left:
                    prn_caption("Setting Parameter", 3)
                    ms_20() # Tambahkan margin
                    sub_left, sub_right = ml_double() # Layouting tampilan web
                    # Tampilan layout kiri
                    with sub_left:
                        # Buat objek text input
                        panjang = st.number_input(
                            "Panjang Citra",
                            min_value= 100,
                            max_value= 1000,
                            value= 300,
                            step= 50,
                            key= "Number input untuk ukuran panjang citra"
                        )
                    # Tampilan layout kanan
                    with sub_right:
                        # Buat objek text input
                        lebar = st.number_input(
                            "Lebar Citra",
                            min_value= 100,
                            max_value= 1000,
                            value= 300,
                            step= 50,
                            key= "Number input untuk ukuran lebar citra"
                        )
                    ms_20() # Tambahkan margin
                    # Button untuk menjalan proses resize
                    btn_resize = st.button(
                        "Resize",
                        use_container_width= True,
                        key= "Button untuk trigger proses resize citra",
                    )
                # Tampilan bagian kanan
                with right:
                    # Cek status dari btn_resize
                    if btn_resize:
                        # Ubah status session state
                        ss.stts_resize = True
                        # Menampilkan spinner progress bar
                        with st.spinner("Resize citra sedang berlangsung..."):
                            # Panggil function untuk mendapatkan informasi PATH setiap citra
                            path_original_img = get_filepath(
                                PATH,
                                "path_images"
                            )
                            # Lakukan proses resize bedasarkan PATH citra yg didapatkan
                            resize_image(
                                path_original_img,
                                panjang,
                                lebar
                            )
                            # Dapatkan path gambar yg telah di resize
                            path_resized_img = get_filepath(
                                "processed/resized",
                                "path_resized"
                            )
                            show_images(path_resized_img)
                    elif ss.stts_resize:
                        # Dapatkan path gambar yg telah di resize
                        path_resized_img = get_csv(
                            "processed/dataframe/path_resized.csv"
                        )
                        show_images(path_resized_img)
        # Exception
        except Exception as e:
            ms_20() # Tambahkan margin
            with ml_main(): # Tampilan sistem
                st.error("Terjadi masalah...")
                # Cek kondisi status error
                if stts_error:
                    st.exception(e) # Tampilkan detail error
                else:
                    pass

        # AUGMENTASI
        # -----------------------------------------------
        
        # Try-catch
        # try:
        #     if selected == "Augmentasi": # Cek pilihan user
        #         ms_20() # Tambahkan margin
        #         prn_judul(
        #             "Augmentasi Gambar",
        #             size= 3,
        #             line= True
        #         ) # Tampilkan judul
                
        #         left, right = ml_right() # Layouting tampilan web
        #         # Tampilan layout kiri
        #         with left:
        #             prn_caption(
        #                 "Setting Parameter",
        #                 3,
        #             )
        #             ms_20() # Tambahkan margin
        #             # Buat objek untuk input number
        #             count_img = st.number_input(
        #                 "Jumlah Data/Kelas",
        #                 min_value= 1,
        #                 max_value= 5000,
        #                 value= 1000,
        #                 step= 50,
        #                 key= "Number input untuk target augmentasi",
        #             )
        #             ms_20() # Tambahkan margin
        #             # Button untuk menjalankan proses augmentasi
        #             btn_augment = st.button(
        #                 "Submit",
        #                 use_container_width= True,
        #                 key= "Button untuk trigger proses augmentasi",
        #             )
        #         # Tampilan layout kanan
        #         with right:
        #             # Cek status dari btn_augment
        #             if btn_augment:
        #                 # Ubah status session state
        #                 ss.stts_augment = True
        #                 # Menampilkan spinner program bar
        #                 with st.spinner("Augmentasi citra sedang berlangsung..."):
        #                     # Dapatkan path gambar yg telah di resize
        #                     path_resized_img = get_csv("processed/dataframe/path_resized.csv")
        #                     # Perulangan untuk setiap label dalam data
        #                     for label in path_resized_img["label"].unique():
        #                         PATH = f"processed/resized/{label}"
        #                         # Lakukan augmentasi pada data train
        #                         augment_image(PATH, count_img)
        #                     # Dapatkan path gambar yg telah di augmentasi
        #                     path_augment_img = get_filepath("processed/resized", "path_augment")
        #                     show_images(path_augment_img) # Tampilkan gambar yg telah diproses
        #             elif ss.stts_augment:
        #                 path_augment_img = get_csv("processed/dataframe/path_augment.csv")
        #                 show_images(path_augment_img) # Tampilkan gambar yg telah diproses
        # # Exception
        # except Exception as e:
        #     ms_20() # Tambahkan margin
        #     with ml_main(): # Tampilan sistem
        #         st.error("Terjadi masalah...")
        #         # Cek kondisi status error
        #         if stts_error:
        #             st.exception(e) # Tampilkan detail error
        #         else:
        #             pass

        # EKSTRAKSI FITUR
        # -----------------------------------------------
        
        # Try-catch
        try:
            if selected == "Ekstraksi Fitur": # Cek pilihan user
                ms_20() # Tambahkan margin
                prn_judul(
                    "Hasil Ekstraksi Fitur HSV",
                    size= 3,
                    line= True
                ) # Tampilkan judul
                
                # Dapatkan DataFrame dari citra yg akan di ekstraksi fitur
                df_img = get_csv("processed/dataframe/path_resized.csv")
                # Buat dataframe kosong untuk menampung hasil pelatihan
                data_features = pd.DataFrame(columns= ["filename", "hue", "saturation", "value", "label"])
                # Loop untuk mengambil setiap label dalam DataFrame
                for label in df_img["label"].unique():
                    # Lakukan ekstraksi fitur untuk masing-masing data
                    features = extraction_HSV_features(f"processed/resized/{label}", label)
                    # Merger dataframe
                    data_features = pd.concat([data_features, features], ignore_index= True)

                with ml_main(): # Tampilan sistem
                    st.dataframe(
                        data_features, 
                        use_container_width= True, 
                        hide_index= True
                    )

                # Simpan dataframe
                save_df(data_features, "processed/dataframe/features")
        # Exception
        except Exception as e:
            ms_20() # Tambahkan margin
            with ml_main(): # Tampilan sistem
                st.error("Terjadi masalah...")
                # Cek kondisi status error
                if stts_error:
                    st.exception(e) # Tampilkan detail error
                else:
                    pass

        # KLASIFIKASI
        # -----------------------------------------------
        
        # Try-catch
        try:
            if selected == "Klasifikasi": # Cek pilihan user
                ms_20() # Tambahkan margin
                prn_judul(
                    "Klasifikasi k-Nearest Neighbour",
                    size= 3,
                    line= True
                ) # Tampilkan judul
                
                # Baca data hasil ekstraksi fitur
                data = get_csv("processed/dataframe/features.csv")
                
                # Layouting kolom
                left, right = ml_right()
                # Tampilan layout kiri
                with left:
                    prn_caption(
                        "Setting Parameter",
                        size= 3,
                    )
                    neighbors = st.number_input(
                        "Nilai k tetangga",
                        min_value= 1,
                        step= 1,
                        max_value= int(len(data) / 2),
                        value= 3,
                        key= "Number input untuk k tetangga terdekat",
                    )
                    kfold = st.selectbox(
                        "Pilih jumlah Fold",
                        ["4-Fold", "5-Fold", "10-Fold"],
                        index= 2,
                        key= "Selectbox untuk memilih jumlah subset Fold",
                    )
                    # Branching nilai berdasarkan value kfold
                    kfold = 4 if kfold == "4-Fold" else (5 if kfold == "5-Fold" else 10)
                    ms_20()
                    btn_clf = st.button(
                        "Submit",
                        use_container_width= True,
                        key= "Button untuk trigger proses klasifikasi",
                    )

                with right: # Tampilan kanan
                    # Cek status dari btn_clf
                    if btn_clf:
                        ss.stts_clf = True
                        # Jalankan analisis klasifikasi
                        kfold_cv(data.iloc[:, 1:4].values, data["label"], k= kfold, neighbor= neighbors)
                        show_predict()

                    elif ss.stts_clf:
                        show_predict()
        # Exception
        except Exception as e:
            ms_20() # Tambahkan margin
            with ml_main(): # Tampilan sistem
                st.error("Terjadi masalah...")
                # Cek kondisi status error
                if stts_error:
                    st.exception(e) # Tampilkan detail error
                else:
                    pass

        # EVALUASI
        # -----------------------------------------------
        
        # Try-catch
        try:
            if selected == "Evaluasi": # Cek pilihan user
                ms_20() # Tambahkan margin
                prn_judul(
                    "Kinerja Hasil Klasifikasi",
                    size= 3,
                    line= True
                ) # Tampilkan judul

                # Memuat list dari file lokal
                with open('processed/picklefile/all_y_test.pkl', 'rb') as file:
                    all_y_test = pickle.load(file)
                with open('processed/picklefile/all_y_pred.pkl', 'rb') as file:
                    all_y_pred = pickle.load(file)
                with open('processed/picklefile/all_score.pkl', 'rb') as file:
                    score_proba = pickle.load(file)
                    
                for fold, (y_test, y_pred) in enumerate(zip(all_y_test, all_y_pred)):
                    LABELS = np.unique(y_test)
                    num_classes = len(LABELS)

                    y_test_bin = label_binarize(y_test, classes=LABELS)

                    with st.expander(f"fold ke-{fold + 1}", expanded= False):
                    # Layouting kolom
                        left, right = ml_double()
                        with left: # Tampilan kiri
                            create_confusion_matrix(y_test, y_pred)
                        with right:
                            acc = show_acc(y_test, y_pred)
                            create_classification_report(y_test, y_pred)
                            ms_20()
                            st.markdown("---")

                            y_score = score_proba[fold]
                            fpr, tpr, roc_auc = dict(), dict(), dict()
                            for i in range(len(LABELS)):
                                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                                roc_auc[i] = auc(fpr[i], tpr[i])
                                

                            plt.figure()
                            for i in range(len(LABELS)):
                                plt.plot(fpr[i], tpr[i], label='Class {0} (area = {1:0.2f})'.format(LABELS[i], roc_auc[i]))

                            plt.plot([0, 1], [0, 1], 'k--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('ROC Curve')
                            plt.legend(loc="lower right")
                            st.pyplot()
                            
        # Exception
        except Exception as e:
            ms_20() # Tambahkan margin
            with ml_main(): # Tampilan sistem
                st.error("Terjadi masalah...")
                # Cek kondisi status error
                if stts_error:
                    st.exception(e) # Tampilkan detail error
                else:
                    pass

        # PREDIKSI
        # -----------------------------------------------
        
        # Try-catch
        try:
            if selected == "Prediksi": # Cek pilihan user
                ms_20() # Tambahkan margin
                prn_judul(
                    "Prediksi Data",
                    size= 3,
                    line= True
                ) # Tampilkan judul

                left, right = ml_left()
                with left: # Tampilan kiri
                    img = st.file_uploader("Upload gambar", type= ["png", "jpg", "jpeg"])
                    if img is not None:
                        # Tampilkan gambar
                        st.image(img, caption= img.name, use_column_width= True)
                with right:
                    if img is not None:
                        temp_file = tempfile.NamedTemporaryFile(delete= False)
                        temp_file.write(img.read())
                        temp_file.close()

                        filepath = temp_file.name
                        
                        # resize image
                        if not os.path.exists("processed/predict"):
                            os.makedirs("processed/predict")

                        img_resized = cv2.imread(filepath)
                        size_image = (224, 224)
                        image_resize = cv2.resize(img_resized, size_image, interpolation= cv2.INTER_AREA)

                        img_HSV = cv2.cvtColor(image_resize, cv2.COLOR_BGR2HSV)
                        med_H = np.median(img_HSV[:, :, 0])
                        med_S = np.median(img_HSV[:, :, 1])
                        med_V = np.median(img_HSV[:, :, 2])

                        # Simpan nilai median masing-masing channel
                        features_data = np.array([[med_H, med_S, med_V]])
                        # data = get_csv("processed/dataframe/features.csv")
                        # X_train, y_train = data.iloc[:, 1:4].values, data["label"].values

                        # knn = KNeighborsClassifier()
                        # knn.fit(X_train, y_train)

                        with open("processed/picklefile/knn_model.pkl", "rb") as file:
                            knn = pickle.load(file)

                        result = knn.predict(features_data)
                        st.caption("## Hasil Prediksi")
                        ms_20()
                        st.info(f"**{result[0]}**")

        # Exception
        except Exception as e:
            ms_20() # Tambahkan margin
            with ml_main(): # Tampilan sistem
                st.error("Terjadi masalah...")
                # Cek kondisi status error
                if stts_error:
                    st.exception(e) # Tampilkan detail error
                else:
                    pass

# Jalankan Program
if __name__ == "__main__":
    main()
