import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import joblib
import json
import os
import time
import re
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Import các hàm từ các module
from meta_model import (
    lstm_train_finetune_predict,
    create_target,
    ultimate_clean,
    merging_df,
    predict_and_save, 
    run_all_models,  
    build_lstm_model,
    fine_tune_lstm_model,
    evaluate_model
)

from RAG import (
    load_raw_data as rag_load_raw_data, # Đổi tên để tránh xung đột nếu có
    clean_data as rag_clean_data,
    format_cleaned_data_to_text as rag_format_cleaned_data_to_text,
    create_embeddings as rag_create_embeddings,
    answer_query_with_rag,
    FILE_CONFIG as RAG_FILE_CONFIG # Import FILE_CONFIG từ RAG.py
)

# Tắt cảnh báo và tải biến môi trường
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
st.set_page_config(layout="wide")
st.title("Ứng dụng Tài chính: Dự báo Giá Cổ phiếu & Hỏi Đáp Thông Minh")
# Định nghĩa các đường dẫn
DATA_DIR_PREDICT = 'DATA EXPLORER CONTEST/Data/' # Dữ liệu cho dự đoán
MODEL_DIR_PREDICT = 'meta model result'          # Mô hình cho dự đoán
NEWS_PATH_RAG = 'DATA EXPLORER CONTEST/News/'    # Dữ liệu tin tức cho RAG

# Cấu hình Ollama và Model Embeddings cho RAG
OLLAMA_MODEL_RAG = os.getenv("OLLAMA_MODEL", "llama3.2:latest") # Lấy từ .env hoặc mặc định
OLLAMA_BASE_URL_RAG = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL_NAME_RAG = os.getenv("EMBEDDING_MODEL_NAME", 'paraphrase-multilingual-MiniLM-L12-v2')

# PHẦN 1: KHỞI TẠO VÀ CACHING CHO DỰ BÁO GIÁ

@st.cache_data
def load_prediction_models_and_data():
    models = {}
    try:
        models['CMG'] = {
            'model': joblib.load(os.path.join(MODEL_DIR_PREDICT, "CMG/model/model.pkl")),
            'X_scaler': joblib.load(os.path.join(MODEL_DIR_PREDICT, "CMG/X_scaler/scaler.pkl")),
            'y_scaler': joblib.load(os.path.join(MODEL_DIR_PREDICT, "CMG/y_scaler/scaler.pkl")),
            'metrics': json.load(open(os.path.join(MODEL_DIR_PREDICT, "CMG/metrics/metrics.json"), 'r', encoding='utf-8')),
        }
        models['FPT'] = {
            'model': joblib.load(os.path.join(MODEL_DIR_PREDICT, "FPT/model/model.pkl")),
            'X_scaler': joblib.load(os.path.join(MODEL_DIR_PREDICT, "FPT/X_scaler/scaler.pkl")),
            'y_scaler': joblib.load(os.path.join(MODEL_DIR_PREDICT, "FPT/y_scaler/scaler.pkl")),
            'metrics': json.load(open(os.path.join(MODEL_DIR_PREDICT, "FPT/metrics/metrics.json"), 'r', encoding='utf-8')),
        }
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình dự đoán: {e}")
        return None, None, None, None

    # Tải dữ liệu đầu vào cho mô hình dự đoán (ví dụ: regressor_cmg, prediction_cmg)
    # Phần này cần đảm bảo dữ liệu được chuẩn bị đúng cách.
    # Dựa trên code gốc, có vẻ như bạn đã có các file CSV chứa prediction_cmg và prediction_fpt
    # và regressor_cmg/fpt được tạo từ dữ liệu gốc.
    try:
        data_hist = {
            "Dữ liệu Lịch sử CMG.csv": pd.read_csv(os.path.join(DATA_DIR_PREDICT, "Dữ liệu Lịch sử CMG.csv")),
            "Dữ liệu Lịch sử FPT.csv": pd.read_csv(os.path.join(DATA_DIR_PREDICT, "Dữ liệu Lịch sử FPT.csv")),
            "FPT_news_features.csv": pd.read_csv(os.path.join(DATA_DIR_PREDICT, "FPT_news_features.csv")),
            "CMG_news_features.csv": pd.read_csv(os.path.join(DATA_DIR_PREDICT, "CMG_news_features.csv")),
            "prediction_CMG_bctc.csv": pd.read_csv(os.path.join(DATA_DIR_PREDICT, "prediction_CMG_bctc.csv")),
            "prediction_CMG_market.csv": pd.read_csv(os.path.join(DATA_DIR_PREDICT, "prediction_CMG_market.csv")),
            "prediction_FPT_bctc.csv": pd.read_csv(os.path.join(DATA_DIR_PREDICT, "prediction_FPT_bctc.csv")),
            "prediction_FPT_market.csv": pd.read_csv(os.path.join(DATA_DIR_PREDICT, "prediction_FPT_market.csv"))
        }
        cleaned_hist_data = ultimate_clean(data_hist)
        merge_cmg = merging_df(cleaned_hist_data, 'CMG').dropna()
        merge_fpt = merging_df(cleaned_hist_data, 'FPT').dropna()

        regressor_cmg_df = create_target(merge_cmg, 'regression')
        regressor_fpt_df = create_target(merge_fpt, 'regression')

        # Tải các file prediction đã được tạo sẵn (quan trọng cho lstm_train_finetune_predict)
        prediction_cmg_df = pd.read_csv(os.path.join(MODEL_DIR_PREDICT, 'Total prediction value/prediction_CMG.csv'))
        prediction_fpt_df = pd.read_csv(os.path.join(MODEL_DIR_PREDICT, 'Total prediction value/prediction_FPT.csv'))

        return models, regressor_cmg_df, regressor_fpt_df, prediction_cmg_df, prediction_fpt_df
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc xử lý dữ liệu dự đoán: {e}")
        return models, None, None, None, None # Trả về models để ít nhất metrics có thể hiển thị

prediction_models, regressor_cmg, regressor_fpt, prediction_cmg, prediction_fpt = load_prediction_models_and_data()

# PHẦN 2: KHỞI TẠO VÀ CACHING CHO HỎI & ĐÁP RAG
@st.cache_resource
def load_embedding_model_rag():
    print(f"ST RAG: Đang tải embedding model: {EMBEDDING_MODEL_NAME_RAG}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME_RAG)
        print("ST RAG: Embedding model đã tải xong.")
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải embedding model cho RAG: {e}")
        return None

@st.cache_resource
def get_ollama_client_rag():
    print(f"ST RAG: Đang kết nối Ollama tại {OLLAMA_BASE_URL_RAG}...")
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL_RAG, api_key='ollama') # api_key thường là 'ollama' hoặc bỏ trống cho local
        print("ST RAG: Đã kết nối Ollama.")
        return client
    except Exception as e:
        st.error(f"Lỗi khi kết nối Ollama: {e}")
        return None

@st.cache_data
def prepare_rag_data(_file_config_rag, _embedding_model_rag):
    """ Chuẩn bị dữ liệu cho RAG """
    if _embedding_model_rag is None:
        st.error("ST RAG: Embedding model chưa được tải, không thể chuẩn bị dữ liệu RAG.")
        return None, None, False

    print("ST RAG: Bắt đầu chuẩn bị dữ liệu RAG...")
    try:
        raw_data_rag = rag_load_raw_data(_file_config_rag) # Sử dụng hàm đã import từ RAG.py
        if not raw_data_rag:
            st.error("ST RAG Lỗi: Không thể tải dữ liệu thô cho RAG.")
            return None, None, False

        cleaned_data_rag = rag_clean_data(raw_data_rag) # Sử dụng hàm đã import
        if not cleaned_data_rag:
            st.error("ST RAG Lỗi: Không thể làm sạch dữ liệu cho RAG.")
            return None, None, False

        all_texts_rag, _ = rag_format_cleaned_data_to_text(cleaned_data_rag, _file_config_rag) # Sử dụng hàm đã import
        if not all_texts_rag:
            st.error("ST RAG Lỗi: Không thể định dạng dữ liệu thành văn bản cho RAG.")
            return None, None, False

        document_embeddings_rag = rag_create_embeddings(all_texts_rag, _embedding_model_rag) # Sử dụng hàm đã import
        if document_embeddings_rag is None:
            st.error("ST RAG Lỗi: Tạo Document Embeddings cho RAG thất bại.")
            return None, None, False

        print(f"ST RAG: Chuẩn bị dữ liệu RAG hoàn tất. {len(all_texts_rag)} văn bản.")
        return all_texts_rag, document_embeddings_rag, True
    except Exception as e:
        st.error(f"Lỗi trong quá trình chuẩn bị dữ liệu RAG: {e}")
        return None, None, False

# Khởi tạo các thành phần RAG
embedding_model_rag_main = load_embedding_model_rag()
ollama_client_rag_main = get_ollama_client_rag()
all_texts_rag_main, document_embeddings_rag_main, rag_is_ready = prepare_rag_data(RAG_FILE_CONFIG, embedding_model_rag_main)

# PHẦN 3: GIAO DIỆN STREAMLIT
tab1, tab2 = st.tabs(["💬 Hỏi & Đáp RAG", "📈 Dự báo Giá Cổ phiếu"])

with tab1:
    st.header("Hỏi & Đáp Thông Minh (RAG)")
    if not rag_is_ready or ollama_client_rag_main is None:
        st.error("Hệ thống Hỏi & Đáp RAG chưa sẵn sàng. Vui lòng kiểm tra console để biết lỗi chi tiết.")
    else:
        st.success(f"Hệ thống RAG đã sẵn sàng với {len(all_texts_rag_main)} mẩu tin tức.")
        user_query_rag = st.text_input("Nhập câu hỏi của bạn về tin tức FPT hoặc CMG:", key="rag_query_input", placeholder="Ví dụ: FPT có thông báo chia cổ tức nào gần đây không?")

        if st.button("Tìm câu trả lời", key="rag_button"):
            if not user_query_rag.strip():
                st.warning("Vui lòng nhập câu hỏi.")
            else:
                with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
                    try:
                        start_time_rag = time.time()
                        answer_rag = answer_query_with_rag(
                            query=user_query_rag,
                            document_embeddings=document_embeddings_rag_main,
                            all_texts=all_texts_rag_main,
                            client=ollama_client_rag_main,
                            model_name=OLLAMA_MODEL_RAG,
                            embedding_model_obj=embedding_model_rag_main
                        )
                        end_time_rag = time.time()
                        processing_time_rag = end_time_rag - start_time_rag
                        st.subheader("Câu trả lời từ Hệ thống RAG:")
                        st.markdown(answer_rag if answer_rag else "Không tìm thấy thông tin hoặc có lỗi trong quá trình xử lý.")
                        st.caption(f"Thời gian xử lý RAG: {processing_time_rag:.2f} giây")
                    except Exception as e:
                        st.error(f"Lỗi khi xử lý câu hỏi RAG: {e}")
        st.subheader('Hướng dẫn sử dụng (Hỏi & Đáp RAG)')
        st.markdown(
        """
        - Nhập câu hỏi của bạn liên quan đến tin tức của FPT hoặc CMG vào ô văn bản.
        - Nhấn "Tìm câu trả lời".
        - Hệ thống sẽ sử dụng kiến thức từ csdl bản tin đã được xử lý để trả lời câu hỏi của bạn.
        """
        )

with tab2:
    st.header("Dự báo Giá Cổ phiếu")
    if prediction_models and regressor_cmg is not None and prediction_cmg is not None: # Kiểm tra dữ liệu đã tải thành công
        ticker = st.selectbox('Chọn mã cổ phiếu', ['CMG', 'FPT'], key='ticker_predict')
        setting = prediction_models[ticker]

        st.subheader(f'Hiệu suất mô hình Meta cho {ticker}')
        # Hiển thị các metrics bạn muốn từ setting['metrics']
        # Ví dụ:
        metrics_display = {k: f"{round(v,4)}%" for k, v in setting['metrics'].items() if k in ['mse', 'mae', 'rmse']}
        st.json(metrics_display)


        days_to_predict = st.slider('Số ngày dự báo', min_value=1, max_value=7, value=3, key='days_slider')
        st.subheader(f'Dự báo giá {ticker} cho {days_to_predict} ngày tới')

        if st.button('Bắt đầu dự báo', key='predict_button'):
            current_regressor = regressor_cmg if ticker == 'CMG' else regressor_fpt
            current_prediction_df = prediction_cmg if ticker == 'CMG' else prediction_fpt

            if current_regressor.empty or current_prediction_df.empty:
                st.error(f"Dữ liệu cần thiết cho dự báo {ticker} chưa sẵn sàng.")
            else:
                with st.spinner(f"Đang dự báo cho {ticker}..."):
                    try:
                        start_time_predict = time.time()
                        _, future_df, fut_metrics = lstm_train_finetune_predict(
                            pred_df=current_prediction_df, # Đây là DataFrame dự đoán từ meta-model
                            actual_df=current_regressor[['Target']], # Đây là DataFrame giá thực tế
                            scaler=setting['X_scaler'], # Scaler của meta-model
                            predict_days=days_to_predict
                        )
                        end_time_predict = time.time()
                        processing_time_predict = end_time_predict - start_time_predict
                        st.markdown(f'**Kết quả dự báo {days_to_predict} ngày tiếp theo cho {ticker}**')
                        st.dataframe(future_df.set_index('Date')) # Dùng dataframe để hiển thị tốt hơn
                        st.markdown(f"**Chỉ số đánh giá của mô hình LSTM tinh chỉnh:**")
                        st.json({
                            'mse': f"{round(fut_metrics['mse'],4)}%",
                            'mae': f"{round(fut_metrics['mae'],4)}%",
                            'rmse': f"{round(fut_metrics['rmse'],4)}%"
                        }) # Hiển thị các metrics của LSTM
                        st.caption(f"Thời gian xử lý dự báo: {processing_time_predict:.2f} giây")
                        # Biểu đồ
                        chart_data = future_df.set_index('Date')[['Predicted Closing Price']]
                        st.line_chart(chart_data)
                    except Exception as e:
                        st.error(f"Lỗi trong quá trình dự báo LSTM: {e}")
        else:
            st.info("Nhấn nút 'Bắt đầu dự báo' để xem kết quả.")

        st.subheader('Hướng dẫn sử dụng (Dự báo)')
        st.markdown(
        """
        - Chọn mã CMG hoặc FPT từ menu thả xuống.
        - Chọn số ngày bạn muốn dự báo bằng thanh trượt.
        - Nhấn "Bắt đầu dự báo" để xem giá dự đoán, các chỉ số đánh giá và biểu đồ xu hướng.
        - Các chỉ số hiệu suất hiển thị ban đầu là của mô hình Meta đã được huấn luyện trước.
        - Các chỉ số sau khi dự báo là của mô hình LSTM được tinh chỉnh nhanh cho việc dự báo ngắn hạn.
        """
        )
    else:
        st.error("Không thể tải dữ liệu hoặc mô hình cần thiết cho chức năng dự báo. Vui lòng kiểm tra lại cấu hình và đường dẫn file.")


