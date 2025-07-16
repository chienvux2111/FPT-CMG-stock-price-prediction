import warnings
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
# --- Tắt cảnh báo ---
warnings.filterwarnings("ignore")
# --- Tải biến môi trường ---
load_dotenv()

# Hàm làm sạch dữ liệu
def clean_data(data_dict):
    """Làm sạch dữ liệu trong dictionary các DataFrame thô."""
    news = data_dict.copy()
    for key, value in news.items():
        print(f"Đang làm sạch dữ liệu từ key: {key}...")
        if not isinstance(value, pd.DataFrame):
            print(f"Cảnh báo: Giá trị cho key '{key}' không phải là DataFrame. Bỏ qua.")
            continue
        df = value.copy()
        date_feature_cols_to_create = ['year', 'month', 'day', 'weekday', 'day_name', 'month_name', 'week', 'quarter', 'is_weekend', 'date_only', 'time_only']

        if key == 'cafef_news':
            df['summary'] = df['summary'].astype(str).str.replace(r'_x000D_\n', '\n', regex=True)
            df['summary'] = df['summary'].astype(str).str.replace(r'_x000D_', '\n', regex=True)
            df['summary'] = df['summary'].str.strip()
            date_col_name = 'date'
            if date_col_name in df.columns:
                df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
                mask = df[date_col_name].notna()
                if mask.any():
                    df.loc[mask, f'{date_col_name}_year'] = df.loc[mask, date_col_name].dt.year
                    df.loc[mask, f'{date_col_name}_month'] = df.loc[mask, date_col_name].dt.month
                    df.loc[mask, f'{date_col_name}_day'] = df.loc[mask, date_col_name].dt.day
                    df.loc[mask, f'{date_col_name}_weekday'] = df.loc[mask, date_col_name].dt.weekday
                    df.loc[mask, f'{date_col_name}_day_name'] = df.loc[mask, date_col_name].dt.day_name()
                    df.loc[mask, f'{date_col_name}_month_name'] = df.loc[mask, date_col_name].dt.month_name()
                    df.loc[mask, f'{date_col_name}_week'] = df.loc[mask, date_col_name].dt.isocalendar().week.astype(pd.Int64Dtype())
                    df.loc[mask, f'{date_col_name}_quarter'] = df.loc[mask, date_col_name].dt.quarter.astype(pd.Int64Dtype())
                    df.loc[mask, f'{date_col_name}_is_weekend'] = df.loc[mask, date_col_name].dt.weekday >= 5
                    df.loc[mask, f'{date_col_name}_date_only'] = df.loc[mask, date_col_name].dt.date
                    df.loc[mask, f'{date_col_name}_time_only'] = df.loc[mask, date_col_name].dt.time
            else:
                print(f"Cảnh báo: Cột ngày '{date_col_name}' không tồn tại trong {key}.")
            # Đảm bảo các cột feature ngày tháng tồn tại
            for feature in date_feature_cols_to_create:
                col_name_feature = f'{date_col_name}_{feature}'
                if col_name_feature not in df.columns:
                    df[col_name_feature] = np.nan if feature not in ['day_name', 'month_name'] else "Không rõ"
            # Thêm cột StockID
            if 'title' in df.columns:
                df['StockID'] = df['title'].apply(lambda x: 'FPT' if 'FPT' in str(x).upper() else ('CMG' if 'CMG' in str(x).upper() else 'Unknown'))

        elif key == 'dividends':
            date_cols_original = ['Ex-Dividend Date', 'Record Date', 'Execution Date']
            for col_orig in date_cols_original:
                if col_orig in df.columns:
                    df[col_orig] = pd.to_datetime(df[col_orig], dayfirst=True, errors='coerce')
                    mask = df[col_orig].notna()
                    if mask.any():
                        df.loc[mask, f'{col_orig}_year'] = df.loc[mask, col_orig].dt.year
                        df.loc[mask, f'{col_orig}_month'] = df.loc[mask, col_orig].dt.month
                        df.loc[mask, f'{col_orig}_day'] = df.loc[mask, col_orig].dt.day
                        df.loc[mask, f'{col_orig}_weekday'] = df.loc[mask, col_orig].dt.weekday
                        df.loc[mask, f'{col_orig}_day_name'] = df.loc[mask, col_orig].dt.day_name()
                        df.loc[mask, f'{col_orig}_month_name'] = df.loc[mask, col_orig].dt.month_name()
                        df.loc[mask, f'{col_orig}_week'] = df.loc[mask, col_orig].dt.isocalendar().week.astype(pd.Int64Dtype())
                        df.loc[mask, f'{col_orig}_quarter'] = df.loc[mask, col_orig].dt.quarter.astype(pd.Int64Dtype())
                        df.loc[mask, f'{col_orig}_is_weekend'] = df.loc[mask, col_orig].dt.weekday >= 5
                        df.loc[mask, f'{col_orig}_date_only'] = df.loc[mask, col_orig].dt.date
                        df.loc[mask, f'{col_orig}_time_only'] = df.loc[mask, col_orig].dt.time
                else: print(f"Cảnh báo: Cột ngày '{col_orig}' không tồn tại trong {key}.")
                for feature in date_feature_cols_to_create:
                    col_name_feature = f'{col_orig}_{feature}'
                    if col_name_feature not in df.columns:
                        df[col_name_feature] = np.nan if feature not in ['day_name', 'month_name'] else "Không rõ"
            df['Dividend (VND)'] = np.nan
            if 'Event Content' in df.columns:
                dividend_extracted = df['Event Content'].astype(str).str.extract(r'(\d{1,3}(?:,\d{3})*) đồng/CP', expand=False)
                valid_dividends = dividend_extracted.dropna()
                if not valid_dividends.empty:
                    numeric_dividends = valid_dividends.str.replace(',', '', regex=False)
                    df.loc[valid_dividends.index, 'Dividend (VND)'] = pd.to_numeric(numeric_dividends, errors='coerce')
            else: print(f"Cảnh báo: Không tìm thấy cột 'Event Content' để trích xuất cổ tức trong {key}.")
            df['Bonus Ratio'] = np.nan
            if 'Event Content' in df.columns:
                ratio_df = df['Event Content'].astype(str).str.extract(r'(\d+):(\d+)', expand=True)
                if ratio_df is not None and not ratio_df.isnull().all(axis=None):
                    ratio_df[0] = pd.to_numeric(ratio_df[0], errors='coerce'); ratio_df[1] = pd.to_numeric(ratio_df[1], errors='coerce')
                    valid_ratios = ratio_df.dropna()
                    if not valid_ratios.empty:
                        df.loc[valid_ratios.index, 'Bonus Ratio'] = valid_ratios.apply(lambda x: x[0]/x[1] if x[1] != 0 else np.nan, axis=1)
            else: print(f"Cảnh báo: Không tìm thấy cột 'Event Content' để trích xuất tỷ lệ thưởng trong {key}.")
            if 'Event Type' in df.columns: df['Event Type'] = df['Event Type'].astype(str).str.lower().str.strip().replace('nan', 'Không rõ')
            else: print(f"Cảnh báo: Không tìm thấy cột 'Event Type' trong {key}.")

        elif key == 'internal_transactions':
            date_cols_original = ['Registered From Date', 'Registered To Date', 'Executed From Date', 'Executed To Date']
            for col_orig in date_cols_original:
                if col_orig in df.columns:
                    df[col_orig] = pd.to_datetime(df[col_orig], dayfirst=True, errors='coerce')
                    mask = df[col_orig].notna()
                    if mask.any():
                        df.loc[mask, f'{col_orig}_year'] = df.loc[mask, col_orig].dt.year
                        df.loc[mask, f'{col_orig}_month'] = df.loc[mask, col_orig].dt.month
                        df.loc[mask, f'{col_orig}_day'] = df.loc[mask, col_orig].dt.day
                        df.loc[mask, f'{col_orig}_weekday'] = df.loc[mask, col_orig].dt.weekday
                        df.loc[mask, f'{col_orig}_day_name'] = df.loc[mask, col_orig].dt.day_name()
                        df.loc[mask, f'{col_orig}_month_name'] = df.loc[mask, col_orig].dt.month_name()
                        df.loc[mask, f'{col_orig}_week'] = df.loc[mask, col_orig].dt.isocalendar().week.astype(pd.Int64Dtype())
                        df.loc[mask, f'{col_orig}_quarter'] = df.loc[mask, col_orig].dt.quarter.astype(pd.Int64Dtype())
                        df.loc[mask, f'{col_orig}_is_weekend'] = df.loc[mask, col_orig].dt.weekday >= 5
                        df.loc[mask, f'{col_orig}_date_only'] = df.loc[mask, col_orig].dt.date
                        df.loc[mask, f'{col_orig}_time_only'] = df.loc[mask, col_orig].dt.time
                else: print(f"Cảnh báo: Cột ngày '{col_orig}' không tồn tại trong {key}.")
                for feature in date_feature_cols_to_create:
                    col_name_feature = f'{col_orig}_{feature}'
                    if col_name_feature not in df.columns:
                        df[col_name_feature] = np.nan if feature not in ['day_name', 'month_name'] else "Không rõ"
            number_cols = ['Before Transaction Volume', 'Before Transaction Percentage', 'Registered Transaction Volume', 'Executed Transaction Volume', 'After Transaction Volume', 'After Transaction Percentage']
            for num_col in number_cols:
                if num_col in df.columns: df[num_col] = pd.to_numeric(df[num_col].astype(str).str.replace(',', '', regex=False).str.replace('−', '-', regex=False).replace(['nan', 'NA', 'None', '', '-'], np.nan), errors='coerce')
                else: print(f"Cảnh báo: Cột số '{num_col}' không tồn tại trong {key}.")
            text_cols = ['StockID', 'Transaction Type', 'Executor Name', 'Executor Position', 'Related Person Name', 'Related Person Position', 'Relation']
            for txt_col in text_cols:
                if txt_col in df.columns: df[txt_col] = df[txt_col].astype(str).str.strip().replace('nan', 'Không rõ')
                else: print(f"Cảnh báo: Cột text '{txt_col}' không tồn tại trong {key}.")

        elif key == 'shareholder_meetings':
            date_cols_original = ['Ex-Rights Date', 'Record Date', 'Execution Date']
            for col_orig in date_cols_original:
                if col_orig in df.columns:
                    df[col_orig] = pd.to_datetime(df[col_orig], dayfirst=True, errors='coerce')
                    mask = df[col_orig].notna()
                    if mask.any():
                        df.loc[mask, f'{col_orig}_year'] = df.loc[mask, col_orig].dt.year
                        df.loc[mask, f'{col_orig}_month'] = df.loc[mask, col_orig].dt.month
                        df.loc[mask, f'{col_orig}_day'] = df.loc[mask, col_orig].dt.day
                        df.loc[mask, f'{col_orig}_weekday'] = df.loc[mask, col_orig].dt.weekday
                        df.loc[mask, f'{col_orig}_day_name'] = df.loc[mask, col_orig].dt.day_name()
                        df.loc[mask, f'{col_orig}_month_name'] = df.loc[mask, col_orig].dt.month_name()
                        df.loc[mask, f'{col_orig}_week'] = df.loc[mask, col_orig].dt.isocalendar().week.astype(pd.Int64Dtype())
                        df.loc[mask, f'{col_orig}_quarter'] = df.loc[mask, col_orig].dt.quarter.astype(pd.Int64Dtype())
                        df.loc[mask, f'{col_orig}_is_weekend'] = df.loc[mask, col_orig].dt.weekday >= 5
                        df.loc[mask, f'{col_orig}_date_only'] = df.loc[mask, col_orig].dt.date
                        df.loc[mask, f'{col_orig}_time_only'] = df.loc[mask, col_orig].dt.time
                else: print(f"  Cảnh báo: Cột ngày '{col_orig}' không tồn tại trong {key}.")
                for feature in date_feature_cols_to_create:
                    col_name_feature = f'{col_orig}_{feature}'
                    if col_name_feature not in df.columns:
                        df[col_name_feature] = np.nan if feature not in ['day_name', 'month_name'] else "Không rõ"
            text_cols = ['StockID', 'Exchange', 'Event Type']
            for txt_col in text_cols:
                if txt_col in df.columns: df[txt_col] = df[txt_col].astype(str).str.strip().replace('nan', 'Không rõ')
                else: print(f"Cảnh báo: Cột text '{txt_col}' không tồn tại trong {key}.")
        else:
            pass
        news[key] = df
    return news

# Quy trình RAG
# Bước 1: Tải dữ liệu
def load_raw_data(file_config):
    raw_data_dict = {}
    for key, config in file_config.items():
        file_path = config['path']
        print(f"Đang tải file: {file_path} (key: {key})...")
        try:
            if file_path.endswith('.xlsx'): df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'): df = pd.read_csv(file_path)
            else: print(f"Cảnh báo: Định dạng file không được hỗ trợ: {file_path}. Bỏ qua."); raw_data_dict[key] = None; continue
            raw_data_dict[key] = df
            print(f"Đã tải thành công {len(df)} dòng từ {file_path}.")
        except FileNotFoundError: print(f"Lỗi: Không tìm thấy file {file_path}. Đánh dấu là None."); raw_data_dict[key] = None
        except Exception as e: print(f"Lỗi khi đọc file {file_path}: {e}. Đánh dấu là None."); raw_data_dict[key] = None
    return {k: v for k, v in raw_data_dict.items() if v is not None}

# Bước 2: Định dạng văn bản cho RAG
def format_value(value, col_name):
    if pd.isna(value):
        if '_year' in col_name or '_month' in col_name or '_day' in col_name or '_weekday' in col_name or '_week' in col_name or '_quarter' in col_name: return "không xác định"
        if 'Date' in col_name or 'date' in col_name: return "không có ngày"
        if 'Volume' in col_name or 'Ratio' in col_name or 'Dividend' in col_name or 'Percentage' in col_name: return "không công bố"
        if '_is_weekend' in col_name: return "không xác định"
        if col_name == 'StockID' and str(value) == 'Unknown': return "Không rõ mã"
        return "không có thông tin"
    if isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
        try: return value.strftime('%d/%m/%Y')
        except ValueError: return "ngày không hợp lệ"
    if '_day_name' in col_name or '_month_name' in col_name: return str(value)
    if '_is_weekend' in col_name: return "Cuối tuần" if value else "Ngày thường"
    if '_date_only' in col_name :
        try: return value.strftime('%d/%m/%Y')
        except AttributeError: return "ngày không hợp lệ"
    if '_time_only' in col_name:
        try: return value.strftime('%H:%M:%S')
        except AttributeError: return "giờ không hợp lệ"
    if isinstance(value, (np.floating, float)):
        if value == int(value): return "{:,.0f}".format(value)
        else: return "{:,.2f}".format(value)
    if isinstance(value, (np.integer, int)): return "{:,.0f}".format(value)
    return str(value).strip()

def format_cleaned_data_to_text(cleaned_data_dict, file_config):
    all_text_data = []
    all_metadata = []
    for key, df in cleaned_data_dict.items():
        if not isinstance(df, pd.DataFrame): continue
        if key not in file_config: continue
        config = file_config[key]
        required_cols = config['cols']
        text_format = config['text_format']
        print(f"Đang định dạng văn bản RAG từ key: {key} ({len(df)} dòng)...")
        missing_cols_in_df = [col for col in required_cols if col not in df.columns]
        if missing_cols_in_df: print(f"  Cảnh báo RAG format: Thiếu các cột {missing_cols_in_df} trong DataFrame của key '{key}'.")
        for index, row in df.iterrows():
            row_data = {}
            for col in required_cols: row_data[col] = format_value(row.get(col, np.nan), col)
            formatted_text = ""
            try:
                if key == 'dividends':
                    div_val = row_data.get('Dividend (VND)', 'không công bố'); bonus_val = row_data.get('Bonus Ratio', 'không công bố')
                    row_data['Dividend_Info'] = f"Cổ tức tiền mặt: {div_val} VND/CP. " if div_val != "không công bố" else ""
                    row_data['Bonus_Info'] = f"Cổ phiếu thưởng tỷ lệ: {bonus_val}. " if bonus_val != "không công bố" else ""
                elif key == 'internal_transactions':
                    rel_person = row_data.get('Related Person Name', 'không có thông tin'); relation = row_data.get('Relation', 'không có thông tin')
                    row_data['Related_Person_Info'] = f"Người liên quan: {rel_person} ({relation}). " if rel_person not in ["không có thông tin", "Không rõ"] else ""
                formatted_text = text_format.format(**row_data)
                all_text_data.append(formatted_text)
                all_metadata.append({'source_key': key, 'row_index': index})
            except KeyError as e: print(f"  Lỗi định dạng RAG (KeyError) hàng {index} key '{key}'. Thiếu key: {e}.")
            except Exception as e_fmt: print(f"  Lỗi định dạng RAG (khác) hàng {index} key '{key}': {e_fmt}.")
        print(f"Hoàn tất định dạng văn bản RAG từ key: {key}.")
    print(f"\nTổng cộng đã định dạng được {len(all_text_data)} mẩu dữ liệu văn bản cho RAG.")
    return all_text_data, all_metadata

# Bước 3: Embeddings
def create_embeddings(text_list, model):
    if not text_list: return None
    start_embed_time = time.time()
    try:
        embeddings = model.encode(text_list, show_progress_bar=True, convert_to_tensor=False)
        end_embed_time = time.time()
        print(f"Đã tạo embeddings thành công trong {end_embed_time - start_embed_time:.2f} giây.")
        print(f"Kích thước embeddings: {embeddings.shape}")
        return embeddings
    except Exception as e: print(f"Lỗi nghiêm trọng trong quá trình tạo embeddings: {e}"); return None

# Bước 4: Truy xuất ngữ cảnh
def retrieve_context_embeddings(query, document_embeddings, all_texts, model, top_n=5):
    if document_embeddings is None or len(document_embeddings) == 0 or not all_texts: return ""
    if len(document_embeddings) != len(all_texts): return ""
    try:
        query_embedding = model.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, document_embeddings).flatten()
        most_similar_indices = similarities.argsort()[::-1][:top_n]
        retrieved_docs = [all_texts[i] for i in most_similar_indices]
        return "\n---\n".join(retrieved_docs)
    except Exception as e: print(f"Lỗi trong quá trình truy xuất context: {e}"); return ""

# Bước 5: Sinh văn bản
def generate_with_ollama(prompt, client, model_name):
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05,
            stream=False,
        )
        if hasattr(response, 'choices') and response.choices:
            result = response.choices[0].message.content.strip()
        else: result = "[Lỗi xử lý response từ Ollama]"
        end_time = time.time()
        print(f"Thời gian tạo sinh của Ollama: {end_time - start_time:.2f} giây")
        return result
    except Exception as e:
        end_time = time.time()
        print(f"Lỗi khi gọi Ollama API ({model_name}) sau {end_time - start_time:.2f} giây: {e}")
        return f"Lỗi khi tạo câu trả lời bằng {model_name} qua Ollama."

# Bước 6: Quản lý quy trình
def answer_query_with_rag(query, document_embeddings, all_texts, client, model_name, embedding_model_obj):
    print(f"Truy vấn RAG: \"{query}\"")
    print("Đang truy xuất ngữ cảnh RAG...")
    start_retrieve_time = time.time()
    context = retrieve_context_embeddings(query, document_embeddings, all_texts, embedding_model_obj, top_n=10)
    end_retrieve_time = time.time()
    print(f"Thời gian truy xuất RAG: {end_retrieve_time - start_retrieve_time:.2f} giây")
    final_prompt = ""
    if not context or context.strip() == "":
        print("Không tìm thấy ngữ cảnh RAG liên quan.")
        final_prompt = f"""Câu hỏi: {query}\nKhông tìm thấy thông tin liên quan trong các tài liệu được cung cấp về FPT và CMG. Dựa trên kiến thức chung của bạn, hãy cố gắng trả lời câu hỏi. Nếu bạn cũng không biết, hãy nói rõ điều đó."""
        print("Gửi prompt (không có context RAG) đến LLM...")
    else:
        print(f"\nNgữ cảnh RAG được truy xuất:\n------\n{context}\n------")
        final_prompt = f"""Bạn là một trợ lý AI chuyên phân tích dữ liệu tài chính.
                        Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và TUYỆT ĐỐI CHỈ DỰA VÀO DỮ LIỆU NGỮ CẢNH được cung cấp dưới đây.
                        Bạn không được suy diễn, không được sử dụng kiến thức bên ngoài, không được bịa đặt thông tin.
                        KHÔNG ĐƯỢC SUY DIỄN, KHÔNG ĐƯỢC SỬ DỤNG KIẾN THỨC BÊN NGOÀI, KHÔNG ĐƯỢC BỊA ĐẶT THÔNG TIN.
                        Nếu thông tin không có trong DỮ LIỆU NGỮ CẢNH, KHÔNG CẦN cung cấp thông tin thêm
                        Nếu thông tin không có trong DỮ LIỆU NGỮ CẢNH, hãy trả lời: "Thông tin này không có trong dữ liệu được cung cấp."

                        DỮ LIỆU NGỮ CẢNH (Trích xuất từ các file đã xử lý):
                        ------
                        {context}
                        ------
                        CÂU HỎI: {query}
                        TRẢ LỜI (chỉ dựa vào DỮ LIỆU NGỮ CẢNH ở trên):"""

        print("\nGửi prompt RAG hoàn chỉnh đến LLM...")
    print(f"Đang yêu cầu {model_name} tạo câu trả lời RAG...")
    answer = generate_with_ollama(final_prompt, client, model_name)
    print(f"\nCâu trả lời từ hệ thống RAG:")
    print(answer if answer else "[Không có câu trả lời RAG được tạo ra hoặc có lỗi]")
    return answer
# Hàm trích xuất các features từ tin tức
def extract_all_features(ticker, cleaned_data_ticker, ollama_client_param, ollama_model_name_param):
    df_features = None
    all_dates = []
    date_cols_to_check = {
        'cafef_news': ['date'],
        'dividends': ['Ex-Dividend Date', 'Record Date', 'Execution Date'],
        'internal_transactions': ['Registered From Date', 'Registered To Date', 'Executed From Date', 'Executed To Date'],
        'shareholder_meetings': ['Ex-Rights Date', 'Record Date', 'Execution Date']
    }

    # Thu thập tất cả ngày từ các bảng dữ liệu
    for key, df_clean in cleaned_data_ticker.items():
        if df_clean is not None and isinstance(df_clean, pd.DataFrame) and not df_clean.empty:
            for col in date_cols_to_check.get(key, []):
                if col in df_clean.columns:
                    dates_series = pd.to_datetime(df_clean[col], errors='coerce')
                    all_dates.extend(dates_series.dropna().tolist())

    if not all_dates:
        print(f"Lỗi: Không thể tìm thấy ngày phù hợp cho cổ phiếu {ticker}.")
        return pd.DataFrame()

    min_date, max_date = min(all_dates), max(all_dates)
    print(f"Khoảng thời gian dữ liệu cổ phiếu {ticker}: {min_date.date()} đến {max_date.date()}")
    date_range_index = pd.date_range(start=min_date, end=max_date, freq='D')
    df_features = pd.DataFrame(index=date_range_index)

    # --- Xử lý tin tức ---

    # Lấy dataframe tin tức cổ phiếu hiện tại
    df_news_ticker = None
    if 'cafef_news' in cleaned_data_ticker and cleaned_data_ticker['cafef_news'] is not None:
        df_news_ticker = cleaned_data_ticker['cafef_news'].copy()

    # Nếu không có dữ liệu tin tức, tạo dataframe rỗng
    if df_news_ticker is None or df_news_ticker.empty:
        df_news_ticker = pd.DataFrame(columns=['date', 'title', 'summary', 'ticker'])

    # Tách tin tức chung (không có ticker hoặc ticker rỗng)
    # Giả sử cột ticker có tên 'ticker' hoặc 'stock_ticker', nếu không có thì bạn cần điều chỉnh tên cột phù hợp
    ticker_col_candidates = ['ticker', 'stock_ticker', 'StockID']
    ticker_col = None
    for col_candidate in ticker_col_candidates:
        if col_candidate in df_news_ticker.columns:
            ticker_col = col_candidate
            break

    if ticker_col is not None:
        df_news_common = df_news_ticker[
            df_news_ticker[ticker_col].isnull() | (df_news_ticker[ticker_col].astype(str).str.strip() == '')
        ].copy()
        df_news_specific = df_news_ticker[
            ~(df_news_ticker[ticker_col].isnull() | (df_news_ticker[ticker_col].astype(str).str.strip() == ''))
        ].copy()
        # Lọc tin tức riêng cổ phiếu hiện tại (giả sử ticker viết hoa chuẩn)
        df_news_specific = df_news_specific[
            df_news_specific[ticker_col].astype(str).str.upper() == ticker.upper()
        ].copy()
    else:
        # Nếu không có cột ticker thì coi toàn bộ tin tức là chung
        df_news_common = df_news_ticker.copy()
        df_news_specific = pd.DataFrame(columns=df_news_ticker.columns)

    # Gộp tin tức chung và tin tức riêng cổ phiếu hiện tại
    df_news_combined = pd.concat([df_news_common, df_news_specific], ignore_index=True)

    # Nếu có dữ liệu ngày
    if 'date' in df_news_combined.columns and not df_news_combined['date'].isnull().all():
        df_news_combined = df_news_combined.dropna(subset=['date']).copy()
        df_news_combined.rename(columns={'date': 'date_dt'}, inplace=True)

        # Tạo các feature độ dài title và summary
        df_news_combined['title_length'] = df_news_combined['title'].astype(str).str.len()
        df_news_combined['summary_length'] = df_news_combined['summary'].astype(str).str.len()

        # Tính các feature cơ bản
        daily_news_features = df_news_combined.groupby(df_news_combined['date_dt'].dt.date).agg(
            news_count=('title', 'size'),
            news_avg_title_len=('title_length', 'mean'),
            news_avg_summary_len=('summary_length', 'mean')
        )
        daily_news_features.index = pd.to_datetime(daily_news_features.index)
        df_features = df_features.join(daily_news_features)

        # Phân tích sentiment sử dụng LLM
        print(f"Ước tính sentiment của tin tức cho cổ phiếu {ticker} sử dụng mô hình ({ollama_model_name_param})...")
        daily_sentiments = pd.Series(index=date_range_index, dtype=float)

        sentiment_prompt_template = (
            "Analyze the overall sentiment expressed in the following financial news headlines and summaries for {ticker} published on {date}. "
            "Classify the potential impact on the stock price ONLY as POSITIVE, NEGATIVE, or NEUTRAL based SOLELY on the provided text snippets.\n\n"
            "News Snippets for {date}:\n------\n{context}\n------\n\n"
            "Respond ONLY with one word: POSITIVE, NEGATIVE, or NEUTRAL."
        )

        for current_date in tqdm(date_range_index, desc="Phân tích sentiment LLM"):
            news_on_date = df_news_combined[df_news_combined['date_dt'].dt.date == current_date.date()]
            if news_on_date.empty:
                daily_sentiments[current_date] = 0.0
                continue

            context_text = "\n---\n".join(
                f"Title: {row['title']}\nSummary: {row['summary']}" for _, row in news_on_date.head(5).iterrows()
            )[:3000]

            prompt = sentiment_prompt_template.format(
                ticker=ticker,
                date=current_date.strftime('%Y-%m-%d'),
                context=context_text
            )

            llm_response = generate_with_ollama(prompt, ollama_client_param, ollama_model_name_param)
            parsed_sentiment = 0.0
            if isinstance(llm_response, str):
                sentiment_label = llm_response.strip().upper()
                if "POSITIVE" in sentiment_label:
                    parsed_sentiment = 1.0
                elif "NEGATIVE" in sentiment_label:
                    parsed_sentiment = -1.0
                else:
                    parsed_sentiment = 0.0

            daily_sentiments[current_date] = parsed_sentiment

        df_features['news_llm_sentiment'] = daily_sentiments

    # --- Xử lý dividend ---
    if 'dividends' in cleaned_data_ticker and cleaned_data_ticker['dividends'] is not None:
        df_div_ticker = cleaned_data_ticker['dividends'].copy()
        align_date_col = 'Ex-Dividend Date'
        if align_date_col in df_div_ticker.columns and not df_div_ticker[align_date_col].isnull().all():
            df_div_ticker = df_div_ticker.dropna(subset=[align_date_col]).copy()
            df_div_ticker.rename(columns={align_date_col: 'date_dt'}, inplace=True)
            if not df_div_ticker.empty:
                daily_div_features = df_div_ticker.groupby(df_div_ticker['date_dt'].dt.date).agg(
                    div_event_count=('StockID', 'size'),
                    div_total_vnd=('Dividend (VND)', 'sum'),
                    div_total_bonus_ratio=('Bonus Ratio', 'sum')
                )
                daily_div_features.index = pd.to_datetime(daily_div_features.index)
                df_features = df_features.join(daily_div_features)

    # --- Xử lý giao dịch nội bộ ---
    if 'internal_transactions' in cleaned_data_ticker and cleaned_data_ticker['internal_transactions'] is not None:
        df_it_ticker = cleaned_data_ticker['internal_transactions'].copy()
        align_date_col = 'Executed From Date'
        if align_date_col in df_it_ticker.columns and not df_it_ticker[align_date_col].isnull().all():
            df_it_ticker = df_it_ticker.dropna(subset=[align_date_col]).copy()
            df_it_ticker.rename(columns={align_date_col: 'date_dt'}, inplace=True)
            if not df_it_ticker.empty:
                daily_it_features = df_it_ticker.groupby(df_it_ticker['date_dt'].dt.date).agg(
                    it_event_count=('StockID', 'size'),
                    it_total_executed_vol=('Executed Transaction Volume', 'sum')
                )
                daily_it_features.index = pd.to_datetime(daily_it_features.index)
                df_features = df_features.join(daily_it_features)

    # --- Xử lý lịch họp cổ đông ---
    if 'shareholder_meetings' in cleaned_data_ticker and cleaned_data_ticker['shareholder_meetings'] is not None:
        df_meet_ticker = cleaned_data_ticker['shareholder_meetings'].copy()
        align_date_col = 'Ex-Rights Date'
        if align_date_col in df_meet_ticker.columns and not df_meet_ticker[align_date_col].isnull().all():
            df_meet_ticker = df_meet_ticker.dropna(subset=[align_date_col]).copy()
            df_meet_ticker.rename(columns={align_date_col: 'date_dt'}, inplace=True)
            if not df_meet_ticker.empty:
                daily_meet_features = df_meet_ticker.groupby(df_meet_ticker['date_dt'].dt.date).agg(
                    meeting_event_count=('StockID', 'size')
                )
                daily_meet_features.index = pd.to_datetime(daily_meet_features.index)
                df_features = df_features.join(daily_meet_features)

    # Điền giá trị thiếu theo trước và sau
    df_features = df_features.bfill().ffill()

    return df_features



# Hàm chính
# Hàm 1: Features extraction
def prepare_feature_data(tickers, file_config, ollama_client_param, ollama_model_name_param):
    """
    Tải, làm sạch (riêng biệt), và trích xuất features (riêng biệt) cho các ticker.
    Cũng trả về bộ dữ liệu đã làm sạch tổng hợp dùng cho RAG.
    """
    feature_dataframes = {}
    cleaned_data_all_for_rag = {}
    raw_data = None
    try:
        # 1. Tải dữ liệu thô
        raw_data = load_raw_data(file_config)
        # if not raw_data: raise RuntimeError("Không thể tải dữ liệu gốc.")
        # # 1.1 Xử lý StockID cho cafef_news thô trước khi lọc
        # if 'cafef_news' in raw_data and raw_data['cafef_news'] is not None:
        #     df_news_raw = raw_data['cafef_news']
        #     if 'StockID' not in df_news_raw.columns and 'title' in df_news_raw.columns:
        #         df_news_raw['StockID'] = df_news_raw['title'].apply(lambda x: 'FPT' if 'FPT' in str(x).upper() else ('CMG' if 'CMG' in str(x).upper() else 'Unknown'))
        # else: print("Warning: Raw cafef_news data not loaded or available.")
        # # 2. Lọc và làm sạch riêng biệt cho từng ticker => Dùng cho extract_all_features
        # for ticker in tickers:
        #     raw_data_ticker = {}
        #     cleaned_data_ticker = {}
        #     # 2.1 Lọc dữ liệu thô cho ticker
        #     print(f"Filtering raw data for {ticker}...")
        #     for key, df_raw in raw_data.items():
        #         if df_raw is None: continue
        #         if 'StockID' in df_raw.columns:
        #             filtered_df = df_raw[df_raw['StockID'] == ticker].copy()
        #             if not filtered_df.empty: raw_data_ticker[key] = filtered_df
        #     # 2.2 Làm sạch dữ liệu đã lọc của ticker (GỌI HÀM clean_data gốc)
        #     print(f"Cleaning filtered data for {ticker}...")
        #     cleaned_data_ticker = clean_data(raw_data_ticker) # Dùng hàm clean_data gốc
        #     # 2.3 Trích xuất features từ dữ liệu sạch của ticker
        #     if cleaned_data_ticker:
        #         df_ticker_features = extract_all_features(
        #             ticker, cleaned_data_ticker, ollama_client_param, ollama_model_name_param
        #         )
        #         feature_dataframes[ticker] = df_ticker_features
        #     else:
        #         print(f"Không có dữ liệu cho cổ phiếu {ticker}.")
        #         feature_dataframes[ticker] = pd.DataFrame()
        #     # Lưu file ra 2 file .csv
        #     if not df_ticker_features.empty:
        #         df_ticker_features.to_csv(f"DATA EXPLORER CONTEST/Data/{ticker}_news_features.csv", index=True)
        #         print("Lưu thành công")
        # 3. Làm sạch bộ dữ liệu tổng hợp (dùng cho RAG - GỌI HÀM clean_data gốc)
        cleaned_data_all_for_rag = clean_data(raw_data)
        # Trả về cleaned_data tổng hợp và dict các feature dataframes riêng lẻ
        return cleaned_data_all_for_rag #, feature_dataframes
    except RuntimeError as e: print(f"\nLỗi trong quá trình chuẩn bị feature data: {e}"); return None, None
    except Exception as e_prep: print(f"\nLỗi không xác định trong quá trình chuẩn bị feature data: {e_prep}"); return None, None

# Hàm 2: Chạy hệ thống RAG
def run_rag_system(cleaned_data_for_rag, file_config_for_rag, embedding_model_obj, ollama_client_obj, ollama_model_name_qa):
    all_texts_for_rag = None; document_embeddings_for_rag = None; rag_ready = False
    try:
        all_texts_for_rag, _ = format_cleaned_data_to_text(cleaned_data_for_rag, file_config_for_rag)
        if not all_texts_for_rag: raise RuntimeError("Không thể định dạng dữ liệu thành văn bản cho RAG.")
        document_embeddings_for_rag = create_embeddings(all_texts_for_rag, embedding_model_obj)
        if document_embeddings_for_rag is None: raise RuntimeError("Tạo embeddings cho RAG thất bại.")
        rag_ready = True
    except RuntimeError as e: print(f"Lỗi chuẩn bị dữ liệu RAG: {e}")
    except Exception as e_rag_prep: print(f"Lỗi không xác định khi chuẩn bị RAG: {e_rag_prep}")
    if rag_ready:
        print("Chào mừng bạn đến với hệ thống hỏi đáp về cổ phiếu FPT và CMG")
        print(f"Đã xử lý {len(all_texts_for_rag)} mẩu dữ liệu cho RAG Q&A.")
        print(f"Sẵn sàng trả lời câu hỏi sử dụng LLM: {ollama_model_name_qa}")
        print("Nhập câu hỏi của bạn bên dưới. Gõ 'quit' hoặc 'exit' để thoát.")
        print("--------------------------------------------------------")
        while True:
            user_query = input("\n>>> Nhập câu hỏi của bạn: ")
            if user_query.lower().strip() in ['quit', 'exit']: print("\nTạm biệt!"); break
            if not user_query.strip(): print("Vui lòng nhập câu hỏi."); continue
            answer_query_with_rag(user_query, document_embeddings_for_rag, all_texts_for_rag, ollama_client_obj, ollama_model_name_qa, embedding_model_obj)
    else:
        print("\nKhông thể khởi động hệ thống hỏi đáp RAG do lỗi chuẩn bị dữ liệu.")

# --- Cấu hình File ---
folder_path = 'DATA EXPLORER CONTEST/News/'
FILE_CONFIG = {
    'cafef_news': {
        'path': os.path.join(folder_path, 'CafeF_News_FPT_CMG.xlsx'),
        'cols': [
            'title', 'date', 'summary', 'StockID',
            'date_day_name', 'date_month_name', 'date_year', 'date_quarter'
        ],
        'text_format': "Tin tức CafeF ({StockID}) ngày {date} ({date_day_name}, Tháng {date_month_name}, Năm {date_year}, Quý {date_quarter}): Tiêu đề '{title}'. Tóm tắt: {summary}"
    },
    'dividends': {
        'path': os.path.join(folder_path, 'news_dividend_issue (FPT_CMG)_processed.csv'),
        'cols': [
            'StockID', 'Ex-Dividend Date', 'Record Date', 'Execution Date',
            'Event Content', 'Event Type', 'Dividend (VND)', 'Bonus Ratio',
            'Ex-Dividend Date_day_name', 'Ex-Dividend Date_month_name', 'Ex-Dividend Date_year', 'Ex-Dividend Date_quarter',
            'Record Date_day_name', 'Record Date_month_name', 'Record Date_year', 'Record Date_quarter',
            'Execution Date_day_name', 'Execution Date_month_name', 'Execution Date_year', 'Execution Date_quarter',
        ],
        'text_format': "Sự kiện cổ tức {StockID} ({Event Type}): Ngày GDKHQ {Ex-Dividend Date} ({Ex-Dividend Date_day_name}, Tháng {Ex-Dividend Date_month_name}/{Ex-Dividend Date_year}, Quý {Ex-Dividend Date_quarter}), Ngày ĐKCC {Record Date} ({Record Date_day_name}, Tháng {Record Date_month_name}/{Record Date_year}, Quý {Record Date_quarter}). {Dividend_Info}{Bonus_Info}Nội dung: {Event Content}. Ngày thực hiện dự kiến: {Execution Date} ({Execution Date_day_name}, Tháng {Execution Date_month_name}/{Execution Date_year}, Quý {Execution Date_quarter})."
    },
    'internal_transactions': {
        'path': os.path.join(folder_path, 'news_internal_transactions (FPT_CMG)_processed.csv'),
        'cols': [
            'StockID', 'Transaction Type', 'Executor Name', 'Executor Position',
            'Related Person Name', 'Relation',
            'Before Transaction Volume', 'Registered Transaction Volume', 'Executed Transaction Volume', 'After Transaction Volume',
            'Registered From Date', 'Registered To Date', 'Executed From Date', 'Executed To Date',
            'Registered From Date_day_name', 'Registered From Date_month_name', 'Registered From Date_year', 'Registered From Date_quarter',
            'Registered To Date_day_name', 'Registered To Date_month_name', 'Registered To Date_year', 'Registered To Date_quarter',
            'Executed From Date_day_name', 'Executed From Date_month_name', 'Executed From Date_year', 'Executed From Date_quarter',
            'Executed To Date_day_name', 'Executed To Date_month_name', 'Executed To Date_year', 'Executed To Date_quarter',
        ],
        'text_format': "Giao dịch nội bộ {StockID}: {Executor Name} ({Executor Position}) {Transaction Type}. {Related_Person_Info}Đăng ký GD: {Registered Transaction Volume} CP (Từ {Registered From Date} ({Registered From Date_day_name}, Tháng {Registered From Date_month_name}/{Registered From Date_year}, Quý {Registered From Date_quarter}) đến {Registered To Date} ({Registered To Date_day_name}, Tháng {Registered To Date_month_name}/{Registered To Date_year}, Quý {Registered To Date_quarter})). Thực hiện GD: {Executed Transaction Volume} CP (Từ {Executed From Date} ({Executed From Date_day_name}, Tháng {Executed From Date_month_name}/{Executed From Date_year}, Quý {Executed From Date_quarter}) đến {Executed To Date} ({Executed To Date_day_name}, Tháng {Executed To Date_month_name}/{Executed To Date_year}, Quý {Executed To Date_quarter})). Số lượng CP trước GD: {Before Transaction Volume}, sau GD: {After Transaction Volume}."
    },
    'shareholder_meetings': {
        'path': os.path.join(folder_path, 'news_shareholder_meeting (FPT_CMG)_processed.csv'),
        'cols': [
            'StockID', 'Ex-Rights Date', 'Record Date', 'Execution Date', 'Event Type',
            'Ex-Rights Date_day_name', 'Ex-Rights Date_month_name', 'Ex-Rights Date_year', 'Ex-Rights Date_quarter',
            'Record Date_day_name', 'Record Date_month_name', 'Record Date_year', 'Record Date_quarter',
            'Execution Date_day_name', 'Execution Date_month_name', 'Execution Date_year', 'Execution Date_quarter',
        ],
        'text_format': "Sự kiện cổ đông {StockID}: {Event Type}. Ngày GDKHQ {Ex-Rights Date} ({Ex-Rights Date_day_name}, Tháng {Ex-Rights Date_month_name}/{Ex-Rights Date_year}, Quý {Ex-Rights Date_quarter}), Ngày ĐKCC {Record Date} ({Record Date_day_name}, Tháng {Record Date_month_name}/{Record Date_year}, Quý {Record Date_quarter}), Ngày thực hiện {Execution Date} ({Execution Date_day_name}, Tháng {Execution Date_month_name}/{Execution Date_year}, Quý {Execution Date_quarter})."
    }
}

# Cấu hình Ollama và Model Embeddings
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"Sử dụng embedding model: {EMBEDDING_MODEL_NAME}")
print(f"Sử dụng LLM Ollama model: {OLLAMA_MODEL}")

# Main
if __name__ == "__main__":
    embedding_model_main = None; ollama_client_main = None; initialization_ok = False
    try:
        print(f"Đang tải embedding model: {EMBEDDING_MODEL_NAME}..."); embedding_model_main = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model đã tải xong.")
        ollama_client_main = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
        print(f"Đã kết nối Ollama tại {OLLAMA_BASE_URL}")
        initialization_ok = True
    except Exception as e: print(f"Lỗi khởi tạo chung: {e}"); exit()
    print("Khởi tạo chung hoàn tất")
    # Chạy hàm Features Extraction
    tickers_to_process = ['FPT', 'CMG']
    # Hàm này trả về cleaned_data tổng hợp và dict các feature df riêng lẻ
    cleaned_data_for_rag_main, feature_dataframes_result = prepare_feature_data(
        tickers=tickers_to_process,
        file_config=FILE_CONFIG,
        ollama_client_param=ollama_client_main, 
        ollama_model_name_param=OLLAMA_MODEL  
    )

    # --- Chạy Hàm 2: Hệ thống Q&A RAG (nếu Hàm 1 thành công) ---
    if cleaned_data_for_rag_main is not None and feature_dataframes_result is not None:
        print("\n--- CHẠY BƯỚC 2: KHỞI ĐỘNG HỆ THỐNG Q&A RAG ---")
        run_rag_system(
            cleaned_data_for_rag=cleaned_data_for_rag_main,
            file_config_for_rag=FILE_CONFIG,
            embedding_model_obj=embedding_model_main,    
            ollama_client_obj=ollama_client_main,         
            ollama_model_name_qa=OLLAMA_MODEL             
        )
    else:
        print("\nLỗi nghiêm trọng: Không thể chuẩn bị dữ liệu. Hệ thống Q&A sẽ không chạy.")

    print("\nKết thúc chương trình")