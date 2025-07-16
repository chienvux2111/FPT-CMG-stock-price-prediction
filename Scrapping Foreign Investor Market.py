import datetime
import time
import random
import pandas as pd
import requests
import json # Thêm thư viện json để xử lý lỗi decode

# =======================================================
# Hằng số Cấu hình
# =======================================================
CAFEF_API_URL = "https://cafef.vn/du-lieu/Ajax/PageNew/DataGDNN/GDNuocNgoai.ashx"
THOI_GIAN_CHO_YEU_CAU = 30  # giây
USER_AGENT_MAC_DINH = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# =======================================================
# Các Hàm Hỗ trợ
# =======================================================

def lay_ngay_giao_dich(ngay_bat_dau, ngay_ket_thuc):
    """Tạo danh sách các ngày giao dịch (Thứ 2 - Thứ 6) trong khoảng thời gian."""
    if ngay_bat_dau > ngay_ket_thuc:
        raise ValueError("Ngày bắt đầu không thể sau ngày kết thúc")
    danh_sach_ngay = []
    ngay_hien_tai = ngay_bat_dau
    while ngay_hien_tai <= ngay_ket_thuc:
        # Thứ 2 là 0, Chủ nhật là 6
        if ngay_hien_tai.weekday() < 5:
            danh_sach_ngay.append(ngay_hien_tai)
        ngay_hien_tai += datetime.timedelta(days=1)
    return danh_sach_ngay

def chuyen_ngay_sang_dinh_dang_api(doi_tuong_ngay):
    """Chuyển đổi đối tượng datetime.date sang định dạng '/Date(timestamp_ms)/'."""
    # Chuyển date thành datetime vào đầu ngày
    dt_obj = datetime.datetime.combine(doi_tuong_ngay, datetime.time.min)
    # Tính timestamp (giây) và chuyển sang millisecond
    moc_thoi_gian_ms = int(time.mktime(dt_obj.timetuple()) * 1000)
    return f"/Date({moc_thoi_gian_ms})/"

def dinh_dang_bytes(so_byte):
    """Định dạng số byte thành KB, MB, GB."""
    if so_byte < 1024:
        return f"{so_byte} Bytes"
    elif so_byte < 1024**2:
        return f"{so_byte / 1024:.2f} KB"
    elif so_byte < 1024**3:
        return f"{so_byte / (1024**2):.2f} MB"
    else:
        return f"{so_byte / (1024**3):.2f} GB"

# =======================================================
# Các Hàm Cào Dữ liệu Cốt lõi
# =======================================================

def cao_du_lieu_ngay_san(session, ten_san, ngay_giao_dich):
    """
    Cào dữ liệu giao dịch khối ngoại chi tiết cho một sàn và một ngày cụ thể.

    Tham số:
        session: Đối tượng requests.Session.
        ten_san: Mã định danh sàn giao dịch (ví dụ: 'HOSE', 'HNX').
        ngay_giao_dich: Đối tượng datetime.date cho ngày giao dịch.

    Trả về:
        Một tuple chứa:
        - pd.DataFrame: DataFrame chứa dữ liệu chi tiết, hoặc DataFrame rỗng nếu lỗi/không có dữ liệu.
        - int: Số byte đã tải về cho yêu cầu này.
    """
    chuoi_ngay_api = chuyen_ngay_sang_dinh_dang_api(ngay_giao_dich)
    # Referer thường dùng định dạng dd/mm/yyyy ngay cả khi tham số API khác
    chuoi_ngay_referer = ngay_giao_dich.strftime("%d/%m/%Y")

    params = {"TradeCenter": ten_san.upper(), "Date": chuoi_ngay_api}
    # Xây dựng URL Referer giống cách trình duyệt có thể làm
    headers = {
        "Referer": f"https://cafef.vn/du-lieu/tracuulichsu2/3/{ten_san.lower()}/{chuoi_ngay_referer}.chn"
        # User-Agent được đặt toàn cục trong session
    }

    try:
        response = session.get(CAFEF_API_URL, params=params, headers=headers, timeout=THOI_GIAN_CHO_YEU_CAU)
        response.raise_for_status()  # Báo lỗi HTTPError nếu phản hồi xấu (4xx hoặc 5xx)
        du_lieu_json = response.json()
        so_byte = len(response.content)

        # Truy cập an toàn vào cấu trúc JSON
        danh_sach_du_lieu = du_lieu_json.get('Data', {}).get('ListDataNN', [])

        if not danh_sach_du_lieu:
            # print(f"Thông tin: Không tìm thấy 'ListDataNN' cho {ten_san} - {ngay_giao_dich}")
            return pd.DataFrame(), so_byte

        df_chi_tiet = pd.DataFrame(danh_sach_du_lieu)
        if not df_chi_tiet.empty:
            # Thêm các cột siêu dữ liệu (metadata)
            df_chi_tiet['TradeDate'] = ngay_giao_dich # Sử dụng đối tượng date gốc
            df_chi_tiet['Exchange'] = ten_san.upper()
        return df_chi_tiet, so_byte

    except requests.exceptions.Timeout:
        print(f"Lỗi: Timeout khi lấy dữ liệu {ten_san} - {ngay_giao_dich}")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi: Yêu cầu thất bại cho {ten_san} - {ngay_giao_dich}: {e}")
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể giải mã JSON cho {ten_san} - {ngay_giao_dich}")
    except Exception as e:
        print(f"Lỗi: Một lỗi không mong muốn đã xảy ra cho {ten_san} - {ngay_giao_dich}: {e}")

    return pd.DataFrame(), 0 # Trả về DataFrame rỗng và 0 byte nếu có lỗi

def cao_du_lieu_khoang_ngay(danh_sach_san, ngay_bat_dau, ngay_ket_thuc, do_tre_toi_thieu=0.6, do_tre_toi_da=3.0):
    """
    Cào dữ liệu giao dịch khối ngoại cho nhiều sàn trong một khoảng ngày.

    Tham số:
        danh_sach_san: Danh sách các mã định danh sàn.
        ngay_bat_dau: Ngày bắt đầu (datetime.date).
        ngay_ket_thuc: Ngày kết thúc (datetime.date).
        do_tre_toi_thieu: Độ trễ tối thiểu giữa các yêu cầu (giây).
        do_tre_toi_da: Độ trễ tối đa giữa các yêu cầu (giây).

    Trả về:
        Một tuple chứa:
        - pd.DataFrame: Một DataFrame duy nhất chứa tất cả dữ liệu chi tiết đã cào.
        - int: Tổng số byte đã tải về.
    """
    cac_ngay_can_cao = lay_ngay_giao_dich(ngay_bat_dau, ngay_ket_thuc)
    if not cac_ngay_can_cao:
        print("Cảnh báo: Không tìm thấy ngày giao dịch nào trong khoảng đã chọn.")
        return pd.DataFrame(), 0

    tat_ca_chi_tiet = []
    tong_so_byte = 0
    tong_so_yeu_cau = len(danh_sach_san) * len(cac_ngay_can_cao)
    dem_yeu_cau = 0

    with requests.Session() as session:
        # Đặt các header chung cho session
        session.headers.update({
            "User-Agent": USER_AGENT_MAC_DINH,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest"
        })

        print(f"Bắt đầu cào dữ liệu cho {len(cac_ngay_can_cao)} ngày và {len(danh_sach_san)} sàn ({tong_so_yeu_cau} yêu cầu)...")

        for ngay_giao_dich in cac_ngay_can_cao:
            for ten_san in danh_sach_san:
                dem_yeu_cau += 1
                print(f"  Đang lấy [{dem_yeu_cau}/{tong_so_yeu_cau}]: {ten_san} - {ngay_giao_dich.strftime('%Y-%m-%d')}")

                df_chi_tiet, so_byte = cao_du_lieu_ngay_san(session, ten_san, ngay_giao_dich)

                if not df_chi_tiet.empty:
                    tat_ca_chi_tiet.append(df_chi_tiet)
                tong_so_byte += so_byte

                # Chỉ áp dụng độ trễ nếu không phải là yêu cầu cuối cùng
                if dem_yeu_cau < tong_so_yeu_cau:
                    time.sleep(random.uniform(do_tre_toi_thieu, do_tre_toi_da))

    print("\nĐang ghép nối kết quả...")
    if not tat_ca_chi_tiet:
        print("Cảnh báo: Không có dữ liệu nào được cào thành công.")
        return pd.DataFrame(), tong_so_byte

    df_cuoi_cung = pd.concat(tat_ca_chi_tiet, ignore_index=True)
    print("Ghép nối hoàn tất.")
    return df_cuoi_cung, tong_so_byte

# =======================================================
# Khối Thực thi Chính
# =======================================================

if __name__ == "__main__":
    # --- Cấu hình ---
    CAC_SAN_CAN_CAO = ['HOSE', 'HNX', 'UPCoM'] # Mã định danh chuẩn
    NGAY_BAT_DAU = datetime.date(2024, 3, 12)
    NGAY_KET_THUC = datetime.date(2025, 3, 12)
    DO_TRE_TOI_THIEU_GIAY = 0.7 # Tăng nhẹ độ trễ tối thiểu
    DO_TRE_TOI_DA_GIAY = 2.5   # Giảm nhẹ độ trễ tối đa
    LUU_RA_CSV = True
    TEN_Tệp_XUAT = f"cafef_foreign_details_{NGAY_BAT_DAU.strftime('%Y%m%d')}_to_{NGAY_KET_THUC.strftime('%Y%m%d')}.csv"
    IN_DAU = 5
    IN_CUOI = 5

    print("=" * 50)
    print(" Công cụ Cào Dữ liệu Giao dịch Khối Ngoại CafeF")
    print("=" * 50)
    print(f"Các sàn: {', '.join(CAC_SAN_CAN_CAO)}")
    print(f"Khoảng ngày: {NGAY_BAT_DAU.strftime('%Y-%m-%d')} đến {NGAY_KET_THUC.strftime('%Y-%m-%d')}")
    print(f"Tệp xuất: {TEN_Tệp_XUAT if LUU_RA_CSV else 'Không lưu'}")
    print("-" * 50)

    thoi_gian_bat_dau = time.time()

    # --- Chạy quá trình cào dữ liệu ---
    du_lieu_chi_tiet, tong_byte_da_tai = cao_du_lieu_khoang_ngay(
        danh_sach_san=CAC_SAN_CAN_CAO,
        ngay_bat_dau=NGAY_BAT_DAU,
        ngay_ket_thuc=NGAY_KET_THUC,
        do_tre_toi_thieu=DO_TRE_TOI_THIEU_GIAY,
        do_tre_toi_da=DO_TRE_TOI_DA_GIAY
    )

    thoi_gian_ket_thuc = time.time()
    thoi_luong = thoi_gian_ket_thuc - thoi_gian_bat_dau

    print("-" * 50)
    print("Quá trình Cào Dữ liệu Hoàn tất")
    print(f"Thời lượng: {thoi_luong:.2f} giây")
    print(f"Tổng dung lượng đã tải: {dinh_dang_bytes(tong_byte_da_tai)}")
    print(f"Tổng số dòng dữ liệu đã cào: {len(du_lieu_chi_tiet)}")

    # --- Xử lý và Lưu Kết quả ---
    if not du_lieu_chi_tiet.empty:
        if IN_DAU > 0:
            print("\n--- Phần đầu Dữ liệu ---")
            print(du_lieu_chi_tiet.head(IN_DAU))

        if IN_CUOI > 0:
            print("\n--- Phần cuối Dữ liệu ---")
            print(du_lieu_chi_tiet.tail(IN_CUOI))

        if LUU_RA_CSV:
            try:
                du_lieu_chi_tiet.to_csv(TEN_Tệp_XUAT, index=False, encoding='utf-8-sig')
                print(f"\nĐã lưu dữ liệu thành công vào {TEN_Tệp_XUAT}")
            except Exception as e:
                print(f"\nLỗi khi lưu dữ liệu ra CSV: {e}")
    else:
        print("\nKhông có dữ liệu nào được cào để hiển thị hoặc lưu.")

    print("=" * 50)