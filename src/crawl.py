from pyinaturalist import get_observations
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import requests, os, time
from tqdm import tqdm

# ĐỊNH NGHĨA TỪ ĐIỂN CHỨA ĐẦY ĐỦ 8 LOÀI HOA LAN
flowers = {
    "paphiopedilum_rothschildianum": {
        "taxon": "Paphiopedilum rothschildianum",
        "keywords": [
            "Paphiopedilum rothschildianum slipper orchid",
            "Paphiopedilum rothschildianum flower",
            "rothschildianum orchid bloom",
            "king of slipper orchid",
            "Paphiopedilum rothschildianum close up",
            "Paph rothschildianum flower"
        ],
    },
    "cattleya_labiata": {
        "taxon": "Cattleya labiata",
        "keywords": [
            "Cattleya labiata orchid",
            "Cattleya labiata purple flower",
            "cattleya labiata bloom",
            "cattleya orchid purple lip",
            "Cattleya labiata close up",
            "cattleya labiata pink flower"
        ],
    },
    # SỬA ĐỔI CHI TIẾT: Đồng bộ sang Cymbidium lowianum thay vì Ensifolium
    "cymbidium_lowianum": {
        "taxon": "Cymbidium lowianum",
        "keywords": [
            "Cymbidium lowianum orchid flower",
            "Cymbidium lowianum green orchid",
            "Cymbidium lowianum bloom",
            "cymbidium lowianum close up",
            "cymbidium lowianum red lip",
            "large cymbidium orchid green yellow"
        ],
    },
    "vanda_coerulea": {
        "taxon": "Vanda coerulea",
        "keywords": [
            "Vanda coerulea orchid",
            "blue vanda orchid",
            "Vanda coerulea tessellation",
            "vanda coerulea flower bloom",
            "blue vanda flower close up",
            "Vanda coerulea purple"
        ],
    },
    "dendrobium_anosmum": {
        "taxon": "Dendrobium anosmum",
        "keywords": [
            "Dendrobium anosmum orchid",
            "Dendrobium anosmum flower",
            "dendrobium anosmum purple bloom",
            "lan gia hac phi diep",
            "Dendrobium anosmum close up flower",
            "dendrobium anosmum orchid spike"
        ],
    },
    "oncidium_sphacelatum": {
        "taxon": "Oncidium sphacelatum",
        "keywords": [
            "Oncidium sphacelatum dancing lady orchid",
            "Oncidium sphacelatum orchid",
            "Oncidium sphacelatum flower bloom",
            "yellow dancing lady orchid",
            "Oncidium sphacelatum cluster",
            "oncidium sphacelatum close up"
        ],
    },
    "phalaenopsis_amabilis": {
        "taxon": "Phalaenopsis amabilis",
        "keywords": [
            "Phalaenopsis amabilis orchid flower",
            "Phalaenopsis amabilis white orchid",
            "moth orchid white bloom",
            "phalaenopsis amabilis close up flower",
            "Phalaenopsis amabilis wild orchid",
            "white phalaenopsis orchid petal"
        ],
    },
    "rhynchostylis_gigantea": {
        "taxon": "Rhynchostylis gigantea",
        "keywords": [
            "Rhynchostylis gigantea orchid",
            "Rhynchostylis gigantea flower bloom",
            "foxtail orchid rhynchostylis gigantea",
            "rhynchostylis gigantea spotted purple",
            "lan dai chau nghinh xuan",
            "rhynchostylis gigantea close up cluster"
        ],
    },
}

BASE_DIR    = "dataset_orchid"
TARGET      = 1000      # Số lượng ảnh gốc đạt chuẩn thiết lập trong báo cáo
PER_KEYWORD = 200       # Giới hạn lấy ảnh trên mỗi từ khóa nhằm tránh bị block IP

def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg','.jpeg','.png'))])

def get_next_index(folder):
    files = os.listdir(folder)
    nums = [int(os.path.splitext(f)[0]) for f in files
            if os.path.splitext(f)[0].isdigit()]
    return max(nums) + 1 if nums else 1

def crawl_inaturalist(taxon, save_dir):
    current = count_images(save_dir)
    idx = get_next_index(save_dir)
    page = 1
    while current < TARGET:
        try:
            results = get_observations(
                taxon_name=taxon, photos=True,
                quality_grade="research",
                per_page=200, page=page,
            )
        except Exception as e:
            print(f"  ⚠️ iNat lỗi: {e}"); break
        obs = results["results"]
        if not obs:
            print(f"  ⚠️ iNat hết data ở page {page}"); break
        for o in tqdm(obs, ncols=60, leave=False):
            if current >= TARGET: break
            try:
                url = o["photos"][0]["url"].replace("square", "medium")
                r = requests.get(url, timeout=8)
                if r.status_code == 200 and len(r.content) > 5000:
                    with open(f"{save_dir}/{idx:06d}.jpg", "wb") as f:
                        f.write(r.content)
                    idx += 1; current += 1
            except: pass
        print(f"  iNat page {page} → tổng hiện tại: {current}")
        page += 1
        time.sleep(1.5)
    return current

def crawl_google(keyword, save_dir, max_num):
    try:
        crawler = GoogleImageCrawler(
            feeder_threads=2, parser_threads=2, downloader_threads=6,
            storage={'root_dir': save_dir}
        )
        crawler.crawl(keyword=keyword, max_num=max_num, min_size=(150,150))
    except Exception as e:
        print(f"  ⚠️ Google lỗi: {e}")

def crawl_bing(keyword, save_dir, max_num):
    try:
        crawler = BingImageCrawler(
            feeder_threads=2, parser_threads=2, downloader_threads=6,
            storage={'root_dir': save_dir}
        )
        crawler.crawl(keyword=keyword, max_num=max_num, min_size=(150,150))
    except Exception as e:
        print(f"  ⚠️ Bing lỗi: {e}")

# ============================================================
# TIẾN TRÌNH KHỞI CHẠY TỔNG THỂ (MAIN EXECUTION)
# ============================================================
print("=" * 60)
print("🌸  HỆ THỐNG MASTER CRAWL DATASET V2 (ĐỒNG BỘ 8 LOÀI LAN)  🌸")
print("=" * 60)

for name, info in flowers.items():
    save_dir = os.path.join(BASE_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    current = count_images(save_dir)
    print(f"\n📂 Thư mục đối tượng: [{name}] — hiện có: {current} ảnh")

    # Kiểm tra nếu thư mục hiện tại đã đạt đủ số lượng mẫu mục tiêu
    if current >= TARGET:
        print("  ✅ Lớp đối tượng này đã đủ mẫu, hệ thống tự động bỏ qua.")
        continue

    # Chiến lược 1: Ưu tiên khai thác ảnh iNaturalist sạch, chuẩn định danh chuyên gia
    print(f"  🌿 Kết nối iNaturalist API: '{info['taxon']}'")
    crawl_inaturalist(info["taxon"], save_dir)

    # Chiến lược 2: Gọi bộ cào đa luồng mở rộng nếu iNaturalist thiếu ảnh
    for kw in info["keywords"]:
        current = count_images(save_dir)
        if current >= TARGET: break
        fetch = min(PER_KEYWORD, TARGET - current + 50)
        
        print(f"  🔍 Kích hoạt bộ cào Google Image: '{kw}'")
        crawl_google(kw, save_dir, fetch)
        time.sleep(2)
        
        current = count_images(save_dir)
        if current < TARGET:
            print(f"  🔍 Kích hoạt bộ cào Bing Image: '{kw}'")
            crawl_bing(kw, save_dir, fetch)
            time.sleep(2)

    print(f"  📊 Tổng kết lớp [{name}]: Đạt {count_images(save_dir)} ảnh")

print("\n" + "=" * 60)
print("✅  QUY TRÌNH MASTER CRAWL ĐỒNG BỘ HOÀN TẤT!")
print("=" * 60)