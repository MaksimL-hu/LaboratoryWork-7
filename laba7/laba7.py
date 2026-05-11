from pathlib import Path
import csv
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------
# НАСТРОЙКИ
# -----------------------------
OUTPUT_DIR = Path("lab7_output")

REFERENCE_DIR = OUTPUT_DIR / "reference_symbols"
INPUT_DIR = OUTPUT_DIR / "input"
EXPERIMENT_DIR = OUTPUT_DIR / "experiment"
VIS_DIR = OUTPUT_DIR / "visualizations"

REFERENCE_FEATURES_CSV = OUTPUT_DIR / "reference_features.csv"
INPUT_HYPOTHESES_TXT = OUTPUT_DIR / "input_hypotheses.txt"
EXPERIMENT_HYPOTHESES_TXT = OUTPUT_DIR / "experiment_hypotheses.txt"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"

for folder in [OUTPUT_DIR, REFERENCE_DIR, INPUT_DIR, EXPERIMENT_DIR, VIS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

INPUT_IMAGE = None

PREFERRED_INPUT_NAMES = [
    "phrase.bmp",
    "text.bmp",
]

GROUND_TRUTH_TEXT = "ТЫ ТАКОЙ КРАСИВЫЙ"

FONT_PATH = r"C:\Users\m9164\AppData\Local\Microsoft\Windows\Fonts\PonomarUnicode.ttf"
FONT_SIZE = 52

EXPERIMENT_FONT_SIZE = FONT_SIZE + 8
EXPERIMENT_SPACING_CANDIDATES = [0, 2, 4, 6, 8, 10, 12, 14, 16]
EXPERIMENT_WORD_SPACING_EXTRA = 10

CANVAS_SIZE = (240, 240)

TEXT_CANVAS_SIZE = (2400, 400)

BINARIZE_THRESHOLD = 200
CROP_PADDING = 0

# Алфавит
ALPHABET = [
    "А", "Б", "В", "Г", "Д", "Є", "Ж", "Ѕ", "З", "И", "І", "Й",
    "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Ѹ",
    "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Ѣ",
    "Ю", "Ѵ", "Ѯ", "Ѱ", "Ѡ", "Ѧ", "Ѩ"
]

LINE_PROFILE_THRESHOLD = 1
CHAR_PROFILE_THRESHOLD = 1

LINE_MIN_GAP = 6
LINE_MIN_RUN = 8

CHAR_MIN_GAP = 0
CHAR_MIN_RUN = 1

CHAR_BODY_TOP_RATIO = 0.15
CHAR_BODY_BOTTOM_RATIO = 0.92

WIDE_SEGMENT_FACTOR = 1.9

MERGE_GAP = 8
SMALL_PART_RATIO = 1.20

NORMALIZED_SIZE = (64, 64)
NORMALIZED_PADDING = 6

SCALAR_WEIGHT = 0.30
PROFILE_X_WEIGHT = 0.20
PROFILE_Y_WEIGHT = 0.20
BITMAP_WEIGHT = 0.30

RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


# -----------------------------
# ПОИСК ФАЙЛОВ
# -----------------------------
def find_input_image() -> Path:
    if INPUT_IMAGE is not None:
        if INPUT_IMAGE.exists():
            return INPUT_IMAGE
        raise FileNotFoundError(f"Входной файл не найден: {INPUT_IMAGE}")

    cwd = Path.cwd()

    for name in PREFERRED_INPUT_NAMES:
        candidate = cwd / name
        if candidate.exists():
            return candidate

    bmps = sorted(cwd.glob("*.bmp"))
    if bmps:
        return bmps[0]

    raise FileNotFoundError(
        "Не найден входной BMP-файл. "
        "Положите файл в папку со скриптом или укажите путь в INPUT_IMAGE."
    )


def find_ponomar_font() -> str:
    if FONT_PATH is not None:
        path = Path(FONT_PATH)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Шрифт не найден: {FONT_PATH}")

    candidates = [
        r"C:\Windows\Fonts\PonomarUnicode.ttf",
        r"C:\Windows\Fonts\PonomarUnicode.otf",
        r"C:\Windows\Fonts\Ponomar Unicode TT.ttf",
        r"C:\Windows\Fonts\Ponomar Unicode.ttf",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    windows_fonts = Path(r"C:\Windows\Fonts")
    if windows_fonts.exists():
        for path in windows_fonts.glob("*Ponomar*.*"):
            if path.suffix.lower() in [".ttf", ".otf"]:
                return str(path)

    raise FileNotFoundError(
        "Не найден шрифт Ponomar Unicode. "
        "Укажите путь к нему в переменной FONT_PATH."
    )


# -----------------------------
# РАБОТА С ИЗОБРАЖЕНИЯМИ
# -----------------------------
def rgb_to_grayscale_manual(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def load_image_as_binary(path: Path, threshold: int = 200) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    gray = rgb_to_grayscale_manual(rgb)
    binary = np.where(gray < threshold, 1, 0).astype(np.uint8)
    return gray, binary


def save_binary_image(binary: np.ndarray, path: Path) -> None:
    img = np.where(binary == 1, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def crop_binary_image(binary: np.ndarray, padding: int = 0) -> np.ndarray:
    ys, xs = np.where(binary == 1)

    if len(xs) == 0 or len(ys) == 0:
        return binary.copy()

    x_min = max(0, int(xs.min()) - padding)
    x_max = min(binary.shape[1], int(xs.max()) + 1 + padding)
    y_min = max(0, int(ys.min()) - padding)
    y_max = min(binary.shape[0], int(ys.max()) + 1 + padding)

    return binary[y_min:y_max, x_min:x_max]


def bounding_box_of_black(binary: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary == 1)

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, binary.shape[1], binary.shape[0]

    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1

    return x_min, y_min, x_max, y_max


def crop_binary(binary: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return binary[y1:y2, x1:x2]


def make_safe_name(symbol: str) -> str:
    return f"U+{ord(symbol):04X}_{symbol}"


# -----------------------------
# РЕНДЕРИНГ СИМВОЛОВ И ТЕКСТА
# -----------------------------
def render_symbol(symbol: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    image = Image.new("L", CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (CANVAS_SIZE[0] - text_w) // 2 - bbox[0]
    y = (CANVAS_SIZE[1] - text_h) // 2 - bbox[1]

    draw.text((x, y), symbol, font=font, fill=0)
    return np.array(image, dtype=np.uint8)


def render_text_line(
    text: str,
    font: ImageFont.FreeTypeFont,
    letter_spacing: int = 0,
    word_spacing: int | None = None
) -> np.ndarray:
    image = Image.new("L", TEXT_CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(image)

    if word_spacing is None:
        word_spacing = max(letter_spacing * 3, font.size // 2)

    # оценка общей высоты
    bboxes = []
    for ch in text:
        if ch == " ":
            continue
        bb = draw.textbbox((0, 0), ch, font=font)
        bboxes.append(bb)

    if not bboxes:
        return np.zeros((1, 1), dtype=np.uint8)

    top = min(bb[1] for bb in bboxes)
    bottom = max(bb[3] for bb in bboxes)
    text_h = bottom - top

    x = 20
    y = (TEXT_CANVAS_SIZE[1] - text_h) // 2 - top

    for ch in text:
        if ch == " ":
            x += word_spacing + EXPERIMENT_WORD_SPACING_EXTRA
            continue

        bb = draw.textbbox((0, 0), ch, font=font)
        ch_w = bb[2] - bb[0]

        draw.text((x - bb[0], y), ch, font=font, fill=0)
        x += ch_w + letter_spacing

    gray = np.array(image, dtype=np.uint8)
    binary = np.where(gray < BINARIZE_THRESHOLD, 1, 0).astype(np.uint8)
    binary = crop_binary_image(binary, padding=0)

    return binary


def choose_best_experiment_render(
    text: str,
    font: ImageFont.FreeTypeFont,
    reference: dict[str, dict],
    font_path: str,
    expected_len: int
) -> dict:
    best_result = None
    median_symbol_width = estimate_symbol_widths_from_alphabet(font_path, EXPERIMENT_FONT_SIZE)

    for spacing in EXPERIMENT_SPACING_CANDIDATES:
        binary = render_text_line(
            text,
            font,
            letter_spacing=spacing,
            word_spacing=max(font.size // 2, spacing * 3)
        )
        gray = np.where(binary == 1, 0, 255).astype(np.uint8)

        text_rect, line_rects, char_rects = segment_text(binary, median_symbol_width)
        all_hypotheses = recognize_characters(binary, char_rects, reference)
        recognized_text = best_text_from_hypotheses(all_hypotheses)

        count_diff = abs(len(char_rects) - expected_len)
        avg_similarity = 0.0
        if all_hypotheses:
            avg_similarity = float(np.mean([h[0][1] for h in all_hypotheses if h]))

        score = (-count_diff, avg_similarity)

        candidate = {
            "spacing": spacing,
            "gray": gray,
            "binary": binary,
            "text_rect": text_rect,
            "line_rects": line_rects,
            "char_rects": char_rects,
            "hypotheses": all_hypotheses,
            "recognized_text": recognized_text,
            "score": score,
        }

        if best_result is None or candidate["score"] > best_result["score"]:
            best_result = candidate

        # идеальный случай: число сегментов совпало
        if len(char_rects) == expected_len:
            break

    return best_result


# -----------------------------
# ПРОФИЛИ
# -----------------------------
def horizontal_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=1).astype(np.int32)


def vertical_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=0).astype(np.int32)


def bridge_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    result = mask.copy()
    n = len(result)
    i = 0

    while i < n:
        if result[i] == 0:
            start = i
            while i < n and result[i] == 0:
                i += 1
            end = i

            gap_len = end - start
            left_is_one = start > 0 and result[start - 1] == 1
            right_is_one = end < n and result[end] == 1

            if left_is_one and right_is_one and gap_len <= max_gap:
                result[start:end] = 1
        else:
            i += 1

    return result


def remove_small_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    result = mask.copy()
    n = len(result)
    i = 0

    while i < n:
        if result[i] == 1:
            start = i
            while i < n and result[i] == 1:
                i += 1
            end = i

            run_len = end - start
            if run_len < min_run:
                result[start:end] = 0
        else:
            i += 1

    return result


def thin_profile(
    profile: np.ndarray,
    threshold: int,
    min_gap: int,
    min_run: int,
    bridge_gaps: bool = True
) -> np.ndarray:
    mask = (profile >= threshold).astype(np.uint8)

    if bridge_gaps and min_gap > 0:
        mask = bridge_small_gaps(mask, max_gap=min_gap)

    mask = remove_small_runs(mask, min_run=min_run)
    return mask


def mask_to_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges = []
    n = len(mask)
    i = 0

    while i < n:
        if mask[i] == 1:
            start = i
            while i < n and mask[i] == 1:
                i += 1
            end = i
            ranges.append((start, end))
        else:
            i += 1

    return ranges


# -----------------------------
# СЕГМЕНТАЦИЯ
# -----------------------------
def get_char_body(line_crop: np.ndarray) -> tuple[np.ndarray, int, int]:
    h = line_crop.shape[0]
    y1 = int(round(h * CHAR_BODY_TOP_RATIO))
    y2 = int(round(h * CHAR_BODY_BOTTOM_RATIO))

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return line_crop[y1:y2, :], y1, y2


def estimate_symbol_widths_from_alphabet(font_path: str, font_size: int) -> float:
    font = ImageFont.truetype(font_path, font_size)
    widths = []

    for symbol in ALPHABET:
        gray = render_symbol(symbol, font)
        binary = np.where(gray < BINARIZE_THRESHOLD, 1, 0).astype(np.uint8)
        binary = crop_binary_image(binary, padding=CROP_PADDING)
        widths.append(binary.shape[1])

    if not widths:
        return 20.0

    return float(np.median(widths))


def split_wide_segment_by_valleys(
    body_profile: np.ndarray,
    start: int,
    end: int,
    target_width: float
) -> list[tuple[int, int]]:
    width = end - start
    if width <= WIDE_SEGMENT_FACTOR * target_width:
        return [(start, end)]

    local = body_profile[start:end]
    if len(local) < 4:
        return [(start, end)]

    valleys = []
    for i in range(1, len(local) - 1):
        if local[i] <= local[i - 1] and local[i] <= local[i + 1]:
            valleys.append((local[i], i))

    if not valleys:
        return [(start, end)]

    expected_parts = max(2, int(round(width / target_width)))
    expected_cuts = expected_parts - 1

    valleys_sorted = sorted(
        valleys,
        key=lambda t: (t[0], abs((start + t[1]) - (start + end) / 2))
    )
    candidate_positions = sorted([start + i for _, i in valleys_sorted[:expected_cuts]])

    if not candidate_positions:
        return [(start, end)]

    segments = []
    prev = start
    for cut in candidate_positions:
        if cut - prev >= 2:
            segments.append((prev, cut))
            prev = cut
    if end - prev >= 2:
        segments.append((prev, end))

    if len(segments) <= 1:
        return [(start, end)]

    return segments


def merge_adjacent_char_rects(
    char_rects: list[tuple[int, int, int, int]],
    median_symbol_width: float
) -> list[tuple[int, int, int, int]]:
    if not char_rects:
        return []

    merged = []
    i = 0

    while i < len(char_rects):
        current = char_rects[i]

        if i == len(char_rects) - 1:
            merged.append(current)
            break

        nxt = char_rects[i + 1]

        x1, y1, x2, y2 = current
        nx1, ny1, nx2, ny2 = nxt

        current_width = x2 - x1
        next_width = nx2 - nx1
        gap = nx1 - x2

        small_width = min(current_width, next_width)
        is_small_part = small_width <= median_symbol_width * SMALL_PART_RATIO

        if gap <= MERGE_GAP and is_small_part:
            new_rect = (
                min(x1, nx1),
                min(y1, ny1),
                max(x2, nx2),
                max(y2, ny2),
            )
            merged.append(new_rect)
            i += 2
        else:
            merged.append(current)
            i += 1

    return merged


def segment_lines(binary: np.ndarray, text_rect: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = text_rect
    text_crop = binary[y1:y2, x1:x2]

    hp = horizontal_profile(text_crop)
    hp_mask = thin_profile(
        hp,
        threshold=LINE_PROFILE_THRESHOLD,
        min_gap=LINE_MIN_GAP,
        min_run=LINE_MIN_RUN,
        bridge_gaps=True,
    )
    line_ranges = mask_to_ranges(hp_mask)

    line_rects = []
    for ys, ye in line_ranges:
        line_rect = (x1, y1 + ys, x2, y1 + ye)

        line_crop = crop_binary(binary, line_rect)
        lx1, ly1, lx2, ly2 = bounding_box_of_black(line_crop)
        refined_rect = (
            line_rect[0] + lx1,
            line_rect[1] + ly1,
            line_rect[0] + lx2,
            line_rect[1] + ly2,
        )
        line_rects.append(refined_rect)

    return line_rects


def segment_characters_in_line(
    binary: np.ndarray,
    line_rect: tuple[int, int, int, int],
    median_symbol_width: float
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = line_rect
    line_crop = binary[y1:y2, x1:x2]

    body_crop, _, _ = get_char_body(line_crop)
    vp = vertical_profile(body_crop)

    vp_mask = thin_profile(
        vp,
        threshold=CHAR_PROFILE_THRESHOLD,
        min_gap=CHAR_MIN_GAP,
        min_run=CHAR_MIN_RUN,
        bridge_gaps=False,
    )

    coarse_ranges = mask_to_ranges(vp_mask)

    final_ranges = []
    for xs, xe in coarse_ranges:
        split_ranges = split_wide_segment_by_valleys(vp, xs, xe, median_symbol_width)
        final_ranges.extend(split_ranges)

    char_rects = []
    for xs, xe in final_ranges:
        char_rect = (x1 + xs, y1, x1 + xe, y2)

        char_crop = crop_binary(binary, char_rect)
        cx1, cy1, cx2, cy2 = bounding_box_of_black(char_crop)
        refined_rect = (
            char_rect[0] + cx1,
            char_rect[1] + cy1,
            char_rect[0] + cx2,
            char_rect[1] + cy2,
        )
        char_rects.append(refined_rect)

    char_rects = merge_adjacent_char_rects(char_rects, median_symbol_width)
    return char_rects


def segment_text(
    binary: np.ndarray,
    median_symbol_width: float
) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    text_rect = bounding_box_of_black(binary)
    line_rects = segment_lines(binary, text_rect)

    char_rects = []
    for line_rect in line_rects:
        chars_in_line = segment_characters_in_line(binary, line_rect, median_symbol_width)
        char_rects.extend(chars_in_line)

    return text_rect, line_rects, char_rects


def region_density(binary: np.ndarray, y1: float, y2: float, x1: float, x2: float) -> float:
    h, w = binary.shape
    yy1 = max(0, min(h, int(round(y1 * h))))
    yy2 = max(yy1 + 1, min(h, int(round(y2 * h))))
    xx1 = max(0, min(w, int(round(x1 * w))))
    xx2 = max(xx1 + 1, min(w, int(round(x2 * w))))

    region = binary[yy1:yy2, xx1:xx2]
    area = region.shape[0] * region.shape[1]
    if area == 0:
        return 0.0
    return float(region.sum()) / area


def main_diagonal_density(binary: np.ndarray, band: int = 4) -> float:
    h, w = binary.shape
    if h == 0 or w == 0:
        return 0.0

    total = 0
    black = 0

    for y in range(h):
        x = int(round(y * (w - 1) / max(1, h - 1)))
        x1 = max(0, x - band)
        x2 = min(w, x + band + 1)
        strip = binary[y:y + 1, x1:x2]
        total += strip.size
        black += int(strip.sum())

    return black / total if total > 0 else 0.0


def middle_horizontal_density(binary: np.ndarray) -> float:
    return region_density(binary, 0.42, 0.58, 0.15, 0.85)


def right_middle_density(binary: np.ndarray) -> float:
    return region_density(binary, 0.35, 0.65, 0.65, 0.95)


def upper_right_density(binary: np.ndarray) -> float:
    return region_density(binary, 0.05, 0.35, 0.60, 0.95)


def lower_right_density(binary: np.ndarray) -> float:
    return region_density(binary, 0.65, 0.95, 0.60, 0.95)


def resolve_confusable_pair(
    normalized: np.ndarray,
    hypotheses: list[tuple[str, float, float]]
) -> list[tuple[str, float, float]]:
    if len(hypotheses) < 2:
        return hypotheses

    top1, top2 = hypotheses[0], hypotheses[1]
    pair = {top1[0], top2[0]}

    # Разбираем только близкие гипотезы
    if abs(top1[1] - top2[1]) > 0.02:
        return hypotheses

    # К vs Н
    if pair == {"К", "Н"}:
        rm = right_middle_density(normalized)
        ur = upper_right_density(normalized)
        lr = lower_right_density(normalized)

        # у Н сильнее правая вертикаль в центре,
        # у К сильнее верхний и нижний правые участки
        score_n = rm
        score_k = 0.5 * (ur + lr)

        preferred = "Н" if score_n >= score_k else "К"

    # И vs П
    elif pair == {"И", "П"}:
        diag = main_diagonal_density(normalized, band=4)
        top_band = region_density(normalized, 0.05, 0.20, 0.10, 0.90)

        # у И сильнее диагональ,
        # у П сильнее верхняя горизонталь
        preferred = "И" if diag >= top_band else "П"

    else:
        return hypotheses

    # Поднимаем выбранную букву наверх
    idx = None
    for i, h in enumerate(hypotheses):
        if h[0] == preferred:
            idx = i
            break

    if idx is None or idx == 0:
        return hypotheses

    fixed = hypotheses[:]
    best = fixed.pop(idx)
    fixed.insert(0, best)
    return fixed


# -----------------------------
# НОРМАЛИЗАЦИЯ И ПРИЗНАКИ
# -----------------------------
def resize_and_center(binary: np.ndarray, out_h: int = 64, out_w: int = 64) -> np.ndarray:
    h, w = binary.shape
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    inner_h = out_h - 2 * NORMALIZED_PADDING
    inner_w = out_w - 2 * NORMALIZED_PADDING
    if inner_h <= 0 or inner_w <= 0:
        raise ValueError("NORMALIZED_PADDING слишком большой для NORMALIZED_SIZE")

    scale = min(inner_w / w, inner_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
    img = img.resize((new_w, new_h), RESAMPLE_NEAREST)

    resized_binary = (np.array(img, dtype=np.uint8) > 0).astype(np.uint8)

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    y0 = (out_h - new_h) // 2
    x0 = (out_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized_binary

    return canvas


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    s = float(profile.sum())
    if s == 0:
        return np.zeros_like(profile, dtype=np.float64)
    return profile.astype(np.float64) / s


def extract_scalar_features(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    area = h * w

    if area == 0:
        return np.zeros(9, dtype=np.float64)

    weight = float(binary.sum())
    mass_rel = weight / area

    if weight == 0:
        return np.zeros(9, dtype=np.float64)

    y_idx, x_idx = np.indices((h, w))

    cx = float((x_idx * binary).sum()) / weight
    cy = float((y_idx * binary).sum()) / weight

    cx_norm = cx / (w - 1) if w > 1 else 0.0
    cy_norm = cy / (h - 1) if h > 1 else 0.0

    ix = float((((y_idx - cy) ** 2) * binary).sum())
    iy = float((((x_idx - cx) ** 2) * binary).sum())

    denom = float((w * w) * (h * h))
    ix_norm = ix / denom if denom > 0 else 0.0
    iy_norm = iy / denom if denom > 0 else 0.0

    q1, q2, q3, q4 = quarter_relative_weights(binary)

    return np.array([
        mass_rel,
        cx_norm,
        cy_norm,
        ix_norm,
        iy_norm,
        q1, q2, q3, q4
    ], dtype=np.float64)


def build_reference_database(font_path: str) -> dict[str, dict]:
    font = ImageFont.truetype(font_path, FONT_SIZE)
    reference = {}

    for symbol in ALPHABET:
        gray = render_symbol(symbol, font)
        binary = np.where(gray < BINARIZE_THRESHOLD, 1, 0).astype(np.uint8)
        binary = crop_binary_image(binary, padding=CROP_PADDING)

        normalized = resize_and_center(binary, NORMALIZED_SIZE[0], NORMALIZED_SIZE[1])
        scalars = extract_scalar_features(normalized)
        profile_x = normalize_profile(vertical_profile(normalized))
        profile_y = normalize_profile(horizontal_profile(normalized))

        safe_name = make_safe_name(symbol)
        img_path = REFERENCE_DIR / f"{safe_name}.png"
        save_binary_image(binary, img_path)

        reference[symbol] = {
            "symbol": symbol,
            "unicode_name": unicodedata.name(symbol, ""),
            "binary_raw": binary,
            "binary_norm": normalized,
            "scalars": scalars,
            "profile_x": profile_x,
            "profile_y": profile_y,
            "image_path": str(img_path),
        }

    return reference


def save_reference_features_csv(reference: dict[str, dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "symbol",
            "unicode_name",
            "mass_rel",
            "center_x_norm",
            "center_y_norm",
            "Ix_norm",
            "Iy_norm",
            "image_path",
        ])

        for symbol in ALPHABET:
            row = reference[symbol]
            fv = row["scalars"]
            writer.writerow([
                symbol,
                row["unicode_name"],
                f"{fv[0]:.6f}",
                f"{fv[1]:.6f}",
                f"{fv[2]:.6f}",
                f"{fv[3]:.6f}",
                f"{fv[4]:.6f}",
                row["image_path"],
            ])


# -----------------------------
# КЛАССИФИКАЦИЯ
# -----------------------------
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))


def similarity_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def classify_symbol(binary: np.ndarray, reference: dict[str, dict]) -> list[tuple[str, float, float]]:
    normalized = resize_and_center(binary, NORMALIZED_SIZE[0], NORMALIZED_SIZE[1])
    scalars = extract_scalar_features(normalized)
    profile_x = normalize_profile(vertical_profile(normalized))
    profile_y = normalize_profile(horizontal_profile(normalized))

    hypotheses = []

    for symbol in ALPHABET:
        ref = reference[symbol]

        d_scalar = euclidean_distance(scalars, ref["scalars"]) / np.sqrt(len(scalars))
        d_px = euclidean_distance(profile_x, ref["profile_x"])
        d_py = euclidean_distance(profile_y, ref["profile_y"])
        d_bitmap = float(np.mean(np.abs(
            normalized.astype(np.float64) - ref["binary_norm"].astype(np.float64)
        )))

        distance = (
            SCALAR_WEIGHT * d_scalar +
            PROFILE_X_WEIGHT * d_px +
            PROFILE_Y_WEIGHT * d_py +
            BITMAP_WEIGHT * d_bitmap
        )

        similarity = similarity_from_distance(distance)
        hypotheses.append((symbol, similarity, distance))

    hypotheses.sort(key=lambda x: (-x[1], x[2], x[0]))
    hypotheses = resolve_knip_cluster(normalized, hypotheses)
    hypotheses = resolve_i_vs_k_strict(normalized, hypotheses)
    return hypotheses


def quarter_relative_weights(binary: np.ndarray) -> list[float]:
    h, w = binary.shape
    mid_y = h // 2
    mid_x = w // 2

    quarters = [
        binary[:mid_y, :mid_x],
        binary[:mid_y, mid_x:],
        binary[mid_y:, :mid_x],
        binary[mid_y:, mid_x:],
    ]

    values = []
    for q in quarters:
        area = q.shape[0] * q.shape[1]
        values.append(float(q.sum()) / area if area > 0 else 0.0)

    return values


def resolve_knip_cluster(
    normalized: np.ndarray,
    hypotheses: list[tuple[str, float, float]]
) -> list[tuple[str, float, float]]:
    if len(hypotheses) < 4:
        return hypotheses

    top4 = [h[0] for h in hypotheses[:4]]
    cluster = {"К", "Н", "И", "П"}

    if len(cluster.intersection(top4)) < 3:
        return hypotheses

    h, w = normalized.shape

    def dens(y1, y2, x1, x2):
        yy1 = max(0, min(h, int(round(y1 * h))))
        yy2 = max(yy1 + 1, min(h, int(round(y2 * h))))
        xx1 = max(0, min(w, int(round(x1 * w))))
        xx2 = max(xx1 + 1, min(w, int(round(x2 * w))))
        region = normalized[yy1:yy2, xx1:xx2]
        area = region.shape[0] * region.shape[1]
        return float(region.sum()) / area if area > 0 else 0.0

    left_mid = dens(0.30, 0.70, 0.05, 0.30)
    right_mid = dens(0.30, 0.70, 0.70, 0.95)
    top_band = dens(0.05, 0.22, 0.10, 0.90)
    upper_right = dens(0.05, 0.35, 0.60, 0.95)
    lower_right = dens(0.65, 0.95, 0.60, 0.95)

    total = 0
    black = 0
    for y in range(h):
        x = int(round(y * (w - 1) / max(1, h - 1)))
        x1 = max(0, x - 4)
        x2 = min(w, x + 5)
        strip = normalized[y:y + 1, x1:x2]
        total += strip.size
        black += int(strip.sum())
    diag = black / total if total > 0 else 0.0

    scores = {}
    scores["Н"] = 0.45 * left_mid + 0.45 * right_mid - 0.25 * diag
    scores["П"] = 0.35 * left_mid + 0.35 * right_mid + 0.40 * top_band - 0.20 * diag
    scores["И"] = 0.30 * left_mid + 0.30 * right_mid + 0.45 * diag - 0.15 * top_band
    scores["К"] = 0.35 * left_mid + 0.30 * upper_right + 0.30 * lower_right - 0.15 * top_band

    preferred = max(scores.items(), key=lambda x: x[1])[0]
    return move_symbol_to_front(hypotheses, preferred)


def row_coverage(binary: np.ndarray, y1: float, y2: float, x1: float, x2: float) -> float:
    h, w = binary.shape
    yy1 = max(0, min(h, int(round(y1 * h))))
    yy2 = max(yy1 + 1, min(h, int(round(y2 * h))))
    xx1 = max(0, min(w, int(round(x1 * w))))
    xx2 = max(xx1 + 1, min(w, int(round(x2 * w))))

    region = binary[yy1:yy2, xx1:xx2]
    if region.size == 0:
        return 0.0

    rows_with_black = np.sum(region.sum(axis=1) > 0)
    return float(rows_with_black) / region.shape[0]


def move_symbol_to_front(
    hypotheses: list[tuple[str, float, float]],
    symbol: str
) -> list[tuple[str, float, float]]:
    idx = None
    for i, h in enumerate(hypotheses):
        if h[0] == symbol:
            idx = i
            break

    if idx is None or idx == 0:
        return hypotheses

    fixed = hypotheses[:]
    best = fixed.pop(idx)
    fixed.insert(0, best)
    return fixed


def resolve_i_vs_k_strict(
    normalized: np.ndarray,
    hypotheses: list[tuple[str, float, float]]
) -> list[tuple[str, float, float]]:
    if len(hypotheses) < 3:
        return hypotheses

    symbols = [h[0] for h in hypotheses[:4]]
    if "К" not in symbols or "И" not in symbols:
        return hypotheses

    top_symbol = hypotheses[0][0]
    if top_symbol != "К":
        return hypotheses

    sim_k = next(h[1] for h in hypotheses if h[0] == "К")
    sim_i = next(h[1] for h in hypotheses if h[0] == "И")

    if sim_k - sim_i > 0.025:
        return hypotheses

    diag = main_diagonal_density(normalized, band=3)

    right_full_cov = row_coverage(normalized, 0.08, 0.92, 0.72, 0.95)
    right_mid_cov = row_coverage(normalized, 0.35, 0.65, 0.72, 0.95)

    ur = upper_right_density(normalized)
    lr = lower_right_density(normalized)

    looks_like_i = (
        diag >= 0.23 and
        right_full_cov >= 0.82 and
        right_mid_cov >= 0.78 and
        ur <= 0.58 and
        lr <= 0.58
    )

    if looks_like_i:
        return move_symbol_to_front(hypotheses, "И")

    return hypotheses


def recognize_characters(
    binary: np.ndarray,
    char_rects: list[tuple[int, int, int, int]],
    reference: dict[str, dict]
) -> list[list[tuple[str, float, float]]]:
    all_hypotheses = []

    for rect in char_rects:
        char_crop = crop_binary(binary, rect)
        hypotheses = classify_symbol(char_crop, reference)
        all_hypotheses.append(hypotheses)

    return all_hypotheses


def best_text_from_hypotheses(all_hypotheses: list[list[tuple[str, float, float]]]) -> str:
    chars = []
    for hypotheses in all_hypotheses:
        if hypotheses:
            chars.append(hypotheses[0][0])
    return "".join(chars)


def normalize_ground_truth(text: str) -> str:
    return "".join(ch for ch in text if ch in ALPHABET)


def compare_texts(recognized: str, ground_truth: str) -> tuple[int, int, float]:
    max_len = max(len(recognized), len(ground_truth))
    if max_len == 0:
        return 0, 0, 100.0

    min_len = min(len(recognized), len(ground_truth))
    correct = 0

    for i in range(min_len):
        if recognized[i] == ground_truth[i]:
            correct += 1

    errors = max_len - correct
    accuracy = 100.0 * correct / max_len
    return errors, correct, accuracy


def save_hypotheses_txt(all_hypotheses: list[list[tuple[str, float, float]]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, hypotheses in enumerate(all_hypotheses, start=1):
            pretty = [(sym, round(sim, 6)) for sym, sim, _ in hypotheses]
            f.write(f"{i}: {pretty}\n")


# -----------------------------
# ВИЗУАЛИЗАЦИЯ
# -----------------------------
def save_boxes_visualization(
    gray: np.ndarray,
    text_rect: tuple[int, int, int, int],
    line_rects: list[tuple[int, int, int, int]],
    char_rects: list[tuple[int, int, int, int]],
    out_path: Path,
    title: str
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

    x1, y1, x2, y2 = text_rect
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="blue", linewidth=2.0))

    for rect in line_rects:
        rx1, ry1, rx2, ry2 = rect
        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False, edgecolor="green", linewidth=1.5))

    for idx, rect in enumerate(char_rects, start=1):
        rx1, ry1, rx2, ry2 = rect
        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False, edgecolor="red", linewidth=0.9))
        ax.text(rx1, max(0, ry1 - 3), str(idx), fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# ОДИН ПОЛНЫЙ ЗАПУСК РАСПОЗНАВАНИЯ
# -----------------------------
def run_recognition_pipeline(
    gray: np.ndarray,
    binary: np.ndarray,
    reference: dict[str, dict],
    median_symbol_width: float,
    hypotheses_txt_path: Path,
    vis_path: Path
) -> dict:
    text_rect, line_rects, char_rects = segment_text(binary, median_symbol_width)
    all_hypotheses = recognize_characters(binary, char_rects, reference)
    recognized_text = best_text_from_hypotheses(all_hypotheses)

    save_hypotheses_txt(all_hypotheses, hypotheses_txt_path)
    save_boxes_visualization(
        gray,
        text_rect,
        line_rects,
        char_rects,
        vis_path,
        "Сегментация и распознавание"
    )

    return {
        "text_rect": text_rect,
        "line_rects": line_rects,
        "char_rects": char_rects,
        "hypotheses": all_hypotheses,
        "recognized_text": recognized_text,
    }


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=== Лабораторная работа №7 ===")
    print("Классификация на основе признаков, анализ профилей")
    print()

    input_path = find_input_image()
    font_path = find_ponomar_font()

    median_symbol_width_input = estimate_symbol_widths_from_alphabet(font_path, FONT_SIZE)
    median_symbol_width_experiment = estimate_symbol_widths_from_alphabet(font_path, EXPERIMENT_FONT_SIZE)

    print(f"Входное изображение: {input_path}")
    print(f"Шрифт:               {font_path}")
    print(f"Средняя ширина (52): {median_symbol_width_input:.2f}")
    print(f"Средняя ширина ({EXPERIMENT_FONT_SIZE}): {median_symbol_width_experiment:.2f}")
    print(f"Алфавит:             {' '.join(ALPHABET)}")
    print()

    # 1. База эталонов
    reference = build_reference_database(font_path)
    save_reference_features_csv(reference, REFERENCE_FEATURES_CSV)

    # 2. Распознавание входного bmp
    gray_input, binary_input = load_image_as_binary(input_path, threshold=BINARIZE_THRESHOLD)
    Image.fromarray(gray_input, mode="L").save(INPUT_DIR / "input_gray.bmp")
    save_binary_image(binary_input, INPUT_DIR / "input_binary.bmp")

    input_result = run_recognition_pipeline(
        gray=gray_input,
        binary=binary_input,
        reference=reference,
        median_symbol_width=median_symbol_width_input,
        hypotheses_txt_path=INPUT_HYPOTHESES_TXT,
        vis_path=VIS_DIR / "input_boxes.png",
    )

    gt_norm = normalize_ground_truth(GROUND_TRUTH_TEXT)
    recognized_input = input_result["recognized_text"]
    errors_input, correct_input, acc_input = compare_texts(recognized_input, gt_norm)

    # 3. Эксперимент с другим размером шрифта
    exp_font = ImageFont.truetype(font_path, EXPERIMENT_FONT_SIZE)

    best_exp = choose_best_experiment_render(
        text=GROUND_TRUTH_TEXT,
        font=exp_font,
        reference=reference,
        font_path=font_path,
        expected_len=len(gt_norm),
    )

    exp_gray = best_exp["gray"]
    exp_binary = best_exp["binary"]

    Image.fromarray(exp_gray, mode="L").save(EXPERIMENT_DIR / "experiment_gray.bmp")
    save_binary_image(exp_binary, EXPERIMENT_DIR / "experiment_binary.bmp")

    save_hypotheses_txt(best_exp["hypotheses"], EXPERIMENT_HYPOTHESES_TXT)
    save_boxes_visualization(
        exp_gray,
        best_exp["text_rect"],
        best_exp["line_rects"],
        best_exp["char_rects"],
        VIS_DIR / "experiment_boxes.png",
        f"Эксперимент: spacing={best_exp['spacing']}"
    )

    experiment_result = {
        "text_rect": best_exp["text_rect"],
        "line_rects": best_exp["line_rects"],
        "char_rects": best_exp["char_rects"],
        "hypotheses": best_exp["hypotheses"],
        "recognized_text": best_exp["recognized_text"],
    }

    recognized_experiment = experiment_result["recognized_text"]
    errors_exp, correct_exp, acc_exp = compare_texts(recognized_experiment, gt_norm)

    # 4. Сохранение сегментированных символов
    for idx, rect in enumerate(input_result["char_rects"], start=1):
        crop = crop_binary(binary_input, rect)
        save_binary_image(crop, INPUT_DIR / f"char_{idx:03d}.png")

    for idx, rect in enumerate(experiment_result["char_rects"], start=1):
        crop = crop_binary(exp_binary, rect)
        save_binary_image(crop, EXPERIMENT_DIR / f"char_{idx:03d}.png")

    # 5. Итоговый отчёт
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("Лабораторная работа №7\n")
        f.write("Классификация на основе признаков, анализ профилей\n\n")

        f.write(f"Исходная фраза: {GROUND_TRUTH_TEXT}\n")
        f.write(f"Нормализованная фраза: {gt_norm}\n\n")

        f.write("=== Входное изображение ===\n")
        f.write(f"Распознанная строка: {recognized_input}\n")
        f.write(f"Количество символов: {len(input_result['char_rects'])}\n")
        f.write(f"Верно: {correct_input}\n")
        f.write(f"Ошибок: {errors_input}\n")
        f.write(f"Точность: {acc_input:.2f}%\n\n")

        f.write("=== Эксперимент (другой размер шрифта) ===\n")
        f.write(f"Размер шрифта: {EXPERIMENT_FONT_SIZE}\n")
        f.write(f"Подобранный spacing: {best_exp['spacing']}\n")
        f.write(f"Распознанная строка: {recognized_experiment}\n")
        f.write(f"Количество символов: {len(experiment_result['char_rects'])}\n")
        f.write(f"Верно: {correct_exp}\n")
        f.write(f"Ошибок: {errors_exp}\n")
        f.write(f"Точность: {acc_exp:.2f}%\n")

    print("=== Результаты ===")
    print(f"Эталонные признаки:    {REFERENCE_FEATURES_CSV}")
    print(f"Гипотезы входа:        {INPUT_HYPOTHESES_TXT}")
    print(f"Гипотезы эксперимента: {EXPERIMENT_HYPOTHESES_TXT}")
    print(f"Сводка:                {SUMMARY_TXT}")
    print()

    print("=== Входное изображение ===")
    print(f"Распознанная строка: {recognized_input}")
    print(f"Верно: {correct_input}")
    print(f"Ошибок: {errors_input}")
    print(f"Точность: {acc_input:.2f}%")
    print()

    print("=== Эксперимент ===")
    print(f"Размер шрифта: {EXPERIMENT_FONT_SIZE}")
    print(f"Подобранный spacing: {best_exp['spacing']}")
    print(f"Распознанная строка: {recognized_experiment}")
    print(f"Верно: {correct_exp}")
    print(f"Ошибок: {errors_exp}")
    print(f"Точность: {acc_exp:.2f}%")


if __name__ == "__main__":
    main()