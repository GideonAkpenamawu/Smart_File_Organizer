"""
main.py
EL Smart Organizer — ML Edition (final)
Features:
 - Thumbnails (images + PDF first-page)
 - Preview pane (images, PDF first page, text)
 - PDF auto-categorization using trainable TF-IDF + LogisticRegression
 - OCR fallback (pytesseract) optional
 - Confidence threshold -> "Review" folder for low-confidence predictions
 - Train classifier UI, save/load model
 - Duplicates mover, scheduled cleanup, analytics (SQLite)
 - Dark/Light theme toggle
 - Cloud placeholders: Google Drive & OneDrive (OAuth-ready placeholders)
 - Packaging-ready (assets directory expected)
"""
import os
import sys
import shutil
import hashlib
import sqlite3
import joblib
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QFileDialog, QMessageBox, QHBoxLayout, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QSpinBox, QTextEdit, QProgressBar, QDialog,
    QDialogButtonBox, QTableWidget, QTableWidgetItem, QInputDialog, QHeaderView,
    QSlider, QSplitter
)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import fitz  # PyMuPDF

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Cloud placeholder libs (optional)
try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GOOGLE_LIBS_AVAILABLE = True
except Exception:
    GOOGLE_LIBS_AVAILABLE = False

try:
    import msal
    import requests
    MSAL_AVAILABLE = True
except Exception:
    MSAL_AVAILABLE = False

# ---------------------------
# Config
IMAGE_EXTS = {"png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff"}
ANALYTICS_DB = "analytics.db"
MODEL_PATH = "pdf_classifier.joblib"
DEFAULT_CONF_THRESHOLD = 0.60  # default confidence threshold

# ---------------------------
# Utilities
def md5_hash(path, block_size=1 << 20):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def get_pdf_thumbnail_bytes(pdf_path, zoom=0.6):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None

def extract_pdf_text_firstpage(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        text = page.get_text("text") or ""
        return text
    except Exception:
        return ""

def ocr_pdf_firstpage(pdf_path):
    if not OCR_AVAILABLE:
        return ""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img) or ""
        return text
    except Exception:
        return ""

# ---------------------------
# Analytics
def analytics_init():
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            files_cleaned INTEGER,
            bytes_duplicates_moved INTEGER
        )
    """)
    conn.commit()
    conn.close()

def analytics_log(files_cleaned: int, bytes_duplicates_moved: int):
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    c.execute("INSERT INTO runs (ts, files_cleaned, bytes_duplicates_moved) VALUES (datetime('now'), ?, ?)",
              (files_cleaned, bytes_duplicates_moved))
    conn.commit()
    conn.close()

def analytics_fetch_all():
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    c.execute("SELECT ts, files_cleaned, bytes_duplicates_moved FROM runs ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return rows

# ---------------------------
# ML helpers
def save_model(obj, path=MODEL_PATH):
    joblib.dump(obj, path)

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def train_pdf_classifier(texts, labels):
    texts_proc = [t.lower() if t else "" for t in texts]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X = vectorizer.fit_transform(texts_proc)
    y = labels
    test_size = 0.2 if len(texts) >= 5 else 0.25
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(set(y))>1 else None)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return {"vectorizer": vectorizer, "model": clf, "accuracy": acc, "report": report}

# ---------------------------
# Cloud placeholders
# Google Drive (placeholder) - requires client secret JSON and google libs
GDRIVE_CLIENT_SECRETS = "credentials_gdrive.json"  # put your credentials here if you want to enable
GDRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.file']

def gdrive_get_service_interactive():
    """
    Starts a local server auth flow (interactive). Requires google-auth libs and credentials JSON.
    """
    if not GOOGLE_LIBS_AVAILABLE:
        raise RuntimeError("Google Drive libraries not installed. pip install google-auth google-auth-oauthlib google-api-python-client")
    if not os.path.exists(GDRIVE_CLIENT_SECRETS):
        raise FileNotFoundError(f"Place your Google client secret JSON as {GDRIVE_CLIENT_SECRETS}")
    flow = InstalledAppFlow.from_client_secrets_file(GDRIVE_CLIENT_SECRETS, GDRIVE_SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('drive', 'v3', credentials=creds)
    return service

# OneDrive placeholder (MSAL)
ONEDRIVE_CLIENT_ID = ""  # add your client id here
ONEDRIVE_TENANT = "common"
ONEDRIVE_SCOPES = ["Files.ReadWrite"]

def onedrive_device_flow_interactive(client_id=ONEDRIVE_CLIENT_ID):
    if not MSAL_AVAILABLE:
        raise RuntimeError("MSAL not installed. pip install msal")
    if not client_id:
        raise RuntimeError("Set your ONEDRIVE_CLIENT_ID in code to use OneDrive integration.")
    app = msal.PublicClientApplication(client_id, authority=f"https://login.microsoftonline.com/{ONEDRIVE_TENANT}")
    flow = app.initiate_device_flow(scopes=ONEDRIVE_SCOPES)
    print(flow["message"])
    result = app.acquire_token_by_device_flow(flow)
    return result

# ---------------------------
# Worker
class WorkerThread(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, task_type, src, dst, options, classifier_bundle=None):
        super().__init__()
        self.task_type = task_type
        self.src = src
        self.dst = dst
        self.options = options or {}
        self.classifier_bundle = classifier_bundle

    def run(self):
        try:
            if self.task_type == "duplicates":
                res = self._find_and_move_duplicates()
                self.finished.emit(res)
            elif self.task_type == "organize":
                res = self._organize_files()
                self.finished.emit(res)
            else:
                self.finished.emit({"error": "Unknown task"})
        except Exception as e:
            self.finished.emit({"error": str(e)})

    def _find_and_move_duplicates(self):
        src = self.src
        duplicates_folder = os.path.join(src, "Duplicates")
        os.makedirs(duplicates_folder, exist_ok=True)
        seen = {}
        moved = 0
        bytes_moved = 0
        files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        total = max(len(files), 1)
        for i, fname in enumerate(files, 1):
            full = os.path.join(src, fname)
            self.message.emit(f"Hashing {fname}...")
            h = md5_hash(full)
            if not h:
                self.progress.emit(int(i / total * 100)); continue
            if h in seen:
                base, ext = os.path.splitext(fname)
                target_name = f"{base}_dup{int(datetime.now().timestamp())}{ext}"
                target = os.path.join(duplicates_folder, target_name)
                try:
                    size = os.path.getsize(full)
                    shutil.move(full, target)
                    moved += 1
                    bytes_moved += size
                    self.message.emit(f"Moved duplicate: {fname}")
                except Exception as e:
                    self.message.emit(f"Failed to move {fname}: {e}")
            else:
                seen[h] = full
            self.progress.emit(int(i / total * 100))
        return {"moved": moved, "bytes_moved": bytes_moved, "scanned": len(files)}

    def _predict_pdf_category(self, pdf_path, use_ocr=False):
        if self.classifier_bundle and "vectorizer" in self.classifier_bundle and "model" in self.classifier_bundle:
            text = extract_pdf_text_firstpage(pdf_path) or ""
            if (not text.strip()) and use_ocr:
                text = ocr_pdf_firstpage(pdf_path) or ""
            t = text.lower()
            try:
                X = self.classifier_bundle["vectorizer"].transform([t])
                pred = self.classifier_bundle["model"].predict(X)[0]
                proba = max(self.classifier_bundle["model"].predict_proba(X)[0])
                return pred, float(proba)
            except Exception:
                return None, 0.0
        return None, 0.0

    def _organize_files(self):
        src = self.src; dst = self.dst
        rename_mode = self.options.get("rename_mode", "none")
        keyword = self.options.get("keyword", "")
        custom_fmt = self.options.get("custom_fmt", "{name}_{date}{ext}")
        remove_duplicates_first = self.options.get("remove_duplicates_first", False)
        pdf_categorize = self.options.get("pdf_categorize", False)
        pdf_use_ocr = self.options.get("pdf_use_ocr", False)
        conf_threshold = float(self.options.get("conf_threshold", DEFAULT_CONF_THRESHOLD))

        files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        total = max(len(files), 1)
        moved = 0; bytes_dup = 0

        if remove_duplicates_first:
            dr = self._find_and_move_duplicates()
            bytes_dup += dr.get("bytes_moved", 0)

        for i, fname in enumerate(files, 1):
            full = os.path.join(src, fname)
            if not os.path.isfile(full): continue
            ext = os.path.splitext(fname)[1].lower()
            ext_clean = ext[1:] if ext.startswith(".") else ext
            folder_name = f"{ext_clean.upper() or 'NOEXT'}_FILES"

            if ext == ".pdf" and pdf_categorize:
                pred, conf = self._predict_pdf_category(full, use_ocr=pdf_use_ocr)
                if pred and conf >= conf_threshold:
                    folder_name = pred
                    self.message.emit(f"Classifier -> {fname} => {pred} (conf={conf:.2f})")
                else:
                    # fallback keywords
                    text = extract_pdf_text_firstpage(full) or ""
                    if (not text.strip()) and pdf_use_ocr:
                        text = ocr_pdf_firstpage(full) or ""
                    txt_l = text.lower()
                    kwmap = {
                        "Invoices": ["invoice", "payment", "receipt", "amount due", "invoice no", "total amount"],
                        "Reports": ["report", "summary", "analysis", "findings", "executive summary"],
                        "Contracts": ["contract", "agreement", "terms", "signature"],
                        "Receipts": ["receipt", "paid", "transaction", "order confirmation"],
                        "Letters": ["dear", "regards", "sincerely"],
                    }
                    found = False
                    for cat, keys in kwmap.items():
                        for k in keys:
                            if k in txt_l:
                                folder_name = cat; found = True; break
                        if found: break
                    if not found:
                        # low-confidence or no keyword -> Review
                        folder_name = "Review"

            new_folder = os.path.join(dst, folder_name)
            os.makedirs(new_folder, exist_ok=True)

            base = os.path.splitext(fname)[0]
            if rename_mode == "date":
                date_part = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y%m%d_%H%M%S")
                new_name = f"{base}_{date_part}{ext}"
            elif rename_mode == "keyword" and keyword:
                safe_kw = "".join(c for c in keyword if c.isalnum() or c in ("-", "_"))
                new_name = f"{safe_kw}_{base}{ext}"
            elif rename_mode == "custom":
                date_part = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = custom_fmt.format(name=base, date=date_part, ext=ext)
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{base}_{ts}{ext}"

            dest_path = os.path.join(new_folder, new_name)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(new_folder, f"{os.path.splitext(new_name)[0]}_{counter}{ext}")
                counter += 1
            try:
                shutil.move(full, dest_path)
                moved += 1
                self.message.emit(f"Moved: {fname} -> {os.path.join(folder_name, os.path.basename(dest_path))}")
            except Exception as e:
                self.message.emit(f"Failed to move {fname}: {e}")

            self.progress.emit(int(i / total * 100))

        return {"moved": moved, "bytes_moved": bytes_dup, "scanned": len(files)}

# ---------------------------
# Preview & Dialogs
from PyQt5.QtWidgets import QVBoxLayout
class PreviewDialog(QDialog):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(file_path))
        self.setMinimumSize(700, 560)
        layout = QVBoxLayout()
        pixmap = None
        ext = file_path.split(".")[-1].lower()
        if ext in IMAGE_EXTS:
            pixmap = QPixmap(file_path)
        elif ext == "pdf":
            b = get_pdf_thumbnail_bytes(file_path, zoom=1.0)
            if b:
                pixmap = QPixmap(); pixmap.loadFromData(b)
        label = QLabel(); label.setAlignment(Qt.AlignCenter)
        if pixmap and not pixmap.isNull():
            label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            label.setText("Preview not available.")
        layout.addWidget(label)
        try:
            info = QLabel(f"Path: {file_path}\nSize: {os.path.getsize(file_path)} bytes\nModified: {datetime.fromtimestamp(os.path.getmtime(file_path))}")
        except Exception:
            info = QLabel("File information not available.")
        info.setWordWrap(True); layout.addWidget(info)
        buttons = QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

class AnalyticsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EL MEDIA Analytics")
        self.setMinimumSize(800, 500)
        layout = QVBoxLayout(); self.setLayout(layout)
        fig1 = plt.Figure(figsize=(6,2.5)); self.canvas1 = FigureCanvas(fig1); layout.addWidget(self.canvas1); self.ax1 = fig1.add_subplot(111)
        fig2 = plt.Figure(figsize=(6,2.5)); self.canvas2 = FigureCanvas(fig2); layout.addWidget(self.canvas2); self.ax2 = fig2.add_subplot(111)
        self._load_and_plot()

    def _load_and_plot(self):
        rows = analytics_fetch_all()
        if not rows:
            self.ax1.text(0.5,0.5,"No analytics data yet.", ha="center"); self.ax2.text(0.5,0.5,"No analytics data yet.", ha="center"); self.canvas1.draw(); self.canvas2.draw(); return
        dates = [r[0] for r in rows]; files = [r[1] for r in rows]; bytes_saved = [r[2]/(1024*1024) for r in rows]
        self.ax1.clear(); self.ax1.plot(dates, files, marker="o"); self.ax1.set_title("Files cleaned per run"); self.ax1.set_ylabel("Files"); self.ax1.set_xticklabels(dates, rotation=45, ha="right"); self.canvas1.draw()
        self.ax2.clear(); self.ax2.plot(dates, bytes_saved, marker="o"); self.ax2.set_title("Bytes moved to 'Duplicates' (MB) per run"); self.ax2.set_ylabel("MB"); self.ax2.set_xticklabels(dates, rotation=45, ha="right"); self.canvas2.draw()

class TrainModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train PDF Classifier")
        self.setMinimumSize(760, 520)
        self.examples = []
        self.bundle = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Add example PDFs and label them. Extracted first-page text will be used for training."))
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add PDF Examples"); self.add_btn.clicked.connect(self._add_examples)
        self.remove_btn = QPushButton("Remove Selected"); self.remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.add_btn); btn_row.addWidget(self.remove_btn); layout.addLayout(btn_row)
        self.table = QTableWidget(0,3); self.table.setHorizontalHeaderLabels(["File","Label","Extracted Text (truncated)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents); self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents); self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.table)
        train_row = QHBoxLayout(); self.test_size_combo = QComboBox(); self.test_size_combo.addItems(["0.2","0.25","0.1"])
        train_row.addWidget(QLabel("Test split:")); train_row.addWidget(self.test_size_combo)
        self.train_btn = QPushButton("Train Model"); self.train_btn.clicked.connect(self._train_model); train_row.addWidget(self.train_btn)
        layout.addLayout(train_row)
        self.report_box = QTextEdit(); self.report_box.setReadOnly(True); self.report_box.setFixedHeight(180); layout.addWidget(self.report_box)
        save_row = QHBoxLayout(); self.save_btn = QPushButton("Save Model"); self.save_btn.clicked.connect(self._save_model); save_row.addStretch(); save_row.addWidget(self.save_btn); layout.addLayout(save_row)
        self.setLayout(layout)

    def _add_examples(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select PDF examples", filter="PDF Files (*.pdf)")
        if not files: return
        for f in files:
            label, ok = QInputDialog.getText(self, "Label", f"Enter label for:\n{os.path.basename(f)}")
            if not ok or not label.strip(): continue
            text = extract_pdf_text_firstpage(f) or ""
            if (not text.strip()) and OCR_AVAILABLE:
                text = ocr_pdf_firstpage(f) or ""
            text_snip = (text[:300].replace("\n"," ") + "...") if text else ""
            self.examples.append((f, label.strip(), text))
            row = self.table.rowCount(); self.table.insertRow(row)
            self.table.setItem(row,0,QTableWidgetItem(os.path.basename(f))); self.table.setItem(row,1,QTableWidgetItem(label.strip())); self.table.setItem(row,2,QTableWidgetItem(text_snip))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
            del self.examples[r]

    def _train_model(self):
        if not self.examples: QMessageBox.warning(self,"No examples","Add labeled examples first."); return
        texts = [ex[2] for ex in self.examples]; labels = [ex[1] for ex in self.examples]
        if len(set(labels)) < 2: QMessageBox.warning(self,"Insufficient labels","Need at least 2 distinct labels to train."); return
        try:
            res = train_pdf_classifier(texts, labels)
            self.bundle = {"vectorizer": res["vectorizer"], "model": res["model"]}
            self.report_box.setPlainText(f"Training accuracy: {res['accuracy']:.3f}\n\nClassification report:\n{res['report']}")
            QMessageBox.information(self,"Training complete",f"Training complete. Accuracy (test): {res['accuracy']:.3f}")
        except Exception as e:
            QMessageBox.critical(self,"Training error",f"Training failed: {e}")

    def _save_model(self):
        if not self.bundle: QMessageBox.warning(self,"No model","Train model before saving"); return
        try:
            save_model(self.bundle)
            QMessageBox.information(self,"Saved",f"Model saved to {MODEL_PATH}")
        except Exception as e:
            QMessageBox.critical(self,"Save error",f"Failed to save model: {e}")

# ---------------------------
# Main UI
class ELSmartFileOrganizerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EL Smart Organizer — EL MEDIA")
        self.setMinimumSize(1000, 780)
        analytics_init()
        self.classifier_bundle = load_model()
        self.dark_mode = True
        self._build_ui()
        self.schedule_timer = QTimer(self); self.schedule_timer.timeout.connect(self._scheduled_run)
        self.worker = None
        if self.classifier_bundle:
            self._log("Loaded classifier model from disk.")
        else:
            self._log("No classifier model found — use Train Classifier to create one.")

    def _build_ui(self):
        root = QVBoxLayout()
        top = QHBoxLayout()
        logo = QLabel(); pix = QPixmap(os.path.join(os.path.dirname(__file__), "assets", "el_logo.png")) if os.path.exists(os.path.join(os.path.dirname(__file__), "assets", "el_logo.png")) else QPixmap()
        if not pix.isNull(): logo.setPixmap(pix.scaled(64,64,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        top.addWidget(logo)
        title = QLabel("EL Smart Organizer"); title.setFont(QFont("Segoe UI", 18, QFont.Bold)); top.addWidget(title)
        top.addStretch()
        self.theme_btn = QPushButton("Switch to Light Mode"); self.theme_btn.clicked.connect(self._toggle_theme); top.addWidget(self.theme_btn)
        # Cloud connect buttons
        self.gdrive_btn = QPushButton("Connect Google Drive"); self.gdrive_btn.clicked.connect(self._connect_gdrive); top.addWidget(self.gdrive_btn)
        self.onedrive_btn = QPushButton("Connect OneDrive"); self.onedrive_btn.clicked.connect(self._connect_onedrive); top.addWidget(self.onedrive_btn)
        root.addLayout(top)
        root.addWidget(QLabel("Organize smarter — click a file once to preview (double-click to open)."))
        sd = QHBoxLayout()
        self.src_input = QLineEdit(); self.src_input.setPlaceholderText("Source folder (e.g., Downloads)")
        self.dst_input = QLineEdit(); self.dst_input.setPlaceholderText("Destination folder")
        btn_src = QPushButton("Browse…"); btn_dst = QPushButton("Browse…")
        btn_src.clicked.connect(self._pick_source); btn_dst.clicked.connect(self._pick_dest)
        sd.addWidget(QLabel("Source:")); sd.addWidget(self.src_input); sd.addWidget(btn_src)
        sd.addSpacing(10); sd.addWidget(QLabel("Destination:")); sd.addWidget(self.dst_input); sd.addWidget(btn_dst)
        root.addLayout(sd)

        split = QSplitter(Qt.Horizontal)
        left_w = QWidget(); left_layout = QVBoxLayout()
        scan_row = QHBoxLayout(); self.scan_btn = QPushButton("Scan Source"); self.scan_btn.clicked.connect(self._scan)
        self.preview_popup_btn = QPushButton("Open Preview Window"); self.preview_popup_btn.clicked.connect(self._open_preview_popup)
        scan_row.addWidget(self.scan_btn); scan_row.addWidget(self.preview_popup_btn)
        left_layout.addLayout(scan_row)
        self.file_list = QListWidget(); self.file_list.setIconSize(QSize(80,80))
        self.file_list.itemClicked.connect(self._preview_selected)
        self.file_list.itemDoubleClicked.connect(self._open_preview_popup_from_item)
        left_layout.addWidget(self.file_list)
        left_w.setLayout(left_layout); split.addWidget(left_w)

        right_w = QWidget(); right_layout = QVBoxLayout()
        self.preview_image = QLabel(); self.preview_image.setAlignment(Qt.AlignCenter); self.preview_image.setFixedHeight(260)
        self.preview_text = QTextEdit(); self.preview_text.setReadOnly(True); self.preview_text.setFixedHeight(200)
        right_layout.addWidget(QLabel("Preview:")); right_layout.addWidget(self.preview_image); right_layout.addWidget(self.preview_text)
        right_layout.addWidget(QLabel("Rename Mode:"))
        self.rename_combo = QComboBox(); self.rename_combo.addItems(["None (append timestamp)","By Modified Date","By Keyword","Custom Format"])
        right_layout.addWidget(self.rename_combo)
        self.keyword_input = QLineEdit(); self.keyword_input.setPlaceholderText("Keyword (if used)"); right_layout.addWidget(self.keyword_input)
        self.custom_fmt_input = QLineEdit(); self.custom_fmt_input.setPlaceholderText("Custom format {name}_{date}{ext}"); self.custom_fmt_input.setText("{name}_{date}{ext}"); right_layout.addWidget(self.custom_fmt_input)
        self.dup_move_checkbox = QCheckBox("Move duplicates to 'Duplicates' folder (safe)"); self.dup_move_checkbox.setChecked(True); right_layout.addWidget(self.dup_move_checkbox)

        self.pdf_categorize_checkbox = QCheckBox("Auto-categorize PDFs by model (or fallback keywords)"); self.pdf_categorize_checkbox.setChecked(True); right_layout.addWidget(self.pdf_categorize_checkbox)
        self.ocr_checkbox = QCheckBox("Enable OCR fallback for scanned PDFs (requires Tesseract)"); self.ocr_checkbox.setChecked(False)
        if not OCR_AVAILABLE: self.ocr_checkbox.setEnabled(False); self.ocr_checkbox.setToolTip("Install pytesseract + Tesseract to enable OCR")
        right_layout.addWidget(self.ocr_checkbox)

        # Confidence threshold slider
        thr_row = QHBoxLayout(); thr_row.addWidget(QLabel("Confidence threshold:"))
        self.conf_slider = QSlider(Qt.Horizontal); self.conf_slider.setMinimum(40); self.conf_slider.setMaximum(95); self.conf_slider.setValue(int(DEFAULT_CONF_THRESHOLD*100))
        self.conf_slider.setTickInterval(5); self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_label = QLabel(f"{DEFAULT_CONF_THRESHOLD:.2f}")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100:.2f}"))
        thr_row.addWidget(self.conf_slider); thr_row.addWidget(self.conf_label)
        right_layout.addLayout(thr_row)

        self.train_btn = QPushButton("Train Classifier"); self.train_btn.clicked.connect(self._open_train_dialog); right_layout.addWidget(self.train_btn)
        right_layout.addSpacing(6)
        self.duplicates_btn = QPushButton("Find & Move Duplicates"); self.duplicates_btn.clicked.connect(self._start_duplicates_worker); right_layout.addWidget(self.duplicates_btn)
        self.organize_btn = QPushButton("Organize Now"); self.organize_btn.clicked.connect(self._start_organize_worker); right_layout.addWidget(self.organize_btn)

        right_layout.addSpacing(10)
        right_layout.addWidget(QLabel("Scheduled Auto-cleanup:"))
        sched = QHBoxLayout(); self.schedule_checkbox = QCheckBox("Enable schedule"); self.schedule_spin = QSpinBox(); self.schedule_spin.setRange(1,7*24*60); self.schedule_spin.setValue(60)
        sched.addWidget(self.schedule_checkbox); sched.addWidget(QLabel("Every (min):")); sched.addWidget(self.schedule_spin); right_layout.addLayout(sched)
        self.start_schedule_btn = QPushButton("Start Schedule"); self.start_schedule_btn.clicked.connect(self._start_schedule)
        self.stop_schedule_btn = QPushButton("Stop Schedule"); self.stop_schedule_btn.clicked.connect(self._stop_schedule)
        right_layout.addWidget(self.start_schedule_btn); right_layout.addWidget(self.stop_schedule_btn)

        self.progress = QProgressBar(); self.progress.setValue(0); right_layout.addWidget(self.progress)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(180); right_layout.addWidget(self.log)
        self.analytics_btn = QPushButton("View Analytics"); self.analytics_btn.clicked.connect(self._show_analytics); right_layout.addWidget(self.analytics_btn)
        right_layout.addStretch()
        right_w.setLayout(right_layout); split.addWidget(right_w)

        root.addWidget(split)
        footer = QHBoxLayout(); footer.addWidget(QLabel("Tip: Test on a small folder first. Duplicates are moved to source/Duplicates.")); footer.addStretch(); root.addLayout(footer)
        self.setLayout(root)
        self._apply_theme()

    # Theme
    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme_btn.setText("Switch to Dark Mode" if not self.dark_mode else "Switch to Light Mode")
        self._apply_theme()

    def _apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget { background-color: #252525; color: #ECECEC; font-family: 'Segoe UI'; }
                QLineEdit, QTextEdit, QListWidget { background-color: #2E2E2E; border: 1px solid #444; color: #ECECEC; }
                QPushButton { background-color: #D4AF37; color: black; font-weight: bold; border-radius: 6px; padding: 6px; }
                QPushButton:hover { background-color: #FFD700; }
            """)
        else:
            self.setStyleSheet("""
                QWidget { background-color: #F5F5F5; color: #111111; font-family: 'Segoe UI'; }
                QLineEdit, QTextEdit, QListWidget { background-color: #FFFFFF; border: 1px solid #CCC; color: #111111; }
                QPushButton { background-color: #2E86AB; color: white; font-weight: bold; border-radius: 6px; padding: 6px; }
                QPushButton:hover { background-color: #1F6B92; }
            """)

    # UI helpers & preview
    def _pick_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder: self.src_input.setText(folder)

    def _pick_dest(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        if folder: self.dst_input.setText(folder)

    def _log(self, text):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); self.log.append(f"[{ts}] {text}")

    def _scan(self):
        src = self.src_input.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid source", "Please choose a valid source folder."); return
        self.file_list.clear()
        files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        for fname in files:
            full = os.path.join(src, fname)
            item = QListWidgetItem(fname)
            ext = fname.split(".")[-1].lower()
            if ext in IMAGE_EXTS:
                pix = QPixmap(full)
                if not pix.isNull():
                    item.setIcon(QIcon(pix.scaled(80,80,Qt.KeepAspectRatio,Qt.SmoothTransformation)))
            elif ext == "pdf":
                png = get_pdf_thumbnail_bytes(full, zoom=0.3)
                if png:
                    pm = QPixmap(); pm.loadFromData(png); item.setIcon(QIcon(pm.scaled(80,80,Qt.KeepAspectRatio,Qt.SmoothTransformation)))
            self.file_list.addItem(item)
        self._log(f"Scanned {len(files)} files in {src}.")

    def _open_preview_popup_from_item(self, item):
        try:
            fname = item.text(); src = self.src_input.text().strip(); full = os.path.join(src, fname)
            if os.path.isfile(full): dlg = PreviewDialog(full, parent=self); dlg.exec_()
        except Exception as e:
            self._log(f"Preview popup error: {e}")

    def _open_preview_popup(self):
        it = self.file_list.currentItem()
        if not it: QMessageBox.information(self,"Preview","Select a file first."); return
        fname = it.text(); src = self.src_input.text().strip(); full = os.path.join(src, fname)
        if os.path.isfile(full): dlg = PreviewDialog(full, parent=self); dlg.exec_()

    def _preview_selected(self, item=None):
        try:
            if item is None:
                item = self.file_list.currentItem()
            if not item or not hasattr(item, "text"): return
            fname = item.text(); src = self.src_input.text().strip()
            if not src: return
            full = os.path.join(src, fname)
            if not os.path.isfile(full):
                self.preview_image.clear(); self.preview_text.setPlainText("File not found."); return
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTS:
                pm = QPixmap(full)
                if not pm.isNull():
                    self.preview_image.setPixmap(pm.scaled(self.preview_image.width(), self.preview_image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.preview_text.clear()
                else:
                    self.preview_image.clear(); self.preview_text.setPlainText("Cannot load image preview.")
                return
            if ext == ".pdf":
                png = get_pdf_thumbnail_bytes(full, zoom=0.9)
                if png:
                    pm = QPixmap(); pm.loadFromData(png)
                    if not pm.isNull():
                        self.preview_image.setPixmap(pm.scaled(self.preview_image.width(), self.preview_image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        self.preview_text.clear(); return
                txt = extract_pdf_text_firstpage(full) or ""
                if (not txt.strip()) and OCR_AVAILABLE and self.ocr_checkbox.isChecked():
                    txt = ocr_pdf_firstpage(full) or ""
                if txt:
                    self.preview_image.clear(); self.preview_text.setPlainText(txt[:2000]); return
                self.preview_image.clear(); self.preview_text.setPlainText("No preview available for this PDF."); return
            if ext in [".txt", ".csv", ".log", ".json"]:
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(4000)
                    self.preview_image.clear(); self.preview_text.setPlainText(content); return
                except Exception as e:
                    self.preview_image.clear(); self.preview_text.setPlainText(f"Text preview error: {e}"); return
            self.preview_image.clear(); self.preview_text.setPlainText("Preview not available for this file type. Double-click to open full preview window.")
        except Exception as e:
            self._log(f"Preview error: {e}")

    # Worker control
    def _start_duplicates_worker(self):
        src = self.src_input.text().strip()
        if not src or not os.path.isdir(src): QMessageBox.warning(self,"Invalid source","Please choose a valid source folder."); return
        if self.worker and self.worker.isRunning(): QMessageBox.information(self,"Worker running","A task is already running."); return
        self.progress.setValue(0)
        self.worker = WorkerThread("duplicates", src, None, {}, classifier_bundle=self.classifier_bundle)
        self.worker.progress.connect(self.progress.setValue); self.worker.message.connect(self._log)
        self.worker.finished.connect(self._duplicates_finished); self.worker.start(); self._log("Started duplicate scanning...")

    def _duplicates_finished(self, result):
        moved = result.get("moved",0); bytes_moved = result.get("bytes_moved",0); scanned = result.get("scanned",0)
        self._log(f"Duplicate scan finished. Scanned {scanned} files, moved {moved} duplicates ({bytes_moved} bytes)."); self.progress.setValue(0); self._scan()

    def _start_organize_worker(self):
        src = self.src_input.text().strip(); dst = self.dst_input.text().strip()
        if not src or not os.path.isdir(src): QMessageBox.warning(self,"Invalid source","Please choose a valid source folder."); return
        if not dst: QMessageBox.warning(self,"Invalid destination","Please choose a destination folder."); return
        if self.worker and self.worker.isRunning(): QMessageBox.information(self,"Worker running","A task is already running."); return
        sel = self.rename_combo.currentIndex(); rename_mode = "none"
        if sel==1: rename_mode="date"
        elif sel==2: rename_mode="keyword"
        elif sel==3: rename_mode="custom"
        options = {
            "rename_mode": rename_mode,
            "keyword": self.keyword_input.text().strip(),
            "custom_fmt": self.custom_fmt_input.text().strip() or "{name}_{date}{ext}",
            "remove_duplicates_first": self.dup_move_checkbox.isChecked(),
            "pdf_categorize": self.pdf_categorize_checkbox.isChecked(),
            "pdf_use_ocr": (self.ocr_checkbox.isChecked() and OCR_AVAILABLE),
            "conf_threshold": self.conf_slider.value() / 100.0
        }
        self.progress.setValue(0)
        self.worker = WorkerThread("organize", src, dst, options, classifier_bundle=self.classifier_bundle)
        self.worker.progress.connect(self.progress.setValue); self.worker.message.connect(self._log)
        self.worker.finished.connect(self._organize_finished); self.worker.start(); self._log("Started organize worker...")

    def _organize_finished(self, result):
        moved = result.get("moved",0); bytes_dup = result.get("bytes_moved",0); scanned = result.get("scanned",0)
        self._log(f"Organize finished. Moved {moved} files. Bytes from duplicates moved: {bytes_dup}.")
        analytics_log(moved, bytes_dup); self.progress.setValue(0); self._scan()

    # Training & analytics
    def _open_train_dialog(self):
        dlg = TrainModelDialog(parent=self); dlg.exec_()
        bundle = load_model()
        if bundle:
            self.classifier_bundle = bundle; self._log("Reloaded classifier model after training.")

    def _show_analytics(self):
        dlg = AnalyticsWindow(parent=self); dlg.exec_()

    # Cloud connect actions (placeholders)
    def _connect_gdrive(self):
        try:
            service = gdrive_get_service_interactive()
            if service:
                QMessageBox.information(self,"Google Drive","Connected to Google Drive successfully (temporary session).")
                self._log("Google Drive: session established.")
        except Exception as e:
            QMessageBox.warning(self,"Google Drive",f"Failed to connect: {e}")

    def _connect_onedrive(self):
        try:
            res = onedrive_device_flow_interactive()
            if res and "access_token" in res:
                QMessageBox.information(self,"OneDrive","Connected to OneDrive (device flow).")
                self._log("OneDrive: token acquired.")
            else:
                QMessageBox.information(self,"OneDrive","Device flow returned: " + str(res))
        except Exception as e:
            QMessageBox.warning(self,"OneDrive",f"Failed to connect: {e}")

    # Scheduling
    def _start_schedule(self):
        if not self.schedule_checkbox.isChecked(): QMessageBox.information(self,"Enable schedule","Check 'Enable schedule' to use scheduling."); return
        minutes = int(self.schedule_spin.value()); self.schedule_timer.start(minutes*60*1000); self._log(f"Scheduled auto-cleanup every {minutes} minutes.")
    def _stop_schedule(self):
        if self.schedule_timer.isActive(): self.schedule_timer.stop(); self._log("Stopped scheduled auto-cleanup."); 
        else: self._log("No active schedule to stop.")
    def _scheduled_run(self):
        if not (self.src_input.text().strip() and self.dst_input.text().strip()): self._log("Scheduled run skipped — invalid source/destination."); return
        self._log("Scheduled run starting..."); self._start_organize_worker()

    # Preview popup
    def _open_preview_popup(self):
        it = self.file_list.currentItem()
        if not it: QMessageBox.information(self,"Preview","Select a file first."); return
        fname = it.text(); src = self.src_input.text().strip(); full = os.path.join(src, fname)
        if os.path.isfile(full): dlg = PreviewDialog(full, parent=self); dlg.exec_()

    # Close handling
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self,"Exit","A background task is running. Stop and exit?") == QMessageBox.Yes:
                try: self.worker.terminate()
                except Exception: pass
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# ---------------------------
# Main
def main():
    app = QApplication(sys.argv)
    window = ELSmartFileOrganizerUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
