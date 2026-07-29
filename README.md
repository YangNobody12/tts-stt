# เครื่องมือดาวน์โหลดและจัดการโมเดล/ชุดข้อมูล TTS-STT (Model & Dataset Loader CLI)

เครื่องมือ CLI ขนาดเล็กแต่ทรงพลัง สำหรับเลือก ดาวน์โหลด และจัดการโมเดล/ชุดข้อมูลเสียงภาษาไทย (Speech-to-Text: STT, Text-to-Speech: TTS และ Neural Audio Codec) จาก **Unsloth**, **SNAC** และ **Hugging Face**

---

## 🚀 คุณสมบัติเด่น (Features)

- **เครื่องมือจัดการโมเดล (`load_model.py` / `load_model.sh`)**:
  - **Default STT Model**: `unsloth/whisper-large-v3`
  - **Default TTS Model**: `unsloth/orpheus-3b-0.1-ft`
  - **Default SNAC Audio Codec**: `hubertsiuzdak/snac_24khz`
  - เมนูโต้ตอบผ่าน Terminal เลือกโมเดลผ่านตัวเลข `[1-9]` หรือใส่ชื่อ Repository ID ได้โดยตรง
- **เครื่องมือจัดการชุดข้อมูลและ Metrics (`load_dataset.py` / `load_dataset.sh`)**:
  - **Default Dataset**: `Thanarit/Thai-Voice-Test7` (ชุดข้อมูลเสียงภาษาไทย)
  - **Pre-download Metrics**: ดาวน์โหลดเกณฑ์วัดความแม่นยำ `wer` และ `cer` ผ่าน `evaluate.load("wer")` เข้า Cache ให้อัตโนมัติ เพื่อนำไปใช้งานแบบ Offline บน Compute Node ได้ทันที
- **รองรับการสร้าง Conda Environment แบบ `--prefix`**: เหมาะสำหรับระบบ HPC ที่จำกัดโควต้าโฟลเดอร์ Home

---

## 📊 รายการโมเดลที่รองรับ (Supported Models)

| ตัวเลือก | ชื่อโมเดล (Model ID) | ประเภท | ขนาดพารามิเตอร์ | คำอธิบาย |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `unsloth/whisper-tiny` | STT | 37.8M | โมเดล Whisper ขนาดเล็กที่สุด เร็วที่สุด |
| **2** | `unsloth/whisper-base` | STT | 72.6M | โมเดลขนาดเบา เหมาะกับการถอดความที่ต้องการความเร็ว |
| **3** | `unsloth/whisper-small` | STT | 0.2B | สมดุลระหว่างความแม่นยำและความเร็ว |
| **4** | `unsloth/whisper-large-v3-turbo` | STT | 0.8B | โมเดลถอดความความแม่นยำสูงที่ได้รับการเพิ่มความเร็ว |
| **5** | `unsloth/whisper-large-v3` | STT **(Default)** | 2B | โมเดล Whisper ที่มีความแม่นยำสูงที่สุด (ค่าเริ่มต้น STT) |
| **6** | `unsloth/orpheus-3b-0.1-ft` | TTS **(Default)** | 3B | โมเดล Orpheus 3B Fine-tuned สำหรับงานเสียง (ค่าเริ่มต้น TTS) |
| **7** | `hubertsiuzdak/snac_24khz` | Codec **(Default)** | 24kHz | Multi-scale neural audio codec สำหรับแปลง/สังเคราะห์สัญญาณเสียง (Default SNAC) |
| **8** | *Custom Repo ID* | กำหนดเอง | - | ระบุชื่อ Hugging Face Repository ID ใดก็ได้ |
| **9** | *All Defaults* | STT+TTS+SNAC | 2B+3B+Codec | โหลดโมเดลเริ่มต้นทั้ง 3 ตัวพร้อมกัน (`whisper-large-v3`, `orpheus`, `snac_24khz`) |

---

## 🐍 การสร้าง Conda Environment ด้วย `--prefix` (สำหรับ LANTA HPC)

การสร้าง Conda Environment ไว้ในโฟลเดอร์โปรเจกต์ด้วยตัวเลือก `--prefix` ช่วยหลีกเลี่ยงปัญหาพื้นที่โควต้าบน Home Directory เต็ม:

```bash
# 1. โหลดโมดูล Mamba บน LANTA HPC
module load Mamba/23.11.0-0

# 2. สร้าง Conda Environment ด้วย --prefix ในโฟลเดอร์โปรเจกต์ (เช่น ./env)
conda create --prefix ./env python=3.10 -y

# 3. เปิดใช้งาน Environment ผ่าน Prefix Path
conda activate ./env

# 4. ติดตั้ง PyTorch CUDA และ Dependencies ทั้งหมดลงใน --prefix env
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🏛️ ขั้นตอนการใช้งานบน LANTA HPC (NSTDA Supercomputer)

### ขั้นตอนที่ 1: ดาวน์โหลดบน Login Node (มีอินเทอร์เน็ต)
```bash
chmod +x load_model.sh load_dataset.sh

# เปิดใช้งาน Environment
conda activate ./env

# ดาวน์โหลดโมเดลเริ่มต้นทั้งหมด (Whisper Large v3, Orpheus 3B, SNAC 24kHz)
./load_model.sh --download-only

# ดาวน์โหลดชุดข้อมูลเริ่มต้น (Thanarit/Thai-Voice-Test7) และ Metrics (WER/CER)
./load_dataset.sh --download-only
```

### ขั้นตอนที่ 2: ส่ง Slurm GPU Job ไปยัง Compute Node
```bash
sbatch submit_lanta.sh
```

### ขั้นตอนที่ 3: ตรวจสอบ Log และค้นหา GPU Node ID
```bash
tail -f tts_finetune_*.log
```

### ขั้นตอนที่ 4: เชื่อมต่อ Jupyter Lab ผ่าน SSH Tunneling (Port Forwarding)

**โครงสร้างคำสั่ง:**
```bash
ssh -L 8888:gpu-id:8888 userxxx@lanta.nstda.or.th
```

**ตัวอย่างการใช้งาน:**
```bash
ssh -L 8888:x1001c2s2b0n0:8888 userxxx@lanta.nstda.or.th
```
> หลังทำ SSH Tunnel สามารถเปิดเบราว์เซอร์บนเครื่องของท่านไปที่ `http://localhost:8888` เพื่อใช้งาน Jupyter Lab ได้ทันที

---

## 📄 ใบอนุญาต (License)

สัญญาอนุญาต MIT License ดูรายละเอียดได้ที่ [LICENSE](LICENSE)