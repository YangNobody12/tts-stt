# เครื่องมือดาวน์โหลดโมเดลและชุดข้อมูล TTS-STT ด้วย `huggingface-cli`

เครื่องมือดาวน์โหลดโมเดลและชุดข้อมูลสำหรับงาน **Speech-to-Text (STT)**, **Text-to-Speech (TTS)** และ **Neural Audio Codec** จาก **Hugging Face** ออกแบบมาเป็นพิเศษสำหรับใช้งานบน **`transfer.lanta.nstda.or.th`** (Transfer Node บน LANTA HPC) โดยใช้ `huggingface-cli download` โดยตรง **ไม่ต้องใช้ Python runtime หรือติดตั้ง PyTorch/CUDA ในขั้นตอนดาวน์โหลด**

---

## 🚀 คุณสมบัติเด่น (Features)

- **ใช้ `huggingface-cli download` โดยตรง**: น้ำหนักเบา รวดเร็ว ดาวน์โหลดผ่าน CLI ไม่ต้องโหลดโมเดลลง RAM/VRAM
- **เหมาะสำหรับ LANTA Transfer Node (`transfer.lanta.nstda.or.th`)**: ไม่ต้องรัน Python scripts ขนาดใหญ่ หรือกังวลเรื่อง CUDA/PyTorch dependencies บน Transfer Node
- **เครื่องมือดาวน์โหลดโมเดล (`load_model.sh`)**:
  - **Default STT Model**: `unsloth/whisper-large-v3`
  - **Default TTS Model**: `unsloth/orpheus-3b-0.1-ft`
  - **Default SNAC Audio Codec**: `hubertsiuzdak/snac_24khz`
  - สามารถดาวน์โหลด Custom Model โดยระบุ Repo ID: `./load_model.sh username/model-name`
- **เครื่องมือดาวน์โหลดชุดข้อมูล (`load_dataset.sh`)**:
  - **Default Dataset**: `Thanarit/Thai-Voice-Test7` (ชุดข้อมูลเสียงภาษาไทย)
  - สามารถดาวน์โหลด Custom Dataset โดยระบุ Repo ID: `./load_dataset.sh username/dataset-name`
- **จัดการ Cache อัตโนมัติในโปรเจกต์ (`./hf_cache`)**: หลีกเลี่ยงปัญหาโควต้า Home Directory บน LANTA เต็ม

---

## 📊 รายการโมเดลและชุดข้อมูลเริ่มต้น (Default Repos)

| ประเภท | ชื่อ Repository (Repo ID) | คำอธิบาย |
| :---: | :--- | :--- |
| **STT Model** | `unsloth/whisper-large-v3` | โมเดล Whisper Large v3 สำหรับถอดความเสียงเป็นข้อความ |
| **TTS Model** | `unsloth/orpheus-3b-0.1-ft` | โมเดล Orpheus 3B Fine-tuned สำหรับงานสังเคราะห์เสียง |
| **SNAC Codec** | `hubertsiuzdak/snac_24khz` | Multi-scale neural audio codec (24kHz) |
| **Dataset** | `Thanarit/Thai-Voice-Test7` | ชุดข้อมูลเสียงภาษาไทยสำหรับเทรน TTS / STT |

---

## 🐍 การสร้าง Conda Environment บน LANTA HPC (พร้อม `jupyterlab`)

การสร้าง Conda Environment ไว้ในโฟลเดอร์โปรเจกต์ด้วย `--prefix` ช่วยหลีกเลี่ยงปัญหาพื้นที่ Home เต็ม:

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

## 🏛️ ขั้นตอนการใช้งานบน LANTA HPC (`transfer.lanta.nstda.or.th`)

### 🗂️ HuggingFace Cache Configuration (ตั้งค่า Cache Path)

shell scripts จะตั้งค่า cache ให้ชี้มาที่ **`./hf_cache`** ภายในโฟลเดอร์โปรเจกต์โดยอัตโนมัติ:

```
<project-dir>/
└── hf_cache/              ← HuggingFace cache ทั้งหมด (ตั้งค่าโดยอัตโนมัติ)
    ├── hub/               ← โมเดล และ ชุดข้อมูล (HUGGINGFACE_HUB_CACHE)
    │   ├── models--unsloth--whisper-large-v3/
    │   ├── models--unsloth--orpheus-3b-0.1-ft/
    │   ├── models--hubertsiuzdak--snac_24khz/
    │   └── datasets--Thanarit--Thai-Voice-Test7/
    └── datasets/          ← HF datasets cache
```

---

### ขั้นตอนที่ 1: ดาวน์โหลดบน Transfer / Login Node (`transfer.lanta.nstda.or.th`)

เชื่อมต่ออินเทอร์เน็ตที่ Transfer Node แล้วสั่งดาวน์โหลดไฟล์ผ่าน CLI:

```bash
chmod +x load_model.sh load_dataset.sh

# 1.1 ดาวน์โหลดโมเดลเริ่มต้นทั้งหมด (Whisper Large v3, Orpheus 3B, SNAC 24kHz)
./load_model.sh

# หรือดาวน์โหลดโมเดลเฉพาะตามต้องการ:
./load_model.sh username/model-name

# 1.2 ดาวน์โหลดชุดข้อมูลเริ่มต้น (Thanarit/Thai-Voice-Test7)
./load_dataset.sh

# หรือดาวน์โหลดชุดข้อมูลเฉพาะตามต้องการ:
./load_dataset.sh username/dataset-name
```

---

### ขั้นตอนที่ 2: ใช้คำสั่ง `huggingface-cli` โดยตรง (ทางเลือกเพิ่มเติม)

ท่านสามารถสั่งงาน `huggingface-cli download` ได้เองจาก Terminal:

```bash
# กำหนด HF Cache เข้าโฟลเดอร์โปรเจกต์
export HF_HOME="$(pwd)/hf_cache"

# ดาวน์โหลดโมเดล
huggingface-cli download username/model-name

# ดาวน์โหลด dataset
huggingface-cli download --repo-type dataset username/dataset-name
```

---

### ขั้นตอนที่ 3: ส่ง Slurm GPU Job ไปยัง Compute Node

```bash
sbatch submit_lanta.sh
```

---

### ขั้นตอนที่ 4: เชื่อมต่อ Jupyter Lab ผ่าน SSH Tunneling (Port Forwarding)

**โครงสร้างคำสั่ง:**
```bash
ssh -L 8888:gpu-id:8888 userxxx@lanta.nstda.or.th
```

**ตัวอย่างการใช้งาน:**
```bash
ssh -L 8888:x1001c2s2b0n0:8888 userxxx@lanta.nstda.or.th
```
> หลังเปิด SSH Tunnel สามารถเปิดเบราว์เซอร์ไปที่ `http://localhost:8888` เพื่อเปิดใช้งาน Jupyter Lab บน GPU Compute Node ได้ทันที

---

## 📄 ใบอนุญาต (License)

สัญญาอนุญาต MIT License ดูรายละเอียดได้ที่ [LICENSE](LICENSE)