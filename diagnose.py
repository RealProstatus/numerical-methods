import ctypes
import glob
import os
import site
import struct
import sys

print(f"=== Python {sys.version} ===")

# 1. Поиск .pyd файла во всех каталогах Python и site-packages
search_roots = [os.getcwd()] + sys.path + site.getsitepackages()
try:
  search_roots.append(site.getusersitepackages())
except Exception:
  pass

pyd_candidates = []
for root in search_roots:
  if os.path.isdir(root):
    matches = glob.glob(
        os.path.join(root, "**", "*_magnus_py*.pyd"), recursive=True
    )
    pyd_candidates.extend(matches)
    matches_direct = glob.glob(os.path.join(root, "*_magnus_py*.pyd"))
    pyd_candidates.extend(matches_direct)

pyd_candidates = list(
    dict.fromkeys([
        os.path.abspath(p) for p in pyd_candidates if os.path.isfile(p)
    ])
)

if not pyd_candidates:
  print("ОШИБКА: Файл _magnus_py*.pyd не найден ни в проекте, ни в site-packages!")
  sys.exit(1)

pyd_path = pyd_candidates[0]
print(f"Найден модуль: {pyd_path}\n")


# 2. Парсинг таблицы импорта PE (Portable Executable)
def get_pe_dependencies(filepath):
  with open(filepath, "rb") as f:
    data = f.read()

  pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
  magic = struct.unpack_from("<H", data, pe_offset + 24)[0]
  num_sections = struct.unpack_from("<H", data, pe_offset + 6)[0]
  opt_hdr_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
  import_rva_offset = pe_offset + 24 + (120 if magic == 0x20B else 104)
  import_rva = struct.unpack_from("<I", data, import_rva_offset)[0]

  sec_table = pe_offset + 24 + opt_hdr_size
  sections = []
  for i in range(num_sections):
    sec = data[sec_table + i * 40 : sec_table + (i + 1) * 40]
    vsize, va, rsize, roff = struct.unpack_from("<IIII", sec, 8)
    sections.append((va, vsize, roff, rsize))

  def rva_to_off(rva):
    for va, vsize, roff, rsize in sections:
      if va <= rva < va + max(vsize, rsize):
        return roff + (rva - va)
    return None

  deps = []
  off = rva_to_off(import_rva)
  if not off:
    return deps

  while True:
    entry = data[off : off + 20]
    if len(entry) < 20:
      break
    ilt, tm, fwd, name_rva, iat = struct.unpack_from("<IIIII", entry, 0)
    if name_rva == 0:
      break
    noff = rva_to_off(name_rva)
    if noff:
      end = data.find(b"\x00", noff)
      deps.append(data[noff:end].decode("ascii", errors="ignore"))
    off += 20
  return deps


deps = get_pe_dependencies(pyd_path)
print(f"Зависимости из PE-заголовка ({len(deps)} шт.):")
for d in deps:
  print(f"  • {d}")

# 3. Регистрация каталогов DLL из __init__.py
search_dirs = [
    r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
    r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin",
    r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
    r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib",
    r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler",
    r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin",
]
search_dirs.extend(
    glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin")
)

active_dirs = []
for p in search_dirs:
  if os.path.isdir(p):
    try:
      active_dirs.append(os.add_dll_directory(p))
      os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
    except OSError:
      pass

print(
    f"\nЗарегистрировано директорий поиска DLL: {len(active_dirs)}\nТестируем"
    " загрузку библиотек:"
)

# 4. Проверка доступности каждой библиотеки
missing = []
for d in deps:
  try:
    ctypes.WinDLL(d)
    print(f"  [OK]      {d}")
  except Exception as e:
    print(f"  [MISSING] {d} -> {e}")
    missing.append(d)

# 5. Поиск недостающих DLL на диске C:\
if missing:
  print("\nПоиск отсутствующих библиотек в Program Files...")
  for m in missing:
    found = []
    for root_search in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit",
        r"C:\Program Files (x86)\Intel",
        r"C:\Program Files\Microsoft Visual Studio",
    ]:
      if os.path.exists(root_search):
        for root, _, files in os.walk(root_search):
          if m.lower() in [f.lower() for f in files]:
            found.append(os.path.join(root, m))
    if found:
      print(f"  -> Найдена {m} в:")
      for f in found:
        print(f"       {os.path.dirname(f)}")
    else:
      print(f"  -> {m} НЕ найдена в стандартных путях.")

print("\nПробуем загрузить сам .pyd напрямую:")
try:
  ctypes.cdll.LoadLibrary(pyd_path)
  print("  [SUCCESS] _magnus_py успешно загружен!")
except Exception as e:
  print(f"  [FAILED] Ошибка загрузки: {e}")