import os

SOURCE_OPEN = r"C:\Users\Het Desai\Downloads\archive\open_eye"

extensions = {}
for f in os.listdir(SOURCE_OPEN):
    ext = os.path.splitext(f)[1].lower()
    extensions[ext] = extensions.get(ext, 0) + 1

print("File types found:")
for ext, count in extensions.items():
    print(f"  {ext}: {count}")