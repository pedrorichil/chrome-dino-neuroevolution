"""Automated build script for compiling Chrome Dino AI into standalone Windows executable and release zip."""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    dist_dir = root / "dist"
    build_dir = root / "build"
    icon_path = root / "assets" / "icon.ico"

    print("=" * 60)
    print(" Compilando Chrome Dino AI (Neuroevolution Engine)")
    print("=" * 60)

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name", "ChromeDinoAI",
        "--add-data", f"assets{os.pathsep}assets",
        "--add-data", f"models{os.pathsep}models",
        "--exclude-module", "scipy",
        "--exclude-module", "pyarrow",
        "--exclude-module", "pandas",
        "--exclude-module", "sqlalchemy",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "--exclude-module", "tkinter",
    ]

    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])

    # Standalone folder distribution (fast startup, 100% reliable for Pygame audio & assets)
    cmd.extend([
        "--onedir",
        str(root / "main.py")
    ])

    print(f"Executando comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=root)

    if result.returncode != 0:
        print("\n[Erro] Falha na compilação do executável com PyInstaller.")
        sys.exit(result.returncode)

    print("\n[Sucesso] Executável compilado em 'dist/ChromeDinoAI/'!")

    # Package into release zip for GitHub Releases
    release_zip = dist_dir / "ChromeDinoAI_Windows_v1.0.zip"
    app_folder = dist_dir / "ChromeDinoAI"

    if app_folder.exists():
        # Create convenient double-click shortcuts inside distribution folder
        (app_folder / "1 - Treinar 2000 Dinos.bat").write_text("@echo off\nstart ChromeDinoAI.exe\n", encoding="utf-8")
        (app_folder / "2 - Assistir Dinossauro Campeao.bat").write_text("@echo off\nstart ChromeDinoAI.exe --mode evaluation\n", encoding="utf-8")
        (app_folder / "3 - Duelo Humano vs IA.bat").write_text("@echo off\nstart ChromeDinoAI.exe --mode versus\n", encoding="utf-8")
        (app_folder / "4 - Treinar Partindo do Campeao.bat").write_text("@echo off\nstart ChromeDinoAI.exe --seed-best\n", encoding="utf-8")

        print(f"Criando pacote zip para GitHub Release: {release_zip.name}...")
        with zipfile.ZipFile(release_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in app_folder.rglob("*"):
                arcname = file_path.relative_to(dist_dir)
                zipf.write(file_path, arcname)

        print(f"\n[Pronto] Pacote de Release criado com sucesso em: {release_zip}")
        print(f"Tamanho do arquivo: {release_zip.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    build()
