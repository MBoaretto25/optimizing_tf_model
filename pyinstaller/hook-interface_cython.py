from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    "tensorflow",
    "tkinter",
    "keras",
    "cv2",
 ]
hiddenimports += collect_submodules('../..')
