import tkMessageBox
def showTkMessage():
    tkMessageBox.showerror(
        "Import error",
        "Install required components!"
    )
try:
    from gui import APV
    app = APV(0)
    app.MainLoop()
except ImportError:
    showTkMessage()
except NameError:
    showTkMessage()
