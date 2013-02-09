import wx
from os.path import join as osjoin
from os import startfile
from core import *
import threading
from copy import copy


myEVT_CORE_UPDATE = wx.NewEventType()
EVT_CORE_UPDATE = wx.PyEventBinder(myEVT_CORE_UPDATE, 1)


class CoreUpdateEvent(wx.PyCommandEvent):
    def __init__(self, etype, eid, text=None):
        wx.PyCommandEvent.__init__(self, etype, eid)


class CoreThread(threading.Thread):
    def __init__(self, parent, text):
        threading.Thread.__init__(self)
        self._parent = parent
        self._text = text

    def run(self):
        vectorize(self._parent)
        try:
            event = CoreUpdateEvent(myEVT_CORE_UPDATE, -1, self._text,)
            wx.PostEvent(self._parent, event)
        except TypeError:
            return 0


class MainWindow(wx.Frame):
    def __init__(self, *args, **kwds):
        kwds["style"] = wx.SYSTEM_MENU
        kwds["style"] |= wx.CAPTION
        kwds["style"] |= wx.CLOSE_BOX
        kwds["style"] |= wx.MINIMIZE_BOX
        wx.Frame.__init__(self, *args, **kwds)

        self.CreateStatusBar()
        self.buttonOpen = wx.Button(self, -1, "Open")
        self.buttonShowRaster = wx.Button(self, -1, "Show Raster")
        choices = [
            "",
            "Combinatorial Hough Transform",
            "Hierarchical Hough Transform",
            "Adaptive Hough Transform"
        ]
        style = wx.CB_DROPDOWN | wx.CB_READONLY
        self.comboboxMethod = wx.ComboBox(
            self,
            -1,
            choices=choices,
            style=style
        )
        self.buttonVectorize = wx.Button(self, -1, "Vectorize")
        self.buttonShowVector = wx.Button(self, -1, "Show Vector")
        self.buttonSave = wx.Button(self, -1, "Save")

        self.__set_properties()
        self.__do_layout()
        self.__add_event_handlers()

        self.running_task = None
        self.task_is_done = False
        self.vectorizing = False

        delete_files()

    def __set_properties(self):
        self.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.SetForegroundColour(wx.Colour(255, 255, 255))

        self.__reset_everything()

    def __do_layout(self):
        boxSizer = wx.BoxSizer(wx.VERTICAL)

        gridSizer = wx.FlexGridSizer(1, 11, 0, 0)
        gridSizer.Add(self.buttonOpen, 0, 0, 0)
        gridSizer.Add((20, 20), 0, 0, 0)
        gridSizer.Add(self.buttonShowRaster, 0, 0, 0)
        gridSizer.Add((20, 20), 0, 0, 0)
        gridSizer.Add(self.comboboxMethod, 0, 0, 0)
        gridSizer.Add((20, 20), 0, 0, 0)
        gridSizer.Add(self.buttonVectorize, 0, 0, 0)
        gridSizer.Add((20, 20), 0, 0, 0)
        gridSizer.Add(self.buttonShowVector, 0, 0, 0)
        gridSizer.Add((20, 20), 0, 0, 0)
        gridSizer.Add(self.buttonSave, 0, 0, 0)

        align = wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL
        boxSizer.Add(gridSizer, 1, align, 0)

        self.SetSizer(boxSizer)
        boxSizer.Fit(self)
        self.Layout()

    def __add_event_handlers(self):
        self.Bind(wx.EVT_BUTTON, self.onButtonOpen, self.buttonOpen)
        self.Bind(
            wx.EVT_BUTTON,
            self.onButtonShowRaster,
            self.buttonShowRaster
        )
        self.Bind(wx.EVT_COMBOBOX, self.onComboboxMethod, self.comboboxMethod)
        self.Bind(wx.EVT_BUTTON, self.onButtonVectorize, self.buttonVectorize)
        self.Bind(
            wx.EVT_BUTTON,
            self.onButtonShowVector,
            self.buttonShowVector
        )
        self.Bind(wx.EVT_BUTTON, self.onButtonSave, self.buttonSave)
        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.Bind(EVT_CORE_UPDATE, self.onCoreUpdate)

    def __reset_everything(self):
        self.SetTitle("Aerial Photography Vectorization")
        self.buttonShowRaster.Enable(False)
        self.comboboxMethod.Enable(False)
        self.comboboxMethod.SetSelection(0)
        self.buttonVectorize.Enable(False)
        self.buttonShowVector.Enable(False)
        self.buttonSave.Enable(False)

        self.task_is_done = False
        self.vectorizing = False

    def __run_task(self, some_function, parameter=None):
        self.running_task = copy(some_function.__name__)
        self.__update('START')
        if not parameter:
            some_function()
        else:
            some_function(parameter)
        self.__update('STOP')

    def __update(self, type):
        if self.running_task == 'openRasterImage':
            if type == 'START':
                self.SetStatusText('Opening a raster image...')
            elif type == 'STOP':
                if self.task_is_done:
                    self.SetTitle("APV - %s" % self.raster_path)
                    self.buttonShowRaster.Enable(True)
                    self.comboboxMethod.Enable(True)
                    self.SetStatusText("The raster image has been opened.")
                else:
                    self.SetStatusText("New raster image hasn't been opened.")
        if self.running_task == 'showImage':
            if type == 'START':
                self.SetStatusText('Showing the image...')
            elif type == 'STOP':
                if self.task_is_done:
                    self.SetStatusText("The image has been shown.")
                else:
                    self.SetStatusText("The image hasn't been shown.")
        if self.running_task == 'vectorizeImage':
            if type == 'START':
                self.buttonOpen.Enable(False)
                self.comboboxMethod.Enable(False)
                self.buttonVectorize.Enable(False)
                self.buttonShowVector.Enable(False)
                self.buttonSave.Enable(False)
                self.SetStatusText("Vectorizing the image...")
            if type == 'STOP':
                if self.task_is_done:
                    self.buttonOpen.Enable(True)
                    self.comboboxMethod.Enable(True)
                    self.buttonVectorize.Enable(True)
                    self.buttonSave.Enable(True)
                    self.SetStatusText("The image has been vectorized.")
        if self.running_task == 'saveVectorImage':
            if type == 'START':
                self.SetStatusText("Saving vector image...")
            elif type == 'STOP':
                if self.task_is_done:
                    self.buttonShowVector.Enable(True)
                    self.SetStatusText("The image has been saved.")
                else:
                    self.SetStatusText("The image hasn't been saved.")

    def onButtonOpen(self, event):
        self.__run_task(self.openRasterImage)

    def onButtonShowRaster(self, event):
        self.__run_task(self.showImage, self.raster_path)

    def onComboboxMethod(self, event):
        self.chosenMethod = self.comboboxMethod.GetCurrentSelection()
        if self.chosenMethod == 0:
            self.buttonVectorize.Enable(False)
        else:
            self.buttonVectorize.Enable(True)

    def onButtonVectorize(self, event):
        if not self.vectorizing:
            self.__run_task(self.vectorizeImage)

    def onButtonShowVector(self, event):
        self.__run_task(self.showImage, self.svg_path)

    def onButtonSave(self, event):
        self.__run_task(self.saveVectorImage)

    def onClose(self, event):
        self.Destroy()

    def onCoreUpdate(self, event):
        self.task_is_done = True
        self.running_task = 'vectorizeImage'
        self.vectorizing = False
        self.__update('STOP')

    def openRasterImage(self):
        self.task_is_done = False
        dialog = wx.FileDialog(
            self,
            message="Choose a raster core",
            wildcard="BMP, PNG, JPG|*.bmp;*.png;*.jpg|"\
                     "BMP files (*.bmp)|*.bmp|"\
                     "PNG files (*.png)|*.png|"\
                     "JPG files (*.jpg)|*.jpg",
            style=wx.OPEN
        )

        if dialog.ShowModal() == wx.ID_OK:
            raster_filename = dialog.GetFilename()
            raster_format = raster_filename.split('.')[-1]
            raster_directory = dialog.GetDirectory()
            raster_path = osjoin(
                raster_directory,
                raster_filename
            )

            if open_image(raster_path):
                self.__reset_everything()
                self.raster_filename = copy(raster_filename)
                self.raster_format = copy(raster_format)
                self.raster_directory = copy(raster_directory)
                self.raster_path = copy(raster_path)
                self.task_is_done = True
            else:
                message_template = "'%s' is invalid %s image."
                message_data = (raster_path, raster_format)
                message = message_template % message_data
                self.showErrorMessage(message)

        dialog.Destroy()

    def showImage(self, path):
        self.task_is_done = False
        try:
            startfile(path)
            self.task_is_done = True
        except WindowsError:
            self.showErrorMessage('Cannot show %s' % path)

    def vectorizeImage(self):
        if not self.vectorizing:
            self.task_is_done = False
            self.thread = CoreThread(self, 1)
            self.thread.start()

    def saveVectorImage(self):
        self.task_is_done = False
        new_name = self.raster_filename.replace(self.raster_format, 'svg')
        dialog = wx.FileDialog(
            self,
            message="Save file as ...",
            defaultDir=self.raster_directory,
            defaultFile=new_name,
            wildcard="SVG file (*.svg)|*.svg",
            style=wx.SAVE | wx.OVERWRITE_PROMPT
        )

        if dialog.ShowModal() == wx.ID_OK:
            self.svg_filename = dialog.GetFilename()
            self.svg_directory = dialog.GetDirectory()
            self.svg_path = osjoin(self.svg_directory, self.svg_filename)

            if not save(self.svg_path):
                message = "Cannot save the file with name\n'%s'." % self.svg_path
                self.showErrorMessage(message)
            else:
                self.task_is_done = True

        dialog.Destroy()

    def showErrorMessage(self, message):
        message = wx.MessageDialog(self, message, "Error", wx.ICON_ERROR)
        message.ShowModal()
        message.Destroy()


class APV(wx.App):
    def OnInit(self):
        wx.InitAllImageHandlers()
        mainWindow = MainWindow(None, -1, "")
        self.SetTopWindow(mainWindow)
        mainWindow.Show()
        return 1
