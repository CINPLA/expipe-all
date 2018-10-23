import ctypes
from ctypes.util import find_library

# OpenGL fix (must be set before other imports)
libGL = find_library("GL")
ctypes.CDLL(libGL, ctypes.RTLD_GLOBAL)

import sys
import re
import os
import json
from collections import OrderedDict

from PyQt5.QtCore import Qt, pyqtProperty, QObject, QUrl, pyqtSignal, pyqtSlot, QRegularExpression, QByteArray, QStandardPaths, QAbstractListModel, QModelIndex
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngine import QtWebEngine
from PyQt5.QtNetwork import QNetworkReply, QNetworkRequest, QNetworkAccessManager, QNetworkDiskCache
from PyQt5.QtQml import qmlRegisterType, QQmlComponent, QQmlApplicationEngine, QQmlNetworkAccessManagerFactory

from expipe import settings
import expipe

streams = []

import time

class StreamHandler(QObject):
    def __init__(self, eventSource, path):
        super().__init__(eventSource)
        self.open = True
        print("Setting up stream")
        self.stream = expipe.core.db.child(path).stream(self.handle_message)
        print("Stream ready")
        self.eventSource = eventSource
        streams.append(self.stream)

    def handle_message(self, message):
        if not open:
            print("ERROR: Received message after close!")
        if message["event"] == "put":
            print("Received message")
            print("Path", message["path"])
            if message["data"] is None:
                print("Data", "None")
            else:
                print("Data", len(message["data"]))
            self.eventSource.put(message["path"], message["data"])

    def close(self):
        streams.remove(self.stream)
        try:
            print("Closing stream...")
            self.stream.close()
            print("Stream closed")
        except AttributeError:
            print("Warning: Trouble closing stream. This is normal.")
        self.open = False


class EventSource(QAbstractListModel):
    path_changed = pyqtSignal("QString", name="pathChanged")

    key_role = Qt.UserRole + 1
    contents_role = Qt.UserRole + 2
    
    def __init__(self, parent=None):
        super().__init__(parent)

        self._path = ""
        self.contents = {}
        self.stream_handler = None
        self._manager = QNetworkAccessManager(self)

    def __del__(self):
        print("Got deleted...")

    def refresh(self):
        for key in self.contents:
            self.contents[key]["id"] = key

    def put(self, path, data):
        path = path.split("/")
        if path[0] == "":
            del(path[0])
        if path[-1] == "":
            del(path[-1])
        if len(path) == 0:
            print(self._path, "Remove all rows", self.rowCount(), len(self.contents))
            self.beginRemoveRows(QModelIndex(), 0, self.rowCount() - 1)
            self.contents = OrderedDict({})
            self.endRemoveRows()
            if data is not None:
                print(self._path, "Insert all rows", len(data))
                self.beginInsertRows(QModelIndex(), 0, len(data) - 1)
                self.contents = OrderedDict(data)
                self.endInsertRows()
        else:
            changed_key = path[0]
            key_list = list(self.contents.keys())
            if changed_key in key_list:
                index = key_list.index(changed_key)
                model_index = self.index(index, 0)
                if len(path) == 1 and data is None:
                    print(self._path, "remove one row", index)
                    self.beginRemoveRows(QModelIndex(), index, index)
                    self.set_nested(path, data)
                    self.endRemoveRows()
                else:
                    print(self._path, "change one row", index)
                    self.set_nested(path, data)
                    self.dataChanged.emit(model_index, model_index)
            else:
                print(self._path, "insert one row")
                self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
                self.set_nested(path, data)
                self.endInsertRows()
        self.refresh()

    def reconnect(self, url):
        print("Reconnecting")
        if self._reply is not None:
            self._reply.abort()
            self._reply = None
        request = QNetworkRequest(url)
        request.setRawHeader(b"Accept", b"text/event-stream")
        self._reply = self._manager.get(request)
        self._reply.readyRead.connect(self.processReadyRead)
        self._reply.finished.connect(self.processFinished)

    @pyqtProperty('QString')
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if path == self._path:
            return
        print("Change path from", self._path, "to", path)
        self.put("/", {})
        if self.stream_handler:
            self.stream_handler.close()
        self.stream_handler = StreamHandler(self, path)
        self._path = path
        self.path_changed.emit(path)

    def data(self, index=QModelIndex(), role=0):
        if role == self.key_role:
            key_list = list(self.contents.keys())
            return key_list[index.row()]
        elif role == self.contents_role:
            value_list = list(self.contents.values())
            return value_list[index.row()]
        else:
            return QVariant()

    def rowCount(self, index=QModelIndex()):
        return len(self.contents)

    def roleNames(self):
        return {
            self.key_role: b"key",
            self.contents_role: b"contents"
        }

    def set_nested(self, path, value):
        dic = self.contents
        for key in path[:-1]:
            dic = dic.setdefault(key, {})
        if value is None:
            del(dic[path[-1]])
        else:
            dic[path[-1]] = value
            

class Clipboard(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    @pyqtSlot(str)
    def setText(self, text):
        QApplication.clipboard().setText(text)

class NetworkAccessManagerFactory(QQmlNetworkAccessManagerFactory):
    def create(self, parent):
        nam = QNetworkAccessManager(parent)
        cache = QNetworkDiskCache(parent)
        cache_dir = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
        cache_subdir = os.path.join(cache_dir, "network")
        print("Cache dir:", cache_subdir)
        cache.setCacheDirectory(cache_subdir)
        nam.setCache(cache)
        return nam

if __name__ == "__main__":
    app = QApplication(sys.argv)
    qmlRegisterType(EventSource, "ExpipeBrowser", 1, 0, "EventSource")
    qmlRegisterType(Clipboard, "ExpipeBrowser", 1, 0, "Clipboard")
    QApplication.setOrganizationName("Cinpla")
    QApplication.setApplicationName("Expipe Browser")
    QtWebEngine.initialize()
    engine = QQmlApplicationEngine()
    engine.setNetworkAccessManagerFactory(NetworkAccessManagerFactory())
    engine.load(QUrl("main.qml"))

    app.exec_()
    for stream in streams:
        print(stream)
        stream.close()

