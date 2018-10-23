import QtQuick 2.4
import QtQuick.Controls 1.4 as QQC1
import QtQuick.Controls 1.4
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1

import ExpipeBrowser 1.0

import "."

import "md5.js" as MD5
import "dicthelper.js" as DictHelper

Rectangle {
    id: root

    property var experimentData
    property string currentProject

    property var modules: {
        return {}
    }

    color: "#fdfdfd"
    border {
        color: "#dedede"
        width: 1
    }

    function refreshModules(path) {
        if(path.length < 1) {
            return
        }
        for(var i = 0; i < moduleView.count; i++) {
            var dictEditor = moduleView.itemAt(i)
            if(dictEditor.key === path[0]) {
                dictEditor.refreshPath(path)
            }
        }
    }

    function refreshMessages(path) {
        if(path.length < 1) {
            return
        }
        for(var i = 0; i < messageView.count; i++) {
            var dictEditor = messageView.itemAt(i)
            if(dictEditor.key === path[0]) {
                dictEditor.refreshPath(path)
            }
        }
    }

    EventSource {
        id: moduleEventSource
        path: experimentData ? "action_modules/" + currentProject + "/" + experimentData.__key : ""

        onPutReceived: {
            refreshModules(path)
        }
        onPatchReceived: {
            refreshModules(path)
        }
    }

    EventSource {
        id: messagesEventSource
        path: experimentData ? "action_messages/" + currentProject + "/" + experimentData.__key : ""

        onPutReceived: {
            refreshMessages(path)
        }
        onPatchReceived: {
            refreshMessages(path)
        }
    }

    Clipboard {
        id: clipboard
    }

    Flickable {
        anchors.fill: parent
        contentHeight: container.height + 360
        // ScrollBar.vertical: ScrollBar {}

        Row {
            anchors {
                right: parent.right
                top: parent.top
                rightMargin: 48
                topMargin: 96
            }
            Button {
                id: codeButton
                property string snippet: "import expipe\n" +
                                         "project = expipe.get_project('" + experimentData.project+ "')\n" +
                                         "action = project.require_action('" + experimentData.__key + "')\n" +
                                         "# continue working with action"
                text: "Copy Python code"
                onClicked: {
                    clipboard.setText(snippet)
                    codePopup.open()
                }
            }

            Button {
                text: "Delete action"
                onClicked: {
                    deleteDialog.open()
                }
            }
        }

        Dialog {
            id: deleteDialog
            title: "Are you sure?"
            standardButtons: Dialog.Cancel | Dialog.Ok
            onAccepted: {
                Firebase.remove("actions/" + currentProject + "/" + experimentData.__key, function(reply) {
                    console.log("Removed action and got reply", reply.responseText)
                })
                Firebase.remove("action_modules/" + currentProject + "/" + experimentData.__key, function(reply) {
                    console.log("Removed action modules and got reply", reply.responseText)
                })
                Firebase.remove("action_messages/" + currentProject + "/" + experimentData.__key, function(reply) {
                    console.log("Removed action messages and got reply", reply.responseText)
                })
            }
        }

        Column {
            id: container
            anchors {
                left: parent.left
                right: parent.right
                top: parent.top
                topMargin: 96
            }

            spacing: 12

            Row {
                x: 140
                spacing: 20

                Identicon {
                    id: image
                    width: 64
                    height: 64
                    action: experimentData
                }

                Label {
                    anchors {
                        bottom: image.bottom
                    }
                    font.pixelSize: 24
                    font.weight: Font.Light
                    text: experimentData["__key"]
                }
            }

            Item {
                width: 1
                height: 24
            }

            ExperimentEdit {
                experimentData: root.experimentData
                property: "type"
                text: "Action type"
            }

            ExperimentEdit {
                experimentData: root.experimentData
                property: "location"
                text: "Location"
            }

            ExperimentEdit {
                experimentData: root.experimentData
                property: "datetime"
                text: "Date and time"
            }

            FirebaseListEdit {
                experimentData: root.experimentData
                property: "users"
                text: "Experimenters"
            }

            FirebaseListEdit {
                experimentData: root.experimentData
                property: "subjects"
                text: "Subjects"
            }

            FirebaseListEdit {
                experimentData: root.experimentData
                property: "tags"
                text: "Tags"
            }

            Item {
                width: 1
                height: 24
            }

            RowLayout {
                anchors {
                    left: parent.left
                    right: parent.right
                    leftMargin: 100
                    rightMargin: 48
                }

                Label {
                    id: messageTitle
                    font.pixelSize: 24
                    font.weight: Font.Light
                    color: "#434343"
                    horizontalAlignment: Text.AlignRight
                    text: "Messages"
                }

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                Button {
                    id: addMessageButton
                    text: "Add message"
                    onClicked: {
                        newMessageDialog.open()
                    }
                }
            }

            Label {
                id: messagesLoadingText
                x: 100
                visible: messagesEventSource.status != EventSource.Connected
                color: "#ababab"
                text: {
                    switch(messagesEventSource.status) {
                    case EventSource.Connecting:
                        return "Loading..."
                    case EventSource.Disconnected:
                        return "Error loading modules!"
                    }
                    return ""
                }
            }

            Label {
                x: 100
                visible: !messagesLoadingText.visible && messageView.count < 1
                color: "#ababab"
                text: "No messages"
            }

            Dialog {
                id: newMessageDialog
                standardButtons: Dialog.Ok | Dialog.Cancel
                Column {
                    anchors {
                        left: parent.left
                        right: parent.right
                    }

                    Label {
                        text: "Author:"
                    }

                    TextField {
                        id: authorField
                        anchors {
                            left: parent.left
                            right: parent.right
                        }
                    }

                    Label {
                        text: "Message:"
                    }

                    TextField {
                        id: messageField
                        anchors {
                            left: parent.left
                            right: parent.right
                        }
                    }
                }
                onAccepted: {
                    var message = {
                        "message": messageField.text,
                        "user": authorField.text,
                        "datetime": (new Date()).toISOString()
                    }
                    Firebase.post("action_messages/" + currentProject + "/" + experimentData.__key,
                    message,
                    function (response) {
                        console.log("Posted message", response)
                    })
                }
            }

            Repeater {
                id: messageView
                model: messagesEventSource
                DictionaryEditor {
                    property string key: model.key
                    x: 100
                    keyString: index
                    contents: model.contents
                    basePath: "action_messages/" + currentProject + "/" + experimentData.__key + "/" + model.key
                    onContentsChanged: {
                        console.log("Contents changed with length", contents.length)
                    }
                }
            }

            Item {
                width: 1
                height: 24
            }

            RowLayout {
                anchors {
                    left: parent.left
                    right: parent.right
                    leftMargin: 100
                    rightMargin: 48
                }

                Label {
                    id: modulesTitle
                    font.pixelSize: 24
                    font.weight: Font.Light
                    color: "#434343"
                    horizontalAlignment: Text.AlignRight
                    text: "Modules"
                }

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                Button {
                    id: addModuleButton
                    text: "Add module"
                    onClicked: {
                        newModuleDialog.open()
                    }
                }

            }

            Label {
                id: modulesLoadingText
                x: 100
                visible: moduleEventSource.status != EventSource.Connected
                color: "#ababab"
                text: {
                    switch(moduleEventSource.status) {
                    case EventSource.Connecting:
                        return "Loading..."
                    case EventSource.Disconnected:
                        return "Error loading modules!"
                    }
                    return ""
                }
            }

            Label {
                x: 100
                visible: !modulesLoadingText.visible && moduleView.count < 1
                color: "#ababab"
                text: "No modules"
            }

            TemplateDialog {
                id: newModuleDialog
                modulesPath: "action_modules/" + currentProject + "/" + experimentData.__key
            }

            Repeater {
                id: moduleView
                model: moduleEventSource
                DictionaryEditor {
                    property string key: model.key
                    x: 100
                    keyString: model.key
                    contents: model.contents
                    basePath: "action_modules/" + currentProject + "/" + experimentData.__key + "/" + model.key
                    onContentsChanged: {
                        console.log("Contents changed with length", contents.length)
                    }
                }
            }
        }
    }


    // Popup {
    //     id: codePopup
    //     modal: true
    //     focus: true
    //     dim: true
    //     x: root.width / 2 - width / 2
    //     y: root.height / 2 - height / 2
    //     width: 320
    //     height: 180
    //     closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
    //     Label {
    //         anchors {
    //             fill: parent
    //             margins: 32
    //         }
    //
    //         text: "Code copied to clipboard\n\n" +
    //               "Paste it in a Jupyter Notebook to load the experiment."
    //         wrapMode: Text.WrapAtWordBoundaryOrAnywhere
    //     }
    // }
}
