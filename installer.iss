; -----------------------------------------
; Gandhi Assistant Installer
; -----------------------------------------

#define MyAppName "Gandhi Assistant"
#define MyAppVersion "1.0"
#define MyAppPublisher "Het Patel"
#define MyAppExe "start.bat"

[Setup]
AppId={{8A8E6B8F-5B5E-4F9A-9F7D-123456789ABC}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Gandhi Assistant
DefaultGroupName=Gandhi Assistant
OutputDir=Output
OutputBaseFilename=GandhiAssistantSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create Desktop Shortcut"; GroupDescription: "Additional Tasks"

[Files]
Source: "*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Gandhi Assistant"; Filename: "{app}\start.bat"
Name: "{commondesktop}\Gandhi Assistant"; Filename: "{app}\start.bat"; Tasks: desktopicon

[Run]
Filename: "{app}\setup.bat"; Description: "Install Python packages and build database"; Flags: waituntilterminated
Filename: "{app}\start.bat"; Description: "Launch Gandhi Assistant"; Flags: nowait postinstall skipifsilent