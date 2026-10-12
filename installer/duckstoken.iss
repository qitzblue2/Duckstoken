; Inno Setup script. Build with:  iscc /DAppVersion=0.2.0 /DSourceDir=dist-installer\Duckstoken installer\duckstoken.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist-installer\Duckstoken"
#endif

[Setup]
AppId={{6B1F4E0A-3C52-4D4E-9B6B-D0C57570CE01}
AppName=Duckstoken
SetupMutex=DuckstokenSetupMutex
AppVersion={#AppVersion}
AppPublisher=Duckstoken contributors
AppPublisherURL=https://github.com/qitzblue2/Duckstoken
DefaultDirName={autopf}\Duckstoken
DefaultGroupName=Duckstoken
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist-installer
OutputBaseFilename=Duckstoken-Setup
SetupIconFile=..\assets\duckstoken.ico
UninstallDisplayIcon={app}\Duckstoken.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked
Name: "startup"; Description: "Start Duckstoken when I sign in to Windows"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Duckstoken"; Filename: "{app}\Duckstoken.exe"
Name: "{autodesktop}\Duckstoken"; Filename: "{app}\Duckstoken.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Duckstoken"; \
  ValueData: """{app}\Duckstoken.exe"" --minimized"; Flags: uninsdeletevalue; Tasks: startup

[Run]
Filename: "{app}\Duckstoken.exe"; Description: "Launch Duckstoken"; Flags: nowait postinstall skipifsilent
