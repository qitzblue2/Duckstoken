"""Build stamp read by duckstoken.updater to tell which build is running.

The values committed here describe a source checkout. The release pipeline (tools/stamp_build.py, run by CI right before
PyInstaller) overwrites this whole file with the commit SHA, the build time (UTC, ISO 8601) and whether the build is the
installer flavour, so never put anything else in it.
"""

BUILD_SHA = ""
BUILD_TIME = ""
INSTALLER = False
