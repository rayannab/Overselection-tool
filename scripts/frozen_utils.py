"""
Utility to handle PyInstaller frozen state.
When the app is packaged as .exe, sys.executable points to the .exe itself,
not Python. This module provides helpers to build correct subprocess commands.
"""
import sys
import os


def is_frozen():
    """Return True if running inside a PyInstaller bundle."""
    return getattr(sys, 'frozen', False)


def get_base_dir():
    """Return the directory where scripts/executables live."""
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def build_subprocess_cmd(script_name, *args):
    """
    Build the correct subprocess command depending on frozen state.
    
    When frozen: calls <script_name>.exe directly
    When not frozen: calls sys.executable -u <script_name>.py
    """
    base = get_base_dir()
    if is_frozen():
        exe_name = script_name.replace('.py', '') + '.exe'
        return [os.path.join(base, exe_name)] + list(args)
    else:
        script_path = os.path.join(base, script_name)
        return [sys.executable, "-u", script_path] + list(args)


def get_resource_path(relative_path):
    """
    Get absolute path to a bundled resource (images, models, etc.).
    Works both in development and when frozen by PyInstaller.
    """
    if is_frozen():
        # PyInstaller extracts to _MEIPASS for --onefile,
        # or uses the exe directory for --onedir
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)
