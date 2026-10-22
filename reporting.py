"""
Infrastructure for recording an analysis run: console logging, provenance and table export.
Nothing in here is specific to the OKN analysis.
"""
import os
import shutil
import subprocess
import sys

from config import resultsPath


class Tee:
    # mirror everything printed to the console into a log file.
    # utf-8 is set explicitly because the output can contain non-ASCII characters (e.g. the
    # '×' in a table caption) and the console encoding on Windows cannot be relied on.
    def __init__(self, path):
        # when the script is run repeatedly in the same interactive session, the Tee from the
        # previous run is still installed as sys.stdout. Unhook it first, otherwise the new
        # Tee wraps the old one and every line is written to the log file twice (three times
        # after the third run, and so on).
        # The check goes via the class name, not isinstance: if this module gets reloaded, the
        # instance left over from the previous run belongs to a different class object and
        # isinstance would miss it.
        while type(sys.stdout).__name__ == 'Tee':
            sys.stdout.close() # close() puts its own console back into sys.stdout
        self.console = sys.stdout
        self.file = open(path, 'w', encoding='utf-8')

    def write(self, text):
        # flush immediately: in an interactive session (Positron, Spyder, Jupyter) the
        # interpreter keeps running, so nothing flushes the buffers at the end of the script.
        # Without this, the last printed lines stay buffered and only appear once the next
        # command is run, which makes them look like output of the following run.
        self.console.write(text)
        self.console.flush()
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        # restore the previous stdout before closing, otherwise the interpreter fails when it
        # flushes sys.stdout during shutdown (exit code 120).
        # both steps are guarded because atexit collects one handler per run in an interactive
        # session, so close() can be called more than once on the same Tee.
        if sys.stdout is self:
            sys.stdout = self.console
        if not self.file.closed:
            self.file.close()


def gitExecutable():
    # git is looked up explicitly instead of letting the operating system resolve the name 'git'
    # through the PATH. A graphical IDE keeps the PATH it was started with, so a session that was
    # opened before git was installed (or before the PATH was changed) cannot call git, although
    # a freshly opened terminal on the same machine can. Falling back to the standard install
    # locations keeps the provenance line complete instead of degrading to the .git fallback.
    exe = shutil.which('git')
    if exe:
        return exe
    for candidate in (r"C:\Program Files\Git\cmd\git.exe",
                      r"C:\Program Files (x86)\Git\cmd\git.exe",
                      os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe")):
        if os.path.isfile(candidate):
            return candidate
    return 'git' # nothing found: let the call fail and be handled by the caller


def gitVersion():
    # commit the results were produced with, so a result file can be traced back to the code
    # git is run in the folder of this file, not in the working directory: an interactive
    # session (or a call from another folder) can have any working directory, which would
    # otherwise report 'unknown' or, worse, the commit of an unrelated repository.
    repo = os.path.dirname(os.path.abspath(__file__))
    git = gitExecutable()
    try:
        # check first that this folder is the repository itself. git searches upwards, so a copy
        # of the code that is not a repository (a zip download from GitHub, unpacked inside some
        # other repository) would be labelled with that surrounding repository's commit. A wrong
        # commit is worse than none, because nothing in the log would reveal it as wrong later.
        top = subprocess.check_output([git, 'rev-parse', '--show-toplevel'],
                                      cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
        if os.path.normcase(os.path.realpath(top)) != os.path.normcase(os.path.realpath(repo)):
            return 'unknown (code is not in a git repository)'
        commit = subprocess.check_output([git, 'rev-parse', '--short', 'HEAD'],
                                         cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
        isDirty = subprocess.check_output([git, 'status', '--porcelain'],
                                          cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
        return f"{commit}{' (uncommitted changes present)' if isDirty else ''}"
    except Exception as err:
        # calling git can fail even though the repository is fine, most commonly because the
        # git executable is not on the PATH of the process running the script (an interactive
        # session keeps the PATH it was started with, so an IDE opened before git was installed
        # never sees it). The commit is then read from the .git folder instead, because that is
        # the part that matters for provenance. Whether the working tree was clean cannot be
        # determined this way, so it is reported as unknown rather than silently assumed clean.
        commit = headCommit(repo)
        if commit:
            return f"{commit} (read from .git, working tree state unknown)"
        if not os.path.exists(os.path.join(repo, '.git')):
            # the usual case here is code downloaded as a zip instead of cloned: there simply is
            # no commit to report, which is not an error worth printing a traceback message for
            return 'unknown (code is not in a git repository)'
        # anything else goes into the log with its reason: a bare 'unknown' gives nothing to act on
        return f"unknown ({type(err).__name__}: {err})"


def headCommit(repo):
    # resolve HEAD without calling git. HEAD either points at a branch ("ref: refs/heads/main")
    # or, with a detached HEAD, contains the commit hash itself. A branch ref lives in its own
    # file under .git until git packs it away, after which it is only listed in packed-refs.
    try:
        gitDir = os.path.join(repo, '.git')
        # in a worktree or submodule, .git is a file pointing at the real git directory
        if os.path.isfile(gitDir):
            with open(gitDir, encoding='utf-8') as f:
                gitDir = os.path.join(repo, f.read().split(':', 1)[1].strip())
        with open(os.path.join(gitDir, 'HEAD'), encoding='utf-8') as f:
            head = f.read().strip()
        if not head.startswith('ref:'):
            return head[:7]
        ref = head.split(':', 1)[1].strip()
        refFile = os.path.join(gitDir, *ref.split('/'))
        if os.path.isfile(refFile):
            with open(refFile, encoding='utf-8') as f:
                return f.read().strip()[:7]
        with open(os.path.join(gitDir, 'packed-refs'), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0][:7]
    except Exception:
        return None
    return None


def saveTable(table, fileName, subFolder, index=True, caption=None):
    # tables that end up in the write-up are written separately, so that they can be
    # opened in Excel instead of being retyped from the log. subFolder groups them by
    # analysis section and is created on demand.
    # utf-8-sig keeps umlauts readable in Excel. If Excel puts everything in one column,
    # switch to sep=';' (German locale).
    folder = os.path.join(resultsPath, subFolder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fileName)
    table.to_csv(path, index=index, encoding='utf-8-sig')
    print(f"[saved] {path}")
    if caption is not None:
        # a csv has no place for a caption, and a table that goes into the write-up is
        # incomplete without it. The same table is therefore written a second time as
        # markdown, table and caption together, ready to be pasted into the write-up.
        markdownPath = os.path.splitext(path)[0] + '.md'
        with open(markdownPath, 'w', encoding='utf-8') as file:
            file.write(table.to_markdown(index=index))
            file.write(f"\n\n{caption}\n")
        print(f"[saved] {markdownPath}")
    return table


def saveText(text, fileName, subFolder):
    # counterpart of saveTable for plain text output such as a model summary, which is kept in
    # a file of its own when it is too long for the log
    folder = os.path.join(resultsPath, subFolder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fileName)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"[saved] {path}")
    return text
