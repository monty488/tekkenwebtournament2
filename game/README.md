# game/

Put **your own** dump of Tekken Tag Tournament 2 (Xbox 360) here. Nothing in this
folder except this README is tracked by git, and nothing derived from it may be
committed or published.

Expected layout after `scripts/extract-game.ps1`:

```
game/
  iso/            original disc image (*.iso), optional once extracted
  files/          extracted disc contents (default.xex + data archives)
  update/         optional title update (default.xexp)
  scan/           output of ttt2-scan (image.bin, imports.txt, ...)
```
