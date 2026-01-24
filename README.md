# PartyFM Player

En desktop musik-player til PartyFM radio stream med FFT visualizer.

## Features

- Streamer live fra PartyFM (128 kbps)
- Viser "Nu spiller" info fra streamen
- FFT frekvens-visualizer der danser med musikken
- Auto-play ved opstart
- Volume kontrol

## Krav

- [Rust](https://rustup.rs/) (1.70+)

### Linux

```bash
sudo apt install libasound2-dev pkg-config libxkbcommon-dev libwayland-dev libfontconfig1-dev
```

### Windows

Ingen ekstra dependencies - bare Rust.

## Byg

### Linux

```bash
cargo build --release
```

Executable: `target/release/partyfm_player`

### Windows

```bash
cargo build --release
```

Executable: `target/release/partyfm_player.exe`

### Cross-compile til Windows (fra Linux)

```bash
# Installer toolchain
rustup target add x86_64-pc-windows-gnu
sudo apt install mingw-w64

# Byg
cargo build --release --target x86_64-pc-windows-gnu
```

Executable: `target/x86_64-pc-windows-gnu/release/partyfm_player.exe`

## Projekt struktur

```
.
├── Cargo.toml          # Dependencies
├── src/
│   └── main.rs         # Kildekode
├── top1_logo.png       # PartyFM logo
└── smiley.png          # PartyFM ikon
```

## Licens

MIT
