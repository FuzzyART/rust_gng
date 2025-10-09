{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "rust-dev-shell";

  buildInputs = with pkgs; [
    neovim
    rustc
    cargo
    rust-analyzer
    lldb
    unzip
    curl
    pkg-config
    openssl
    zlib
    cmake
    gcc
  ];

  shellHook = ''
    export CODELLDB_DIR="$HOME/.local/share/codelldb/extension/adapter"
    export PATH=$CODELLDB_DIR:$PATH

    if [ ! -f "$CODELLDB_DIR/codelldb" ]; then
      echo "Installing CodeLLDB..."
      mkdir -p ~/.local/share/codelldb
      curl -L -o ~/.local/share/codelldb/codelldb.zip https://github.com/vadimcn/vscode-lldb/releases/latest/download/codelldb-x86_64-linux.vsix
      unzip -o ~/.local/share/codelldb/codelldb.zip -d ~/.local/share/codelldb
    fi
  '';
}
