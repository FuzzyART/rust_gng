{
  description = "Rust + Python dev shell with Maturin and Jupyter";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3;
        pythonPackages = python.pkgs;
      in {
        devShell = pkgs.mkShell {
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
            vscodium

            python
            pythonPackages.pip
            pythonPackages.numpy
            pythonPackages.pandas
            pythonPackages.matplotlib
            pythonPackages.scikitlearn
            pythonPackages.ipykernel
            pythonPackages.jupyter
            pythonPackages.pyzmq
          ];

          shellHook = ''
            echo "Activating virtualenv..."
            python3 -m venv .venv
            source .venv/bin/activate
            pip install maturin

            export CODELLDB_DIR="$HOME/.local/share/codelldb/extension/adapter"
            export PATH=$CODELLDB_DIR:$PATH

            if [ ! -f "$CODELLDB_DIR/codelldb" ]; then
              echo "Installing CodeLLDB..."
              mkdir -p ~/.local/share/codelldb
              curl -L -o ~/.local/share/codelldb/codelldb.zip https://github.com/vadimcn/vscode-lldb/releases/latest/download/codelldb-x86_64-linux.vsix
              unzip -o ~/.local/share/codelldb/codelldb.zip -d ~/.local/share/codelldb
            fi
          '';
        };
      });
}
