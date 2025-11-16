{
  description = "Rust + Python build";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
        let pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        in {
# Dev shell: interactive environment
        devShell = pkgs.mkShell {
        name = "gng_py build";
        buildInputs = with pkgs; [
        rustc
        cargo
        maturin
        python
        vscodium
        gcc
        python.pkgs.jupyterlab
        python.pkgs.jupyter
        python.pkgs.ipykernel
        python.pkgs.notebook
        python.pkgs.numpy
        python.pkgs.pandas  
        python.pkgs.matplotlib
        ];

# Make sure the Python packages in this shell are visible to venvs
        shellHook = ''
          export PYTHONPATH=${python.sitePackages}
        echo "Tip: Now run 'python -m venv .venv && source .venv/bin/activate'"
          echo "Then you can 'pip install dist/*.whl' after building with maturin."
          '';


        };

# Declarative package: this is what nix build uses
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "gng_py";
          version = "0.1.1";
          src = ./gng_py;

# Prefetch crates according to Cargo.lock
          cargoLock = {
            lockFile = ./gng_py/Cargo.lock;
          };

            # Add Python & Maturin for building the wheel
            nativeBuildInputs = [ python pkgs.maturin ];

            buildPhase = ''
              echo "Building gng_py with maturin..."
              maturin build --release --locked -o dist --skip-auditwheel
              '';

            installPhase = ''
              mkdir -p $out/dist
              cp dist/*.whl $out/dist/
                       '';
          };
        }
  );
}


