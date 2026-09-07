{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
      python3Packages.pip
      python3Packages.numpy
      python3Packages.pandas
      python3Packages.matplotlib
      python3Packages.scikitlearn
      python3Packages.ipykernel
      python3Packages.jupyter
      python3Packages.pyzmq
      vscodium
  ];
    shellHook = ''
    echo "Activating virtualenv..."
    python3 -m venv .venv
    source .venv/bin/activate
  '';
}
